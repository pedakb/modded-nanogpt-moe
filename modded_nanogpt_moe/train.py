"""Active dense/MoE training, validation, benchmarking, logging, and profiling."""

import json
import os
import random
import statistics
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_save_checkpoint,
    collect_environment_metadata,
    make_repro_diagnostic,
    make_training_checkpoint,
    restore_rng_state,
    restore_training_checkpoint,
    save_repro_diagnostic,
)
from .config import parse_train_args
from .data import distributed_data_generator
from .model import GPT, eager_prefix, make_head_loss, resolve_mlp_hidden_dim
from .optim import build_optimizers


def nsys_range(enabled: bool, name: str):
    """Return an NVTX range only while the opt-in Nsight capture is active."""
    return torch.cuda.nvtx.range(name) if enabled else nullcontext()


def benchmark_settings_from_environment(environment=None):
    """Resolve the opt-in training benchmark controls without touching CUDA."""
    environment = os.environ if environment is None else environment
    enabled_value = environment.get("TRAINING_BENCHMARK", "0")
    if enabled_value not in ("0", "1"):
        raise ValueError(
            f"TRAINING_BENCHMARK must be 0 or 1, got {enabled_value!r}")
    enabled = enabled_value == "1"
    if not enabled:
        return {
            "enabled": False,
            "warmup_updates": 10,
            "measured_updates": 30,
        }
    warmup_updates = int(environment.get("BENCHMARK_WARMUP_UPDATES", "10"))
    measured_updates = int(environment.get("BENCHMARK_MEASURED_UPDATES", "30"))
    if warmup_updates < 0:
        raise ValueError("BENCHMARK_WARMUP_UPDATES must be nonnegative")
    if measured_updates <= 0:
        raise ValueError("BENCHMARK_MEASURED_UPDATES must be positive")
    return {
        "enabled": enabled,
        "warmup_updates": warmup_updates,
        "measured_updates": measured_updates,
    }


def summarize_training_benchmark(update_seconds, tokens_per_update,
                                 peak_allocated_bytes, peak_reserved_bytes):
    """Return complete-update latency, throughput, and memory statistics."""
    if not update_seconds or any(duration <= 0 for duration in update_seconds):
        raise ValueError("benchmark update durations must be positive and nonempty")
    mean_seconds = statistics.fmean(update_seconds)
    return {
        "mean_ms_per_update": 1000 * mean_seconds,
        "median_ms_per_update": 1000 * statistics.median(update_seconds),
        "tokens_per_second": tokens_per_update / mean_seconds,
        "peak_allocated_gib": peak_allocated_bytes / 2**30,
        "peak_reserved_gib": peak_reserved_bytes / 2**30,
    }


def tensorboard_run_directory(root, system, run_name):
    return Path(root) / "modded-nanogpt-moe" / system / str(run_name)


def require_unused_tensorboard_run_directory(root, system, run_name):
    run_directory = tensorboard_run_directory(root, system, run_name)
    if run_directory.exists():
        raise FileExistsError(
            f"TensorBoard run directory already exists: {run_directory}. "
            "Choose a new run_name or remove the existing directory explicitly.")
    return run_directory


def read_source_snapshot():
    package_dir = Path(__file__).resolve().parent
    repository_root = package_dir.parent
    source_files = [
        package_dir / "model.py",
        package_dir / "optim.py",
        package_dir / "data.py",
        package_dir / "checkpoint.py",
        package_dir / "config.py",
        package_dir / "train.py",
    ]
    sections = []
    for source_file in source_files:
        relative_path = source_file.relative_to(repository_root)
        sections.append(f"# ===== {relative_path} =====\n{source_file.read_text()}")
    return "\n".join(sections)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    experiment_config, config_path = parse_train_args(argv[1:])
    code = read_source_snapshot()

    # torchrun sets these environment variables.
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    # this code can be run equivalently with 1, 2, 4, or 8 gpus.
    assert 8 % dist.get_world_size() == 0
    
    num_trials = experiment_config["num_trials"]
    benchmark = benchmark_settings_from_environment()
    if benchmark["enabled"]:
        if dist.get_world_size() != 1:
            raise ValueError("TRAINING_BENCHMARK=1 currently requires exactly one GPU")
        if num_trials != 1:
            raise ValueError("TRAINING_BENCHMARK=1 currently requires exactly one trial")
    repro_diagnostics_dir = os.environ.get("REPRO_DIAGNOSTICS_DIR", "")
    if repro_diagnostics_dir:
        if dist.get_world_size() != 1:
            raise ValueError("REPRO_DIAGNOSTICS_DIR currently requires exactly one GPU")
        if num_trials != 1:
            raise ValueError("REPRO_DIAGNOSTICS_DIR currently requires exactly one trial")
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "")
    checkpoint_interval = int(os.environ.get("CHECKPOINT_INTERVAL", 0))
    resume_checkpoint_path = os.environ.get("RESUME_CHECKPOINT", "")
    stop_after_value = os.environ.get("STOP_AFTER_COMPLETED_UPDATES", "")
    stop_after_updates = int(stop_after_value) if stop_after_value else None
    checkpointing_requested = bool(
        checkpoint_dir or checkpoint_interval or resume_checkpoint_path
        or stop_after_updates is not None)
    if checkpoint_interval < 0:
        raise ValueError("CHECKPOINT_INTERVAL must be nonnegative")
    if checkpointing_requested:
        if dist.get_world_size() != 1:
            raise ValueError("checkpoint/resume currently requires exactly one GPU")
        if num_trials != 1:
            raise ValueError("checkpoint/resume currently requires exactly one trial")
        if not checkpoint_dir:
            raise ValueError(
                "CHECKPOINT_DIR is required when checkpointing, resuming, or stopping early")
    if benchmark["enabled"] and checkpointing_requested:
        raise ValueError(
            "TRAINING_BENCHMARK cannot be combined with checkpoint/resume controls")
    if benchmark["enabled"] and repro_diagnostics_dir:
        raise ValueError(
            "TRAINING_BENCHMARK cannot be combined with reproducibility diagnostics")
    if stop_after_updates is not None and stop_after_updates <= 0:
        raise ValueError("STOP_AFTER_COMPLETED_UPDATES must be positive")
    
    resume_checkpoint = None
    if resume_checkpoint_path:
        resume_checkpoint = torch.load(
            resume_checkpoint_path, map_location="cpu", weights_only=False)
        if resume_checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format version: "
                f"{resume_checkpoint.get('format_version')}")
        if resume_checkpoint.get("run", {}).get("trial_idx") != 0:
            raise ValueError("resume currently supports only trial_idx=0")
    
    seed = experiment_config["seed"]
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    tb_root = os.environ.get("TB_ROOT", "")
    tb_system = os.environ.get("TB_SYSTEM", "unknown")
    tensorboard_log = bool(tb_root)
    if benchmark["enabled"] and tensorboard_log:
        raise ValueError("TRAINING_BENCHMARK requires TensorBoard to be disabled")

    requested_run_name = experiment_config["run_name"]
    run_id = (resume_checkpoint["run"]["run_id"]
              if resume_checkpoint is not None
              else requested_run_name or str(uuid.uuid4()))
    if (resume_checkpoint is not None and requested_run_name is not None
            and requested_run_name != run_id):
        raise ValueError(
            f"config run_name {requested_run_name!r} does not match checkpoint "
            f"run identity {run_id!r}")
    experiment_config["run_name"] = run_id
    if tensorboard_log and resume_checkpoint is None:
        require_unused_tensorboard_run_directory(tb_root, tb_system, run_id)

    # logging setup
    logfile = None
    if dist.get_rank() == 0 and not benchmark["enabled"]:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)
    def print0(s, console=False, log=True):
        if dist.get_rank() == 0:
            if console:
                print(s)
            if log and logfile is not None:
                with open(logfile, "a") as f:
                    print(s, file=f)
    
    # we begin by logging this file itself
    print0(code)
    print0("="*100)
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
           + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
    print0("="*100)
    if checkpointing_requested:
        print0(
            f"checkpointing: directory={checkpoint_dir} interval={checkpoint_interval} "
            f"resume={resume_checkpoint_path or 'none'} "
            f"stop_after={stop_after_updates if stop_after_updates is not None else 'none'}",
            console=True,
        )
    if repro_diagnostics_dir:
        print0(
            f"reproducibility diagnostics: directory={repro_diagnostics_dir}",
            console=True,
        )
    if benchmark["enabled"]:
        print0(
            "Training benchmark enabled: "
            f"warmup_updates={benchmark['warmup_updates']} "
            f"measured_updates={benchmark['measured_updates']}",
            console=True,
        )
    
    model_config = experiment_config["model"]
    training_config = experiment_config["training"]
    optimizer_config = experiment_config["optimizers"]
    val_tokens = training_config["validation_tokens"]
    batch_size = training_config["global_batch_tokens"]
    sequence_length = training_config["sequence_length"]
    training_shard_pattern = training_config["training_shard_pattern"]
    validation_shard_pattern = training_config["validation_shard_pattern"]
    data_root = os.environ.get("DATA_ROOT", str(Path.cwd()))
    # (MBS_OVERRIDE: opt-in override for smoke tests; unset -> unchanged default of 64.
    # Global batch is preserved regardless of mbs: the gradient-accumulation loop
    # below runs len(inputs)//mbs microbatches per step, so a smaller mbs means
    # more microbatches accumulated into the same fixed batch_size, not a smaller
    # effective step.)
    mbs = training_config["microbatch_sequences"]

    # MLP architecture: "dense" is the original single MLP; "moe" is a SparseMoE
    # of num_experts experts, routing each token to its top_k experts.
    # (*_OVERRIDE: opt-in overrides for smoke tests; all unset -> unchanged defaults:
    # mlp_type="dense", mlp_ratio=4, num_experts=1, top_k=1, moe_backend="loop")
    mlp_type = model_config["mlp_type"]
    mlp_ratio = model_config["mlp_ratio"]
    num_experts = model_config["num_experts"]
    top_k = model_config["top_k"]
    normalize_topk = model_config["normalize_topk"]
    moe_backend = model_config["moe_backend"]
    model_dim = model_config["model_dim"]
    hidden_dim = resolve_mlp_hidden_dim(model_dim, mlp_ratio)
    if batch_size % dist.get_world_size():
        raise ValueError(
            "training.global_batch_tokens must be divisible by world size")
    local_tokens = batch_size // dist.get_world_size()
    if local_tokens % sequence_length:
        raise ValueError(
            "training.global_batch_tokens must be divisible by world size and "
            "training.sequence_length")
    local_sequences = local_tokens // sequence_length
    if local_sequences % mbs:
        raise ValueError(
            "local sequences per update must be divisible by "
            "training.microbatch_sequences")
    accumulation_count = local_sequences // mbs
    experiment_config["model"]["hidden_dim"] = hidden_dim
    experiment_config["training"]["accumulation_count"] = accumulation_count
    print0(
        "Resolved experiment config"
        + (f" ({config_path})" if config_path is not None else " (built-in defaults)")
        + ":\n"
        + json.dumps(experiment_config, indent=2, sort_keys=True),
        console=True,
    )

    val_loader = None
    val_inputs = val_targets = None
    if not benchmark["enabled"]:
        val_loader = distributed_data_generator(
            validation_shard_pattern, val_tokens, seq_len=sequence_length,
            data_root=data_root)
        val_inputs, val_targets = next(val_loader)

    nsys_profile_value = os.environ.get("NSYS_PROFILE", "0")
    if nsys_profile_value not in ("0", "1"):
        raise ValueError(f"NSYS_PROFILE must be 0 or 1, got {nsys_profile_value!r}")
    nsys_profile = nsys_profile_value == "1"
    nsys_warmup_steps = 10
    nsys_active_steps = 2
    if nsys_profile:
        nsys_warmup_steps = int(os.environ.get("NSYS_WARMUP_STEPS", nsys_warmup_steps))
        nsys_active_steps = int(os.environ.get("NSYS_ACTIVE_STEPS", nsys_active_steps))
        if dist.get_world_size() != 1:
            raise ValueError("NSYS_PROFILE=1 currently requires exactly one GPU")
        if num_trials != 1:
            raise ValueError("NSYS_PROFILE=1 currently requires exactly one trial")
        if nsys_warmup_steps < 0:
            raise ValueError("NSYS_WARMUP_STEPS must be nonnegative")
        if nsys_active_steps <= 0:
            raise ValueError("NSYS_ACTIVE_STEPS must be positive")
        print0(
            f"Nsight Systems profiling enabled: warmup_updates={nsys_warmup_steps} "
            f"captured_updates={nsys_warmup_steps + 1}-"
            f"{nsys_warmup_steps + nsys_active_steps}",
            console=True,
        )
    if nsys_profile and checkpointing_requested:
        raise ValueError(
            "combining NSYS_PROFILE with checkpoint/resume is not yet supported")
    if benchmark["enabled"] and nsys_profile:
        raise ValueError("TRAINING_BENCHMARK cannot be combined with NSYS_PROFILE")

    data_root_path = Path(data_root).expanduser().resolve()
    runtime_config = {
        "config_path": (
            str(Path(config_path).expanduser().resolve())
            if config_path is not None else None
        ),
        "run_name": run_id,
        "system": tb_system,
        "data": {
            "root": str(data_root_path),
            "training_shards": [
                str(path.resolve())
                for path in sorted(data_root_path.glob(training_shard_pattern))
            ],
            "validation_shards": [
                str(path.resolve())
                for path in sorted(data_root_path.glob(validation_shard_pattern))
            ],
        },
        "tensorboard": {
            "enabled": tensorboard_log,
            "root": (
                str(Path(tb_root).expanduser().resolve()) if tensorboard_log else None
            ),
            "run_directory": (
                str(tensorboard_run_directory(tb_root, tb_system, run_id).resolve())
                if tensorboard_log else None
            ),
        },
        "checkpoint": {
            "enabled": checkpointing_requested,
            "directory": (
                str(Path(checkpoint_dir).expanduser().resolve())
                if checkpoint_dir else None
            ),
            "interval": checkpoint_interval,
            "resume": (
                str(Path(resume_checkpoint_path).expanduser().resolve())
                if resume_checkpoint_path else None
            ),
            "stop_after_completed_updates": stop_after_updates,
        },
        "profiling": {
            "nsys_enabled": nsys_profile,
            "warmup_steps": nsys_warmup_steps,
            "active_steps": nsys_active_steps,
        },
        "distributed": {
            "world_size": dist.get_world_size(),
            "device": str(device),
        },
    }
    if benchmark["enabled"]:
        runtime_config["benchmark"] = benchmark
    startup_environment = collect_environment_metadata()
    print0(
        "Resolved runtime settings:\n"
        + json.dumps(
            {"runtime": runtime_config, "environment": startup_environment},
            indent=2,
            sort_keys=True,
        ),
        console=True,
    )
    model = GPT(vocab_size=model_config["vocab_size"],
                num_layers=model_config["num_layers"], model_dim=model_dim, mlp_type=mlp_type,
                num_experts=num_experts, top_k=top_k, normalize_topk=normalize_topk,
                moe_backend=moe_backend, mlp_ratio=mlp_ratio)
    assert model.hidden_dim == hidden_dim
    print0(
        f"configuration: model_dim={model.model_dim} mlp_ratio={float(model.mlp_ratio):g} "
        f"hidden_dim={model.hidden_dim} model_type={mlp_type} moe_backend={moe_backend} "
        f"E={num_experts} k={top_k} microbatch={mbs} "
        f"global_batch={batch_size} accumulation_count={accumulation_count} "
        f"trial_count={num_trials}",
        console=True,
    )
    model.cuda()
    
    # Compilation strategy, constructed once, outside the trial/step loops:
    # - dense: unchanged whole-model compile (existing baseline, untouched).
    # - moe: the expert dispatch loop forces a graph break inside GPT.forward's
    #   own block loop, which skips compiling the WHOLE frame (including the
    #   much larger head/loss tail) for either mlp_type. Instead, run the
    #   block loop eagerly (dispatch/routing unchanged) and compile only the
    #   head/loss tail, in isolation, with fullgraph=True so any future
    #   internal break in that region fails loudly instead of silently
    #   falling back. Verified on an A100 (eval peak 6.944 GiB vs an OOM).
    if mlp_type == "dense":
        model.compile(dynamic=False)
        def run_forward(inputs, targets):
            return model(inputs, targets)
    elif mlp_type == "moe":
        compiled_head_loss = torch.compile(make_head_loss(model), fullgraph=True, dynamic=False)
        def run_forward(inputs, targets):
            x = eager_prefix(model, inputs)
            return compiled_head_loss(x, targets)
    else:
        raise ValueError(f"unknown mlp_type: {mlp_type!r}")
    
    environment_metadata = None
    if checkpointing_requested:
        environment_metadata = dict(startup_environment)
        environment_metadata["runtime"] = runtime_config
    stopped_early = False
    for trial_idx in range(num_trials):
    
    
        ########################################
        #       Init & Optim Hyperparams       #
        ########################################
    
        # we want to minimize this while still reaching 3.28 val loss
        train_steps = training_config["total_steps"]
        if train_steps <= 0:
            raise ValueError("TRAIN_STEPS_OVERRIDE must be positive")
        if stop_after_updates is not None and stop_after_updates > train_steps:
            raise ValueError(
                "STOP_AFTER_COMPLETED_UPDATES cannot exceed the total training steps")
        if nsys_profile and train_steps < nsys_warmup_steps + nsys_active_steps:
            raise ValueError(
                f"NSYS_PROFILE capture ends after update "
                f"{nsys_warmup_steps + nsys_active_steps}, but TRAIN_STEPS_OVERRIDE "
                f"requests only {train_steps} updates")
        benchmark_total_updates = (
            benchmark["warmup_updates"] + benchmark["measured_updates"])
        if benchmark["enabled"] and benchmark_total_updates > train_steps:
            raise ValueError(
                "benchmark warmup plus measured updates cannot exceed the config's "
                "training.total_steps schedule horizon")
    
        # initialize model parameters
        for name, p in model.named_parameters():
            w = p.data
            if name.endswith("weight"):
                if "proj" in name:
                    w.zero_()
                elif "embed" in name:
                    w.normal_()  # default torch init
                else:
                    w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
            elif name.endswith("bias"):
                w.zero_()
            elif name.endswith("gains"):
                w.normal_(mean=1, std=0)
            else:
                raise Exception(f"Uninitialized parameter: {name}")
    
        # create the optimizer(s)
        optimizers = build_optimizers(model, optimizer_config)
    
        # learning rate schedule: stable then decay
        def set_hparams(step, cooldown_frac=training_config["cooldown_fraction"]):
            progress = step / train_steps
            assert 0 <= progress < 1
            if progress < 1 - cooldown_frac:
                eta = 1.0
            else:
                eta = (1 - progress) / cooldown_frac
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * eta
    
        resolved_config = {
            "model": {
                "vocab_size": model_config["vocab_size"],
                "num_layers": model_config["num_layers"],
                "model_dim": model.model_dim,
                "mlp_type": mlp_type,
                "mlp_ratio": float(model.mlp_ratio),
                "hidden_dim": model.hidden_dim,
                "num_experts": num_experts,
                "top_k": top_k,
                "normalize_topk": normalize_topk,
                "moe_backend": moe_backend,
            },
            "training": {
                "sequence_length": sequence_length,
                "global_batch_tokens": batch_size,
                "microbatch_sequences": mbs,
                "accumulation_count": accumulation_count,
                "validation_tokens": val_tokens,
                "total_steps": train_steps,
                "cooldown_fraction": training_config["cooldown_fraction"],
                "training_shard_pattern": training_shard_pattern,
                "validation_shard_pattern": validation_shard_pattern,
            },
            "optimizers": optimizer_config,
            "datasets": {
                "validation_shards": (
                    val_loader.shard_identities if val_loader is not None else []),
            },
            "seed_override": seed,
        }
    
    
        ########################################
        #        Training and Validation       #
        ########################################
    
        train_loader = distributed_data_generator(
            training_shard_pattern, batch_size, seq_len=sequence_length,
            data_root=data_root)

        repro_runtime = {
            "seed_override": seed,
            "stop_after_completed_updates": stop_after_updates,
            "train_steps": train_steps,
            "mlp_type": mlp_type,
            "moe_backend": moe_backend,
            "torch_initial_seed": torch.initial_seed(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }

        def save_repro(stage, completed, filename, **extra):
            if not repro_diagnostics_dir:
                return
            payload = make_repro_diagnostic(
                stage,
                model,
                optimizers,
                train_loader,
                completed,
                repro_runtime,
                **extra,
            )
            path = save_repro_diagnostic(payload, repro_diagnostics_dir, filename)
            print0(f"Saved reproducibility diagnostic: {path}", console=True)
    
        completed_updates = 0
        training_time = 0.0
        current_segment_time = 0.0
        last_val_step = 0
        if resume_checkpoint is not None:
            restored = restore_training_checkpoint(
                resume_checkpoint,
                resolved_config,
                model,
                optimizers,
                train_loader,
                train_steps,
                batch_size,
                stop_after_updates,
            )
            completed_updates = restored["completed_updates"]
            training_time = restored["training_time"]
            current_segment_time = restored["current_segment_time"]
            last_val_step = restored["last_val_step"]
            print0(
                f"Resuming {run_id} from {resume_checkpoint_path}: "
                f"completed_updates={completed_updates} "
                f"processed_training_tokens={resume_checkpoint['processed_training_tokens']}",
                console=True,
            )
    
        # tensorboard writer: rank 0 only, one run directory per trial, disabled by default
        writer = None
        if tensorboard_log and dist.get_rank() == 0:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = tensorboard_run_directory(
                tb_root, tb_system, run_id) / f"trial_{trial_idx}"
            print0(f"TensorBoard event directory: {tb_dir}", console=True)
            writer_kwargs = {"log_dir": str(tb_dir)}
            if resume_checkpoint is not None:
                # Hide any stale events at or after the restored update. This keeps a
                # reused run directory coherent if work progressed past the checkpoint.
                writer_kwargs["purge_step"] = completed_updates
            writer = SummaryWriter(**writer_kwargs)
            if resume_checkpoint is not None and completed_updates > 0:
                restored_elapsed = training_time + current_segment_time
                writer.add_scalar(
                    "perf/approx_training_time_s", restored_elapsed, completed_updates)
                writer.add_scalar(
                    "perf/step_avg_ms",
                    1000 * restored_elapsed / completed_updates,
                    completed_updates,
                )
                writer.flush()
    
        for p in model.parameters():
            dist.broadcast(p.detach(), 0)
        if resume_checkpoint is not None:
            # Model/optimizer construction and writer setup may consume randomness.
            # Restore last so the next training update sees the saved RNG streams.
            restore_rng_state(resume_checkpoint["rng"])
            save_repro(
                "after_restore",
                completed_updates,
                f"after_restore_update_{completed_updates}.pt",
            )
        else:
            save_repro("initialized", 0, "initialized.pt")
        # start the clock
        dist.barrier()
        t0 = time.perf_counter() - current_segment_time
        nsys_capture_active = False
        benchmark_update_seconds = []
        final_loop_step = benchmark_total_updates if benchmark["enabled"] else train_steps
        for step in range(completed_updates, final_loop_step + 1):
    
            # --------------- VALIDATION SECTION -----------------
            val_step_freq = 125 if step / train_steps < 0.9 else 25
            if (not benchmark["enabled"]
                    and (step == train_steps or step % val_step_freq == 0)):
                # stop the clock
                dist.barrier()
                time_since_last_val = time.perf_counter() - t0
                step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
                last_val_step = step
                training_time += time_since_last_val
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    assert len(val_inputs) % mbs == 0
                    for i in range(len(val_inputs) // mbs):
                        val_loss += run_forward(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                val_loss /= val_tokens
                print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                       + f" step_avg:{1000*step_avg:.2f}ms"
                       + f" mem_alloc:{torch.cuda.memory_allocated()/2**30:.3f}GiB"
                       + f" mem_alloc_peak:{torch.cuda.max_memory_allocated()/2**30:.3f}GiB"
                       + f" mem_reserved:{torch.cuda.memory_reserved()/2**30:.3f}GiB"
                       + f" mem_reserved_peak:{torch.cuda.max_memory_reserved()/2**30:.3f}GiB",
                       console=True)
                if writer is not None:
                    writer.add_scalar("eval/val_loss", float(val_loss), step)
                    writer.add_scalar("perf/step_avg_ms", 1000 * step_avg, step)
                    writer.flush()
                model.train()
                # start the clock again
                dist.barrier()
                t0 = time.perf_counter()
    
            if step == final_loop_step:
                break
    
            # --------------- TRAINING SECTION -----------------
            if nsys_profile and step == nsys_warmup_steps:
                torch.cuda.synchronize()
                print0(f"Nsight Systems capture starting before update {step + 1}", console=True)
                torch.cuda.profiler.start()
                nsys_capture_active = True

            benchmark_step_started = None
            if benchmark["enabled"] and step == benchmark["warmup_updates"]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                print0(
                    f"Benchmark measurement starting before update {step + 1}",
                    console=True,
                )
            if benchmark["enabled"] and step >= benchmark["warmup_updates"]:
                torch.cuda.synchronize()
                benchmark_step_started = time.perf_counter()
    
            with nsys_range(nsys_capture_active, f"optimizer_step.update_{step + 1}"):
                with nsys_range(nsys_capture_active, "data_preparation"):
                    inputs, targets = next(train_loader)
                if repro_diagnostics_dir and step == 0:
                    save_repro(
                        "first_training_batch",
                        0,
                        "first_training_batch.pt",
                        include_training_state=False,
                        inputs=inputs,
                        targets=targets,
                    )
                if repro_diagnostics_dir and step == 1:
                    save_repro(
                        "before_update_2_forward",
                        1,
                        "before_update_2_forward.pt",
                        inputs=inputs,
                        targets=targets,
                    )
                # accumulate across microbatches in case we are running with fewer than 8 gpus
                assert len(inputs) % mbs == 0
                diagnostic_update = (
                    step + 1 if repro_diagnostics_dir and step in (0, 1) else None)
                diagnostic_losses = [] if diagnostic_update is not None else None
                for i in range(len(inputs) // mbs):
                    with nsys_range(nsys_capture_active, f"forward.microbatch_{i}"):
                        loss = run_forward(
                            inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
                        if diagnostic_losses is not None:
                            diagnostic_losses.append(loss.detach())
                    with nsys_range(nsys_capture_active, f"backward.microbatch_{i}"):
                        loss.backward()
                    del loss
                for name, p in model.named_parameters():
                    assert p.grad is not None, name
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                diagnostic_gradients = None
                if diagnostic_losses is not None:
                    diagnostic_gradients = {
                        name: parameter.grad.detach().cpu().clone()
                        for name, parameter in model.named_parameters()
                    }
                    diagnostic_losses = torch.stack(diagnostic_losses).cpu()
                # set optimization hyperparameters and take a step
                set_hparams(step)
                if writer is not None:
                    for opt_idx, opt in enumerate(optimizers):
                        for grp_idx, group in enumerate(opt.param_groups):
                            writer.add_scalar(
                                f"optim/lr_opt{opt_idx}_group{grp_idx}", group["lr"], step)
                for opt_idx, opt in enumerate(optimizers):
                    optimizer_name = type(opt).__name__
                    with nsys_range(
                            nsys_capture_active,
                            f"optimizer_update.index_{opt_idx}.{optimizer_name}"):
                        opt.step()
                model.zero_grad(set_to_none=True)
                if diagnostic_gradients is not None:
                    stage = (
                        "after_first_update"
                        if diagnostic_update == 1
                        else "after_update_2"
                    )
                    filename = (
                        "after_first_update.pt"
                        if diagnostic_update == 1
                        else "after_update_2.pt"
                    )
                    save_repro(
                        stage,
                        diagnostic_update,
                        filename,
                        losses=diagnostic_losses,
                        gradients=diagnostic_gradients,
                    )

            if benchmark_step_started is not None:
                torch.cuda.synchronize()
                benchmark_update_seconds.append(
                    time.perf_counter() - benchmark_step_started)
    
            if (nsys_capture_active
                    and step + 1 == nsys_warmup_steps + nsys_active_steps):
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                nsys_capture_active = False
                print0(f"Nsight Systems capture ended after update {step + 1}", console=True)
            approx_training_time = training_time + (time.perf_counter() - t0)
            if not benchmark["enabled"]:
                print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
                       + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms"
                       + f" mem_alloc:{torch.cuda.memory_allocated()/2**30:.3f}GiB"
                       + f" mem_alloc_peak:{torch.cuda.max_memory_allocated()/2**30:.3f}GiB"
                       + f" mem_reserved:{torch.cuda.memory_reserved()/2**30:.3f}GiB"
                       + f" mem_reserved_peak:{torch.cuda.max_memory_reserved()/2**30:.3f}GiB",
                       console=True, log=False)
            if writer is not None:
                writer.add_scalar("perf/approx_training_time_s", approx_training_time, step + 1)
                writer.add_scalar("perf/step_avg_ms", 1000 * approx_training_time / (step + 1), step + 1)
                writer.flush()
    
            completed_updates = step + 1
            save_due = bool(checkpoint_dir) and (
                completed_updates == train_steps
                or (checkpoint_interval > 0
                    and completed_updates % checkpoint_interval == 0)
                or completed_updates == stop_after_updates)
            if save_due:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        raise RuntimeError(
                            f"refusing to checkpoint before gradients are cleared: {name}")
                save_repro(
                    "before_checkpoint_save",
                    completed_updates,
                    f"before_save_update_{completed_updates}.pt",
                )
                checkpoint_started = time.perf_counter()
                checkpoint = make_training_checkpoint(
                    model=model,
                    optimizers=optimizers,
                    completed_updates=completed_updates,
                    batch_size=batch_size,
                    resolved_config=resolved_config,
                    train_loader=train_loader,
                    run_id=run_id,
                    trial_idx=trial_idx,
                    training_time=training_time,
                    current_segment_time=checkpoint_started - t0,
                    last_val_step=last_val_step,
                    environment_metadata=environment_metadata,
                )
                saved_path = atomic_save_checkpoint(checkpoint, checkpoint_dir)
                checkpoint_duration = time.perf_counter() - checkpoint_started
                # Checkpoint I/O is bookkeeping rather than training time.
                t0 += checkpoint_duration
                print0(
                    f"Saved checkpoint after update {completed_updates}: {saved_path}",
                    console=True,
                )
    
            if completed_updates == stop_after_updates:
                if nsys_capture_active:
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                    nsys_capture_active = False
                    print0(
                        f"Nsight Systems capture ended early after update {completed_updates}",
                        console=True,
                    )
                print0(
                    f"Stopped cleanly after requested update {completed_updates}; "
                    f"schedule horizon remains {train_steps}",
                    console=True,
                )
                stopped_early = True
                break

        if benchmark["enabled"]:
            if len(benchmark_update_seconds) != benchmark["measured_updates"]:
                raise RuntimeError(
                    "benchmark measured an unexpected number of optimizer updates: "
                    f"expected {benchmark['measured_updates']}, "
                    f"got {len(benchmark_update_seconds)}")
            summary = summarize_training_benchmark(
                benchmark_update_seconds,
                batch_size,
                torch.cuda.max_memory_allocated(device),
                torch.cuda.max_memory_reserved(device),
            )
            print0(
                "Training benchmark result:\n"
                f"  model_dim={model.model_dim} hidden_dim={model.hidden_dim} "
                f"E={num_experts} k={top_k} backend={moe_backend}\n"
                f"  warmup_updates={benchmark['warmup_updates']} "
                f"measured_updates={benchmark['measured_updates']}\n"
                f"  mean_ms/update={summary['mean_ms_per_update']:.3f}\n"
                f"  median_ms/update={summary['median_ms_per_update']:.3f}\n"
                f"  tokens/sec={summary['tokens_per_second']:.3f}\n"
                f"  peak_allocated_GiB={summary['peak_allocated_gib']:.3f}\n"
                f"  peak_reserved_GiB={summary['peak_reserved_gib']:.3f}",
                console=True,
            )
    
        if writer is not None:
            writer.close()
        if stopped_early:
            break
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
