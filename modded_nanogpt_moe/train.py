"""Active dense/MoE training, validation, compilation, logging, and profiling."""

import os
import random
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
    make_training_checkpoint,
    restore_rng_state,
    restore_training_checkpoint,
)
from .data import distributed_data_generator
from .model import GPT, eager_prefix, make_head_loss
from .optim import build_optimizers


def nsys_range(enabled: bool, name: str):
    """Return an NVTX range only while the opt-in Nsight capture is active."""
    return torch.cuda.nvtx.range(name) if enabled else nullcontext()


def read_source_snapshot():
    package_dir = Path(__file__).resolve().parent
    repository_root = package_dir.parent
    source_files = [
        package_dir / "model.py",
        package_dir / "optim.py",
        package_dir / "data.py",
        package_dir / "checkpoint.py",
        package_dir / "train.py",
        repository_root / "records/track_3_optimization/train_gpt_simple.py",
    ]
    sections = []
    for source_file in source_files:
        relative_path = source_file.relative_to(repository_root)
        sections.append(f"# ===== {relative_path} =====\n{source_file.read_text()}")
    return "\n".join(sections)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    code = read_source_snapshot()
    
    # torchrun sets these env variables
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    # this code can be run equivalently with 1, 2, 4, or 8 gpus.
    assert 8 % dist.get_world_size() == 0
    
    num_trials = int(argv[-1]) if len(argv) > 1 else 1
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
    
    seed_value = os.environ.get("SEED_OVERRIDE", "")
    seed = int(seed_value) if seed_value else None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # logging setup
    if dist.get_rank() == 0:
        os.makedirs("logs", exist_ok=True)
        run_id = (resume_checkpoint["run"]["run_id"]
                  if resume_checkpoint is not None else str(uuid.uuid4()))
        logfile = f"logs/{run_id}.txt"
        print(logfile)
    def print0(s, console=False, log=True):
        if dist.get_rank() == 0:
            if console:
                print(s)
            if log:
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
    
    val_tokens = 20 * 524288
    batch_size = 8 * 64 * 1024
    data_root = os.environ.get("DATA_ROOT", str(Path.cwd()))
    # (MBS_OVERRIDE: opt-in override for smoke tests; unset -> unchanged default of 64.
    # Global batch is preserved regardless of mbs: the gradient-accumulation loop
    # below runs len(inputs)//mbs microbatches per step, so a smaller mbs means
    # more microbatches accumulated into the same fixed batch_size, not a smaller
    # effective step.)
    mbs = int(os.environ.get("MBS_OVERRIDE", 64))
    val_loader = distributed_data_generator(
        "data/fineweb10B/fineweb_val_*.bin", val_tokens, data_root=data_root)
    val_inputs, val_targets = next(val_loader)
    
    # MLP architecture: "dense" is the original single MLP; "moe" is a SparseMoE
    # of num_experts experts, routing each token to its top_k experts.
    # (*_OVERRIDE: opt-in overrides for smoke tests; all unset -> unchanged defaults:
    # mlp_type="dense", mlp_ratio=4, num_experts=1, top_k=1, moe_backend="loop")
    mlp_type = os.environ.get("MLP_TYPE_OVERRIDE", "dense")   # "dense" or "moe"
    mlp_ratio = float(os.environ.get("MLP_RATIO_OVERRIDE", 4))
    num_experts = int(os.environ.get("NUM_EXPERTS_OVERRIDE", 1))
    top_k = int(os.environ.get("TOP_K_OVERRIDE", 1))
    normalize_topk = True
    moe_backend = os.environ.get("MOE_BACKEND_OVERRIDE", "loop")   # "loop" or "grouped_gemm"
    
    # TensorBoard logging is opt-in via a shared root; rank 0 only.
    tb_root = os.environ.get("TB_ROOT", "")
    tb_system = os.environ.get("TB_SYSTEM", "unknown")
    tensorboard_log = bool(tb_root)
    
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
    model_dim = 768
    model = GPT(vocab_size=50304, num_layers=12, model_dim=model_dim, mlp_type=mlp_type,
                num_experts=num_experts, top_k=top_k, normalize_topk=normalize_topk,
                moe_backend=moe_backend, mlp_ratio=mlp_ratio)
    local_sequences = batch_size // dist.get_world_size() // 1024
    assert local_sequences % mbs == 0
    accumulation_count = local_sequences // mbs
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
    
    environment_metadata = (
        collect_environment_metadata() if checkpointing_requested else None)
    stopped_early = False
    for trial_idx in range(num_trials):
    
    
        ########################################
        #       Init & Optim Hyperparams       #
        ########################################
    
        # we want to minimize this while still reaching 3.28 val loss
        # (TRAIN_STEPS_OVERRIDE: opt-in override for short smoke tests; unset -> unchanged default)
        train_steps = int(os.environ.get("TRAIN_STEPS_OVERRIDE", 3250))
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
        optimizers = build_optimizers(model)
    
        # learning rate schedule: stable then decay
        def set_hparams(step, cooldown_frac=0.7):
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
                "vocab_size": 50304,
                "num_layers": 12,
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
                "sequence_length": 1024,
                "global_batch_tokens": batch_size,
                "microbatch_sequences": mbs,
                "accumulation_count": accumulation_count,
                "validation_tokens": val_tokens,
                "total_steps": train_steps,
                "cooldown_fraction": 0.7,
                "training_shard_pattern": "data/fineweb10B/fineweb_train_*.bin",
                "validation_shard_pattern": "data/fineweb10B/fineweb_val_*.bin",
            },
            "optimizers": {
                "adamw": {
                    "group_lrs": [0.7, 0.004, 0.015],
                    "betas": [0.8, 0.95],
                    "eps": 1e-10,
                    "weight_decay": 0.001,
                    "fused": True,
                },
                "muon": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95},
            },
            "datasets": {
                "validation_shards": val_loader.shard_identities,
            },
            "seed_override": seed,
        }
    
    
        ########################################
        #        Training and Validation       #
        ########################################
    
        train_loader = distributed_data_generator(
            "data/fineweb10B/fineweb_train_*.bin", batch_size, data_root=data_root)
    
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
            tb_dir = os.path.join(
                tb_root, "modded-nanogpt-moe", tb_system, str(run_id), f"trial_{trial_idx}")
            print0(f"TensorBoard event directory: {tb_dir}", console=True)
            writer_kwargs = {"log_dir": tb_dir}
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
        # start the clock
        dist.barrier()
        t0 = time.perf_counter() - current_segment_time
        nsys_capture_active = False
        for step in range(completed_updates, train_steps + 1):
    
            # --------------- VALIDATION SECTION -----------------
            val_step_freq = 125 if step / train_steps < 0.9 else 25
            if step == train_steps or step % val_step_freq == 0:
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
    
            if step == train_steps:
                break
    
            # --------------- TRAINING SECTION -----------------
            if nsys_profile and step == nsys_warmup_steps:
                torch.cuda.synchronize()
                print0(f"Nsight Systems capture starting before update {step + 1}", console=True)
                torch.cuda.profiler.start()
                nsys_capture_active = True
    
            with nsys_range(nsys_capture_active, f"optimizer_step.update_{step + 1}"):
                with nsys_range(nsys_capture_active, "data_preparation"):
                    inputs, targets = next(train_loader)
                # accumulate across microbatches in case we are running with fewer than 8 gpus
                assert len(inputs) % mbs == 0
                for i in range(len(inputs) // mbs):
                    with nsys_range(nsys_capture_active, f"forward.microbatch_{i}"):
                        loss = run_forward(
                            inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
                    with nsys_range(nsys_capture_active, f"backward.microbatch_{i}"):
                        loss.backward()
                    del loss
                for name, p in model.named_parameters():
                    assert p.grad is not None, name
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
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
    
            if (nsys_capture_active
                    and step + 1 == nsys_warmup_steps + nsys_active_steps):
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                nsys_capture_active = False
                print0(f"Nsight Systems capture ended after update {step + 1}", console=True)
            approx_training_time = training_time + (time.perf_counter() - t0)
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
    
        if writer is not None:
            writer.close()
        if stopped_early:
            break
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
