import io
import os
import subprocess
import tomllib
import types
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from modded_nanogpt_moe.checkpoint import restore_training_checkpoint
from modded_nanogpt_moe.config import load_experiment_config, parse_train_args
from modded_nanogpt_moe.model import GPT, initialize_model_parameters, resolve_mlp_hidden_dim
from modded_nanogpt_moe.optim import Muon, build_optimizers
from modded_nanogpt_moe.train import (
    benchmark_settings_from_environment,
    checkpoint_directory_from_environment,
    main,
    read_source_snapshot,
    require_unused_tensorboard_run_directory,
    summarize_training_benchmark,
    tensorboard_run_directory,
)


BASELINE_COMMIT = "372221a"
BASELINE_PATH = "records/track_3_optimization/train_gpt_simple.py"


@pytest.fixture(scope="module")
def baseline_module():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:{BASELINE_PATH}"],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"committed extraction baseline is unavailable: {result.stderr}")
    module = types.ModuleType("active_trainer_before_package_extraction")
    module.__file__ = str(repository_root / BASELINE_PATH)
    exec(compile(result.stdout, module.__file__, "exec"), module.__dict__)
    return module


def _initialize_like_trainer(model):
    for name, parameter in model.named_parameters():
        value = parameter.data
        if name.endswith("weight"):
            if "proj" in name:
                value.zero_()
            elif "embed" in name:
                value.normal_()
            else:
                value.normal_(std=0.33**0.5 / value.size(-1)**0.5)
        elif name.endswith("bias"):
            value.zero_()
        elif name.endswith("gains"):
            value.normal_(mean=1, std=0)
        else:
            raise AssertionError(f"unexpected parameter: {name}")


def _legacy_optimizers(module, model):
    optimizer1 = AdamW(
        [
            dict(params=[model.embed.weight], lr=0.7),
            dict(params=[model.proj.weight], lr=0.004),
            dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.001,
        fused=True,
    )
    optimizer2 = module.Muon(
        [p for p in model.blocks.parameters() if p.ndim >= 2],
        lr=0.025,
        weight_decay=0.05,
    )
    optimizers = [optimizer1, optimizer2]
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers


def _parameter_names_by_group(model, optimizers):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        [[names[id(parameter)] for parameter in group["params"]]
         for group in optimizer.param_groups]
        for optimizer in optimizers
    ]


def _assert_nested_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert list(actual) == list(expected)
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "model_kwargs",
    [
        dict(vocab_size=37, num_layers=2, model_dim=128, mlp_ratio=4),
        dict(vocab_size=37, num_layers=2, model_dim=128, mlp_type="moe",
             mlp_ratio=2, num_experts=3, top_k=2, moe_backend="loop"),
    ],
)
def test_extracted_models_match_committed_baseline_exactly(baseline_module, model_kwargs):
    torch.manual_seed(2026)
    baseline = baseline_module.GPT(**model_kwargs)
    torch.manual_seed(2026)
    extracted = GPT(**model_kwargs)

    assert list(baseline.state_dict()) == list(extracted.state_dict())
    assert [tuple(value.shape) for value in baseline.state_dict().values()] == [
        tuple(value.shape) for value in extracted.state_dict().values()]
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, extracted.state_dict()[name], rtol=0, atol=0)

    torch.manual_seed(17)
    _initialize_like_trainer(baseline)
    torch.manual_seed(17)
    initialize_model_parameters(extracted)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, extracted.state_dict()[name], rtol=0, atol=0)


def test_optimizer_groups_and_order_match_committed_baseline(baseline_module):
    kwargs = dict(vocab_size=37, num_layers=2, model_dim=128, mlp_type="moe",
                  mlp_ratio=2, num_experts=3, top_k=2, moe_backend="loop")
    baseline_model = baseline_module.GPT(**kwargs)
    extracted_model = GPT(**kwargs)
    baseline_optimizers = _legacy_optimizers(baseline_module, baseline_model)
    extracted_optimizers = build_optimizers(extracted_model)

    assert [type(optimizer).__name__ for optimizer in baseline_optimizers] == [
        type(optimizer).__name__ for optimizer in extracted_optimizers]
    assert _parameter_names_by_group(baseline_model, baseline_optimizers) == (
        _parameter_names_by_group(extracted_model, extracted_optimizers))
    for baseline_optimizer, extracted_optimizer in zip(
            baseline_optimizers, extracted_optimizers):
        baseline_groups = baseline_optimizer.state_dict()["param_groups"]
        extracted_groups = extracted_optimizer.state_dict()["param_groups"]
        _assert_nested_equal(extracted_groups, baseline_groups)


def test_configured_default_optimizers_match_legacy_construction():
    kwargs = dict(vocab_size=37, num_layers=1, model_dim=128, mlp_ratio=4)
    legacy_model = GPT(**kwargs)
    configured_model = GPT(**kwargs)
    legacy = build_optimizers(legacy_model)
    configured = build_optimizers(
        configured_model, load_experiment_config()["optimizers"])

    assert _parameter_names_by_group(legacy_model, legacy) == (
        _parameter_names_by_group(configured_model, configured))
    for legacy_optimizer, configured_optimizer in zip(legacy, configured):
        _assert_nested_equal(
            configured_optimizer.state_dict()["param_groups"],
            legacy_optimizer.state_dict()["param_groups"],
        )


class _SerializableLoader:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state


def test_loads_checkpoint_generated_by_committed_baseline(baseline_module):
    kwargs = dict(vocab_size=37, num_layers=1, model_dim=128, mlp_ratio=2)
    torch.manual_seed(91)
    baseline_model = baseline_module.GPT(**kwargs)
    _initialize_like_trainer(baseline_model)
    baseline_optimizers = _legacy_optimizers(baseline_module, baseline_model)
    muon_parameter = baseline_optimizers[1].param_groups[0]["params"][0]
    baseline_optimizers[1].state[muon_parameter]["momentum"] = torch.randn_like(
        muon_parameter)
    loader_state = {"format_version": 1, "cursor": 23}
    resolved_config = {"model": kwargs, "training": {"total_steps": 4}}
    legacy_checkpoint = baseline_module.make_training_checkpoint(
        baseline_model,
        baseline_optimizers,
        completed_updates=2,
        batch_size=16,
        resolved_config=resolved_config,
        train_loader=_SerializableLoader(loader_state),
        run_id="legacy-run",
        trial_idx=0,
        training_time=1.25,
        current_segment_time=0.5,
        last_val_step=0,
        environment_metadata={"source": BASELINE_COMMIT},
    )
    serialized = io.BytesIO()
    torch.save(legacy_checkpoint, serialized)
    serialized.seek(0)
    legacy_checkpoint = torch.load(serialized, map_location="cpu", weights_only=False)

    extracted_model = GPT(**kwargs)
    extracted_optimizers = build_optimizers(extracted_model)
    extracted_loader = _SerializableLoader({})
    restored = restore_training_checkpoint(
        legacy_checkpoint,
        resolved_config,
        extracted_model,
        extracted_optimizers,
        extracted_loader,
        train_steps=4,
        batch_size=16,
    )

    _assert_nested_equal(extracted_model.state_dict(), baseline_model.state_dict())
    for extracted_optimizer, baseline_optimizer in zip(
            extracted_optimizers, baseline_optimizers):
        _assert_nested_equal(
            extracted_optimizer.state_dict(), baseline_optimizer.state_dict())
    assert extracted_loader.state == loader_state
    assert restored == {
        "completed_updates": 2,
        "training_time": 1.25,
        "current_segment_time": 0.5,
        "last_val_step": 0,
    }


def test_module_entry_point_logs_package_sources():
    assert callable(main)
    snapshot = read_source_snapshot()
    for relative_path in (
        "modded_nanogpt_moe/model.py",
        "modded_nanogpt_moe/_segmented_bias.py",
        "modded_nanogpt_moe/_combine.py",
        "modded_nanogpt_moe/_grouped_gemm.py",
        "modded_nanogpt_moe/diagnostics.py",
        "modded_nanogpt_moe/optim.py",
        "modded_nanogpt_moe/data.py",
        "modded_nanogpt_moe/checkpoint.py",
        "modded_nanogpt_moe/config.py",
        "modded_nanogpt_moe/train.py",
    ):
        assert f"# ===== {relative_path} =====" in snapshot


def test_training_benchmark_settings_and_summary_are_cpu_testable():
    assert benchmark_settings_from_environment({}) == {
        "enabled": False,
        "warmup_updates": 10,
        "measured_updates": 30,
    }
    assert benchmark_settings_from_environment({
        "BENCHMARK_MEASURED_UPDATES": "ignored-while-disabled",
    })["enabled"] is False
    assert benchmark_settings_from_environment({
        "TRAINING_BENCHMARK": "1",
        "BENCHMARK_WARMUP_UPDATES": "2",
        "BENCHMARK_MEASURED_UPDATES": "3",
    }) == {
        "enabled": True,
        "warmup_updates": 2,
        "measured_updates": 3,
    }
    with pytest.raises(ValueError, match="must be positive"):
        benchmark_settings_from_environment({
            "TRAINING_BENCHMARK": "1",
            "BENCHMARK_MEASURED_UPDATES": "0",
        })

    summary = summarize_training_benchmark(
        [0.1, 0.2, 0.3],
        tokens_per_update=600,
        peak_allocated_bytes=2 * 2**30,
        peak_reserved_bytes=3 * 2**30,
    )
    assert summary == {
        "mean_ms_per_update": pytest.approx(200),
        "median_ms_per_update": pytest.approx(200),
        "tokens_per_second": pytest.approx(3000),
        "peak_allocated_gib": pytest.approx(2),
        "peak_reserved_gib": pytest.approx(3),
    }


def test_dense_example_config_resolves_current_training_defaults():
    repository_root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(repository_root / "configs/dense_baseline.toml")
    grouped_config = load_experiment_config(
        repository_root / "configs/moe_e8k2_r2.toml")

    assert config["run_name"] == "dense-baseline"
    assert config["num_trials"] == 1
    assert config["seed"] == 1234
    assert config["checkpoint"] == {"interval": 100}
    assert config["evaluation"] == {
        "tokens": 10485760,
        "interval": 125,
        "final_fraction": 0.10,
        "final_interval": 25,
    }
    assert config["model"] == {
        "vocab_size": 50304,
        "num_layers": 12,
        "model_dim": 768,
        "mlp_type": "dense",
        "mlp_ratio": 4,
        "num_experts": 1,
        "top_k": 1,
        "normalize_topk": True,
        "moe_backend": "loop",
        "moe_parameter_layout": "modulelist",
    }
    assert config["training"]["global_batch_tokens"] == 524288
    assert config["training"]["microbatch_sequences"] == 64
    assert config["training"]["total_steps"] == 3250
    assert config["training"]["training_shard_pattern"] == (
        "data/fineweb10B/fineweb_train_*.bin")
    assert config["training"]["validation_shard_pattern"] == (
        "data/fineweb10B/fineweb_val_*.bin")

    defaults = load_experiment_config()
    for candidate in (defaults, grouped_config):
        assert candidate["training"]["training_shard_pattern"] == (
            config["training"]["training_shard_pattern"])
        assert candidate["training"]["validation_shard_pattern"] == (
            config["training"]["validation_shard_pattern"])


def test_olmoe_style_production_config_changes_only_controlled_geometry():
    repository_root = Path(__file__).resolve().parents[1]
    intended_reference = load_experiment_config(
        repository_root / "configs/moe_e8k2_r2.toml")
    experiment = load_experiment_config(
        repository_root / "configs/moe_e64k8_r0.5.toml")

    intended_reference["run_name"] = "moe-e64k8-r0.5"
    intended_reference["model"] = {
        **intended_reference["model"],
        "mlp_ratio": 0.5,
        "num_experts": 64,
        "top_k": 8,
    }
    assert experiment == intended_reference
    assert resolve_mlp_hidden_dim(
        experiment["model"]["model_dim"],
        experiment["model"]["mlp_ratio"],
    ) == 384


def test_vista_launcher_scopes_machine_and_run_environment(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    launcher = repository_root / "scripts/vista/train.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    module = fake_bin / "module"
    module.write_text("#!/usr/bin/env bash\nexit 0\n")
    module.chmod(0o755)
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$PWD\" \"${DATA_ROOT-}\" "
        "\"${TRAIN_STEPS_OVERRIDE-}\" \"${MOE_GMM_IMPLEMENTATION-}\" "
        "\"${TRAINING_BENCHMARK-}\" \"${NSYS_PROFILE-}\" "
        "\"${RESUME_CHECKPOINT-}\" \"${STOP_AFTER_COMPLETED_UPDATES-}\" "
        "\"${CHECKPOINT_DIR-}\" \"${CHECKPOINT_INTERVAL-}\" "
        "\"${CHECKPOINT_ROOT-}\" \"${CHECKPOINT_POLICY_DISABLED-}\" "
        "\"${TB_ROOT-}\" \"${TB_SYSTEM-}\" "
        "> \"$LAUNCH_CAPTURE\"\n"
        "printf '%s\\n' \"$@\" >> \"$LAUNCH_CAPTURE\"\n"
    )
    uv.chmod(0o755)
    capture = tmp_path / "launch.txt"
    outside_repository = tmp_path / "outside"
    outside_repository.mkdir()
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["LAUNCH_CAPTURE"] = str(capture)
    environment["STOCKYARD"] = str(tmp_path / "stockyard")
    environment["DATA_ROOT"] = "/stale/data"
    environment["TRAIN_STEPS_OVERRIDE"] = "999"
    environment["MOE_GMM_IMPLEMENTATION"] = "extension"
    environment["TRAINING_BENCHMARK"] = "1"
    environment["NSYS_PROFILE"] = "1"
    environment["RESUME_CHECKPOINT"] = "/stale/checkpoint.pt"
    environment["STOP_AFTER_COMPLETED_UPDATES"] = "2"
    environment["CHECKPOINT_DIR"] = "/stale/checkpoints"
    environment["CHECKPOINT_INTERVAL"] = "99"
    environment["CHECKPOINT_ROOT"] = "/stale/checkpoint-root"
    environment["CHECKPOINT_POLICY_DISABLED"] = "0"

    subprocess.run(
        ["bash", str(launcher), "--steps", "17", "configs/dense_baseline.toml"],
        cwd=outside_repository,
        env=environment,
        check=True,
    )

    captured = capture.read_text().splitlines()
    assert captured[:14] == [
        str(repository_root), str(repository_root), "17", "torch",
        "", "", "", "", "", "",
        str(tmp_path / "stockyard/checkpoints/modded-nanogpt-moe"), "1", "", "vista",
    ]
    assert captured[14:] == [
        "run", "--no-sync", "torchrun", "--standalone", "--nproc_per_node=1",
        "--module", "modded_nanogpt_moe.train", "--config",
        str(repository_root / "configs/dense_baseline.toml"),
    ]


@pytest.mark.parametrize("mail_user", [None, "", "researcher@example.com"])
def test_vista_launcher_submits_itself_as_nonrecursive_worker(tmp_path, mail_user):
    repository_root = Path(__file__).resolve().parents[1]
    launcher = repository_root / "scripts/vista/train.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$SUBMIT_CAPTURE\"\n"
    )
    sbatch.chmod(0o755)
    capture = tmp_path / "submit.txt"
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["SUBMIT_CAPTURE"] = str(capture)
    environment.pop("SLURM_MAIL_USER", None)
    if mail_user is not None:
        environment["SLURM_MAIL_USER"] = mail_user

    subprocess.run(
        [
            "bash", str(launcher), "--submit", "--time", "08:00:00",
            "--job-name", "dense-moe comparison", "--account", "allocation",
            "--sbatch-arg=--partition=gh", "--sbatch-arg", "--exclusive",
            "configs/dense_baseline.toml", "configs/moe_e8k2_r2.toml",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    assert capture.read_text().splitlines() == [
        "--time=08:00:00",
        *([f"--mail-user={mail_user}", "--mail-type=ALL"] if mail_user else []),
        "--job-name=dense-moe comparison", "--account=allocation", "--partition=gh", "--exclusive",
        str(launcher),
        "--worker",
        "--",
        str(repository_root / "configs/dense_baseline.toml"),
        str(repository_root / "configs/moe_e8k2_r2.toml"),
    ]


@pytest.mark.parametrize("mail_user", [None, "", "researcher@example.com"])
def test_vista_submission_email_options_without_cluster_setup(tmp_path, mail_user):
    # Exercise the actual submission block independently of the launcher's
    # Bash-4-only config preflight, so this test also runs on macOS Bash 3.2.
    launcher = Path(__file__).resolve().parents[1] / "scripts/vista/train.sh"
    source = launcher.read_text()
    start = source.index('if [[ "$submit" -eq 1 ]]; then\n')
    end = source.index('if [[ "$worker" -eq 1 &&', start)
    sbatch = tmp_path / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SUBMIT_CAPTURE"\nexit 23\n')
    sbatch.chmod(0o755)
    capture = tmp_path / "args.txt"
    environment = os.environ.copy()
    environment.update(PATH=f"{tmp_path}{os.pathsep}{environment['PATH']}",
                       SUBMIT_CAPTURE=str(capture))
    environment.pop("SLURM_MAIL_USER", None)
    if mail_user is not None:
        environment["SLURM_MAIL_USER"] = mail_user
    setup = ('set -euo pipefail\nsubmit=1\nwalltime_set=0\nsbatch_options=()\n'
             'script_path="/repo with spaces/scripts/vista/train.sh"\n'
             'checkpoint_interval=""\nresume_checkpoint=""\n'
             'config_paths=("/repo with spaces/config.toml")\n')
    result = subprocess.run(["bash", "-c", setup + source[start:end]],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 23  # sbatch status is preserved, no real submission.
    assert capture.read_text().splitlines() == [
        *([f"--mail-user={mail_user}", "--mail-type=ALL"] if mail_user else []),
        "/repo with spaces/scripts/vista/train.sh", "--worker", "--",
        "/repo with spaces/config.toml",
    ]


def test_vista_launcher_runs_configs_in_order_with_explicit_checkpoints(
        tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    launcher = repository_root / "scripts/vista/train.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    module = fake_bin / "module"
    module.write_text("#!/usr/bin/env bash\nexit 0\n")
    module.chmod(0o755)
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s|%s|%s|%s\\n' \"${CHECKPOINT_ROOT-}\" "
        "\"${CHECKPOINT_INTERVAL-}\" \"${RESUME_CHECKPOINT-}\" \"$*\" "
        ">> \"$LAUNCH_CAPTURE\"\n"
    )
    uv.chmod(0o755)
    capture = tmp_path / "launches.txt"
    resume = tmp_path / "latest.pt"
    resume.touch()
    stockyard = tmp_path / "stockyard"
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["LAUNCH_CAPTURE"] = str(capture)
    environment["STOCKYARD"] = str(stockyard)

    subprocess.run(
        [
            "bash", str(launcher), "--checkpoint-interval", "7",
            "--resume", str(resume), "configs/dense_baseline.toml",
            "configs/moe_e8k2_r2.toml",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    command_prefix = (
        "run --no-sync torchrun --standalone --nproc_per_node=1 "
        "--module modded_nanogpt_moe.train --config "
    )
    assert capture.read_text().splitlines() == [
        f"{stockyard}/checkpoints/modded-nanogpt-moe|7|"
        f"{resume}|{command_prefix}"
        f"{repository_root / 'configs/dense_baseline.toml'}",
        f"{stockyard}/checkpoints/modded-nanogpt-moe|7||"
        f"{command_prefix}{repository_root / 'configs/moe_e8k2_r2.toml'}",
    ]


def test_vista_benchmark_launcher_uses_trainer_without_artifacts(
        tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    launcher = repository_root / "scripts/vista/benchmark.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    module = fake_bin / "module"
    module.write_text("#!/usr/bin/env bash\nexit 0\n")
    module.chmod(0o755)
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n' \"$PWD\" "
        "\"${TRAINING_BENCHMARK-}\" \"${TB_ROOT+x}\" \"${TB_ROOT-}\" "
        "\"${BENCHMARK_WARMUP_UPDATES-}\" "
        "\"${BENCHMARK_MEASURED_UPDATES-}\" > \"$LAUNCH_CAPTURE\"\n"
        "printf '%s\\n' \"$@\" >> \"$LAUNCH_CAPTURE\"\n"
    )
    uv.chmod(0o755)
    capture = tmp_path / "benchmark-launch.txt"
    outside_repository = tmp_path / "outside"
    outside_repository.mkdir()
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["LAUNCH_CAPTURE"] = str(capture)
    environment["STOCKYARD"] = str(tmp_path / "stockyard")
    environment["TB_ROOT"] = "/must/not/be/used"
    environment["BENCHMARK_WARMUP_UPDATES"] = "2"
    environment["BENCHMARK_MEASURED_UPDATES"] = "3"
    for name in (
        "SEED_OVERRIDE", "MBS_OVERRIDE", "TRAIN_STEPS_OVERRIDE",
        "MLP_TYPE_OVERRIDE", "MLP_RATIO_OVERRIDE", "NUM_EXPERTS_OVERRIDE",
        "TOP_K_OVERRIDE", "MOE_BACKEND_OVERRIDE", "CHECKPOINT_DIR",
        "CHECKPOINT_INTERVAL", "RESUME_CHECKPOINT",
        "STOP_AFTER_COMPLETED_UPDATES", "REPRO_DIAGNOSTICS_DIR", "NSYS_PROFILE",
    ):
        environment.pop(name, None)

    subprocess.run(
        ["bash", str(launcher), "configs/moe_e8k2_r2.toml"],
        cwd=outside_repository,
        env=environment,
        check=True,
    )

    captured = capture.read_text().splitlines()
    assert captured[:6] == [str(repository_root), "1", "x", "", "2", "3"]
    assert captured[6:] == [
        "run", "--no-sync", "torchrun", "--standalone", "--nproc_per_node=1",
        "--module", "modded_nanogpt_moe.train", "--config",
        str(repository_root / "configs/moe_e8k2_r2.toml"),
    ]


def test_config_requires_run_name_and_rejects_unknown_fields(tmp_path):
    missing_name = tmp_path / "missing_name.toml"
    missing_name.write_text("[model]\nmlp_type = 'dense'\n")
    with pytest.raises(ValueError, match="run_name is required"):
        load_experiment_config(missing_name)

    unknown = tmp_path / "unknown.toml"
    unknown.write_text("run_name = 'bad-key'\n[training]\nsteps = 10\n")
    with pytest.raises(ValueError, match="training.steps"):
        load_experiment_config(unknown)


def test_checkpoint_config_interval_validation_and_default(tmp_path):
    assert load_experiment_config()["checkpoint"] == {"interval": None}

    configured = tmp_path / "configured.toml"
    configured.write_text(
        'run_name = "checkpointed"\n[checkpoint]\ninterval = 0\n')
    assert load_experiment_config(configured)["checkpoint"] == {"interval": 0}

    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        'run_name = "invalid-checkpoint"\n[checkpoint]\ninterval = -1\n')
    with pytest.raises(ValueError, match="checkpoint.interval must be a nonnegative integer"):
        load_experiment_config(invalid)


def test_checkpoint_directory_uses_run_name_under_machine_root(tmp_path):
    root = tmp_path / "checkpoints" / "modded-nanogpt-moe"
    assert checkpoint_directory_from_environment(
        "run-a", True, {"CHECKPOINT_ROOT": str(root)}) == str(root / "run-a")
    assert checkpoint_directory_from_environment(
        "run-a", True,
        {"CHECKPOINT_ROOT": str(root), "CHECKPOINT_DIR": "/explicit"},
    ) == "/explicit"
    assert checkpoint_directory_from_environment(
        "run-a", False, {"CHECKPOINT_ROOT": str(root)}) == ""


def test_existing_environment_overrides_take_precedence(monkeypatch):
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("MBS_OVERRIDE", "32")
    monkeypatch.setenv("TRAIN_STEPS_OVERRIDE", "20")
    monkeypatch.setenv("MLP_TYPE_OVERRIDE", "moe")
    monkeypatch.setenv("MLP_RATIO_OVERRIDE", "2")
    monkeypatch.setenv("NUM_EXPERTS_OVERRIDE", "8")
    monkeypatch.setenv("TOP_K_OVERRIDE", "2")
    monkeypatch.setenv("MOE_BACKEND_OVERRIDE", "grouped_gemm")
    config, path = parse_train_args([
        "--config", str(repository_root / "configs/dense_baseline.toml")])

    assert path == repository_root / "configs/dense_baseline.toml"
    assert config["training"]["microbatch_sequences"] == 32
    assert config["training"]["total_steps"] == 20
    assert config["model"]["mlp_type"] == "moe"
    assert config["model"]["mlp_ratio"] == 2
    assert config["model"]["num_experts"] == 8
    assert config["model"]["top_k"] == 2
    assert config["model"]["moe_backend"] == "grouped_gemm"


def test_checkpoint_environment_override_precedes_toml(monkeypatch):
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("CHECKPOINT_INTERVAL", "17")

    config, _ = parse_train_args([
        "--config", str(repository_root / "configs/moe_e64k8_r0.5.toml")])

    assert config["checkpoint"]["interval"] == 17


@pytest.mark.parametrize("filename,run_name,ratio,experts,top_k", [
    ("dense_baseline.toml", "dense-baseline", 4, None, None),
    ("moe_e8k2_r2.toml", "moe-e8k2-r2", 2, 8, 2),
    ("moe_e64k8_r0.5.toml", "moe-e64k8-r0.5", 0.5, 64, 8),
])
def test_production_configs_share_training_policy(filename, run_name, ratio, experts, top_k):
    repository_root = Path(__file__).resolve().parents[1]
    path = repository_root / "configs" / filename
    config = load_experiment_config(path)
    raw = tomllib.loads(path.read_text())
    assert list(raw) == ["run_name", "num_trials", "seed", "model", "training",
                         "evaluation", "optimizers", "diagnostics", "checkpoint"]
    assert list(raw["optimizers"]) == ["adamw", "muon"]
    assert config["run_name"] == run_name
    assert config["num_trials"] == 1
    assert config["seed"] == 1234
    assert config["checkpoint"] == {"interval": 100}
    assert config["diagnostics"] == dict(scalar_interval=25, histogram_interval=0, during_nsys=False)
    assert config["training"] == {
        "sequence_length": 1024, "global_batch_tokens": 524288,
        "microbatch_sequences": 64, "total_steps": 3250, "cooldown_fraction": 0.7,
        "training_shard_pattern": "data/fineweb10B/fineweb_train_*.bin",
        "validation_shard_pattern": "data/fineweb10B/fineweb_val_*.bin",
    }
    assert config["evaluation"] == {
        "tokens": 10485760, "interval": 125,
        "final_fraction": 0.10, "final_interval": 25,
    }
    assert config["optimizers"] == {
        "adamw": dict(group_lrs=[0.7, 0.004, 0.015], betas=[0.8, 0.95],
                      eps=1e-10, weight_decay=0.001, fused=True),
        "muon": dict(lr=0.025, weight_decay=0.05, mu=0.95),
    }
    expected_model = dict(vocab_size=50304, num_layers=12, model_dim=768,
                          mlp_type="moe" if experts else "dense", mlp_ratio=ratio)
    if experts:
        expected_model.update(num_experts=experts, top_k=top_k, normalize_topk=True,
                              moe_backend="grouped_gemm", moe_parameter_layout="packed")
    assert raw["model"] == expected_model  # Dense doesn't explicitly carry MoE-only fields.
    assert resolve_mlp_hidden_dim(768, ratio) == {4: 3072, 2: 1536, 0.5: 384}[ratio]


def test_tensorboard_run_name_is_used_and_existing_directory_is_rejected(tmp_path):
    expected = tmp_path / "modded-nanogpt-moe" / "vista" / "dense-baseline"
    assert tensorboard_run_directory(
        tmp_path, "vista", "dense-baseline") == expected
    assert require_unused_tensorboard_run_directory(
        tmp_path, "vista", "dense-baseline") == expected

    expected.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="dense-baseline"):
        require_unused_tensorboard_run_directory(
            tmp_path, "vista", "dense-baseline")
