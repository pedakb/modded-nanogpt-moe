import io
import subprocess
import types
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

import train_gpt_simple as compatibility_wrapper
from modded_nanogpt_moe.checkpoint import restore_training_checkpoint
from modded_nanogpt_moe.config import load_experiment_config, parse_train_args
from modded_nanogpt_moe.model import GPT
from modded_nanogpt_moe.optim import Muon, build_optimizers
from modded_nanogpt_moe.train import (
    main,
    read_source_snapshot,
    require_unused_tensorboard_run_directory,
    tensorboard_run_directory,
)


BASELINE_COMMIT = "372221a"
BASELINE_PATH = "records/track_3_optimization/train_gpt_simple.py"


@pytest.fixture(scope="module")
def baseline_module():
    repository_root = Path(__file__).resolve().parents[2]
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
    _initialize_like_trainer(extracted)
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


def test_wrapper_and_module_entry_points_share_main_and_log_package_sources():
    assert compatibility_wrapper.main is main
    snapshot = read_source_snapshot()
    for relative_path in (
        "modded_nanogpt_moe/model.py",
        "modded_nanogpt_moe/optim.py",
        "modded_nanogpt_moe/data.py",
        "modded_nanogpt_moe/checkpoint.py",
        "modded_nanogpt_moe/config.py",
        "modded_nanogpt_moe/train.py",
        "records/track_3_optimization/train_gpt_simple.py",
    ):
        assert f"# ===== {relative_path} =====" in snapshot


def test_dense_example_config_resolves_current_training_defaults():
    repository_root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(repository_root / "configs/dense_baseline.toml")

    assert config["run_name"] == "dense-baseline"
    assert config["num_trials"] == 1
    assert config["seed"] == 1234
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
    }
    assert config["training"]["global_batch_tokens"] == 524288
    assert config["training"]["microbatch_sequences"] == 64
    assert config["training"]["total_steps"] == 3250


def test_config_requires_run_name_and_rejects_unknown_fields(tmp_path):
    missing_name = tmp_path / "missing_name.toml"
    missing_name.write_text("[model]\nmlp_type = 'dense'\n")
    with pytest.raises(ValueError, match="run_name is required"):
        load_experiment_config(missing_name)

    unknown = tmp_path / "unknown.toml"
    unknown.write_text("run_name = 'bad-key'\n[training]\nsteps = 10\n")
    with pytest.raises(ValueError, match="training.steps"):
        load_experiment_config(unknown)


def test_existing_environment_overrides_take_precedence(monkeypatch):
    repository_root = Path(__file__).resolve().parents[2]
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
