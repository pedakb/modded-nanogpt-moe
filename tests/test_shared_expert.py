"""Shared dense expert behavior across BP, Grad-EM, and persistence."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modded_nanogpt_moe import _grouped_gemm as gemm
from modded_nanogpt_moe.checkpoint import (
    atomic_save_checkpoint,
    make_training_checkpoint,
    restore_training_checkpoint,
    validate_checkpoint_config,
)
from modded_nanogpt_moe.config import (
    load_experiment_config,
    validate_experiment_config,
)
from modded_nanogpt_moe.model import GPT, MoE, eager_prefix, make_head_loss
from modded_nanogpt_moe.optim import build_optimizers


@pytest.fixture
def cpu_grouped_gemm(monkeypatch):
    """Install differentiable CPU doubles for both grouped-GEMM APIs."""
    def raw(a, b, counts, trans_a=False, trans_b=False):
        segments = a.split(counts.tolist())
        if trans_a:
            return torch.stack([
                x.mT @ y for x, y in zip(segments, b.split(counts.tolist()))
            ])
        return torch.cat([
            x @ (weight.mT if trans_b else weight)
            for x, weight in zip(segments, b)
        ])

    def extension(a, b, counts, trans_b=False):
        assert not trans_b
        return gemm._ProfiledExtensionGemm.apply(a, b, counts, "test")

    monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(
        ops=SimpleNamespace(gmm=extension), backend=SimpleNamespace(gmm=raw)))
    monkeypatch.delenv("MOE_GMM_IMPLEMENTATION", raising=False)


def _copy_routed(source, destination):
    source_state = source.state_dict()
    destination.load_state_dict(
        {name: value for name, value in source_state.items()
         if not name.startswith("shared_expert.")},
        strict=False,
    )


def _routed_grads(module):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if not name.startswith("shared_expert.")
    }


def test_shared_expert_forward_is_additive_and_disabled_path_is_exact():
    torch.manual_seed(10)
    legacy = MoE(8, 4, 2, hidden_dim=6)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(10)
    disabled = MoE(8, 4, 2, hidden_dim=6, shared_expert_hidden_dim=None)
    assert torch.equal(torch.get_rng_state(), rng)
    assert list(disabled.state_dict()) == list(legacy.state_dict())

    enabled = MoE(8, 4, 2, hidden_dim=6, shared_expert_hidden_dim=5)
    _copy_routed(legacy, enabled)
    x = torch.randn(2, 3, 8)
    torch.testing.assert_close(disabled(x), legacy(x), rtol=0, atol=0)
    expected = legacy(x) + enabled.shared_expert(x)
    torch.testing.assert_close(enabled(x), expected, rtol=0, atol=0)
    assert enabled(x).shape == x.shape


def test_shared_expert_initialization_matches_across_routed_layouts(cpu_grouped_gemm):
    kwargs = dict(
        dim=8, num_experts=4, top_k=2, hidden_dim=6,
        shared_expert_hidden_dim=5, moe_backend="grouped_gemm",
    )
    torch.manual_seed(101)
    modulelist = MoE(**kwargs)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(101)
    packed = MoE(**kwargs, moe_parameter_layout="packed")
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in modulelist.shared_expert.state_dict().items():
        torch.testing.assert_close(
            packed.shared_expert.state_dict()[name], value, rtol=0, atol=0)


def test_shared_expert_receives_standard_bp_gradients():
    module = MoE(8, 4, 2, hidden_dim=6, shared_expert_hidden_dim=5)
    x = torch.randn(2, 3, 8, requires_grad=True)
    module(x).square().sum().backward()
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in module.shared_expert.parameters())
    assert all(torch.count_nonzero(parameter.grad)
               for parameter in module.shared_expert.parameters())


def test_local_bp_shared_expert_remains_trainable_with_frozen_routed_parameters():
    module = MoE(
        8, 4, 2, hidden_dim=6, shared_expert_hidden_dim=5,
        moe_backward="grad_em", grad_em_mode="local_bp",
    )
    for name, parameter in module.named_parameters():
        if not name.startswith("shared_expert."):
            parameter.requires_grad_(False)
    x = torch.randn(2, 3, 8, requires_grad=True)
    module(x).sum().backward()
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in module.shared_expert.parameters())
    assert all(parameter.grad is None for name, parameter in module.named_parameters()
               if not name.startswith("shared_expert."))


@pytest.mark.parametrize("mode,backend", [
    ("global", "grouped_gemm"),
    ("local_bp", "grouped_gemm"),
    ("local_bp", "loop"),
])
def test_shared_expert_uses_bp_without_changing_routed_grad_em_signals(
        cpu_grouped_gemm, mode, backend):
    kwargs = dict(
        dim=8, num_experts=4, top_k=2, hidden_dim=6,
        moe_backend=backend, moe_backward="grad_em", grad_em_mode=mode,
        grad_em_eta=0.2,
    )
    torch.manual_seed(11)
    routed_only = MoE(**kwargs)
    torch.manual_seed(11)
    with_shared = MoE(**kwargs, shared_expert_hidden_dim=5)
    _copy_routed(routed_only, with_shared)

    value = torch.randn(2, 3, 8)
    upstream = torch.randn_like(value)
    x_routed = value.clone().requires_grad_(True)
    x_shared = value.clone().requires_grad_(True)
    routed_only(x_routed).backward(upstream)
    with_shared(x_shared).backward(upstream)

    for name, expected in _routed_grads(routed_only).items():
        torch.testing.assert_close(
            _routed_grads(with_shared)[name], expected, rtol=0, atol=0)
    assert all(parameter.grad is not None for parameter in with_shared.shared_expert.parameters())
    assert all(torch.count_nonzero(parameter.grad) for parameter in with_shared.shared_expert.parameters())

    reference = copy.deepcopy(with_shared.shared_expert)
    reference.zero_grad(set_to_none=True)
    x_reference = value.clone().requires_grad_(True)
    reference(x_reference).backward(upstream)
    for actual, expected in zip(
            with_shared.shared_expert.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        x_shared.grad, x_routed.grad + x_reference.grad, rtol=0, atol=0)


def test_loop_and_grouped_routed_backends_match_with_shared_expert(cpu_grouped_gemm):
    torch.manual_seed(12)
    loop = MoE(8, 5, 2, hidden_dim=6, shared_expert_hidden_dim=5)
    grouped = MoE(
        8, 5, 2, hidden_dim=6, shared_expert_hidden_dim=5,
        moe_backend="grouped_gemm",
    )
    grouped.load_state_dict(loop.state_dict())
    value = torch.randn(2, 4, 8)
    upstream = torch.randn_like(value)
    x_loop = value.clone().requires_grad_(True)
    x_grouped = value.clone().requires_grad_(True)
    loop(x_loop).backward(upstream)
    grouped(x_grouped).backward(upstream)
    torch.testing.assert_close(grouped(x_grouped.detach()), loop(x_loop.detach()))
    torch.testing.assert_close(x_grouped.grad, x_loop.grad)
    for grouped_parameter, loop_parameter in zip(grouped.parameters(), loop.parameters()):
        torch.testing.assert_close(grouped_parameter.grad, loop_parameter.grad)


def _optimizer_holder(moe):
    model = nn.Module()
    model.embed = nn.Embedding(4, 8)
    model.proj = nn.Linear(8, 4)
    model.blocks = nn.ModuleList([moe])
    return model


def test_shared_expert_uses_dense_optimizer_ownership_and_updates(
        cpu_grouped_gemm, monkeypatch):
    moe = MoE(
        8, 4, 2, hidden_dim=6, shared_expert_hidden_dim=5,
        moe_backend="grouped_gemm", moe_parameter_layout="packed",
    )
    model = _optimizer_holder(moe)
    adamw, muon = build_optimizers(model)
    adam_parameters = {p for group in adamw.param_groups for p in group["params"]}
    muon_parameters = {p for group in muon.param_groups for p in group["params"]}
    assert {moe.shared_expert.fc.bias, moe.shared_expert.proj.bias} <= adam_parameters
    assert {moe.shared_expert.fc.weight, moe.shared_expert.proj.weight} <= muon_parameters
    assert not ({moe.shared_expert.fc.weight, moe.shared_expert.proj.weight}
                & muon.transposed_params)

    monkeypatch.setattr("modded_nanogpt_moe.optim.dist.get_world_size", lambda: 1)
    monkeypatch.setattr("modded_nanogpt_moe.optim.dist.get_rank", lambda: 0)
    monkeypatch.setattr("modded_nanogpt_moe.optim.dist.all_gather", lambda outputs, value: None)
    before = {p: p.detach().clone() for p in moe.shared_expert.parameters()}
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    adamw.step()
    muon.step()
    assert all(not torch.equal(parameter, before[parameter])
               for parameter in moe.shared_expert.parameters())


def test_shared_expert_checkpoint_round_trip(cpu_grouped_gemm, tmp_path):
    class Loader:
        def __init__(self, state):
            self.state = state
        def state_dict(self):
            return self.state
        def load_state_dict(self, state):
            self.state = state

    def construct():
        moe = MoE(
            8, 4, 2, hidden_dim=6, shared_expert_hidden_dim=5,
            moe_backend="grouped_gemm", moe_parameter_layout="packed",
        )
        model = _optimizer_holder(moe)
        return model, build_optimizers(model)

    original, optimizers = construct()
    for parameter in original.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizers[0].step()
    for parameter in optimizers[1].param_groups[0]["params"]:
        optimizers[1].state[parameter]["momentum"] = torch.randn_like(parameter)
    original.zero_grad(set_to_none=True)
    config = {
        "model": {"moe_parameter_layout": "packed", "num_shared_experts": 1,
                  "shared_expert_ratio": 0.625},
        "training": {"total_steps": 20},
    }
    checkpoint = make_training_checkpoint(
        original, optimizers, 2, 16, config, Loader({"cursor": 3}), "shared", 0,
        1.0, 0.5, 0, {},
    )
    path = atomic_save_checkpoint(checkpoint, tmp_path)
    restored, restored_optimizers = construct()
    loader = Loader({})
    result = restore_training_checkpoint(
        torch.load(path, weights_only=False), config, restored,
        restored_optimizers, loader, train_steps=20, batch_size=16,
    )
    assert result["completed_updates"] == 2
    assert loader.state == {"cursor": 3}
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for old, new in zip(optimizers, restored_optimizers):
        assert old.state_dict()["param_groups"] == new.state_dict()["param_groups"]
        for index, state in old.state_dict()["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(
                    new.state_dict()["state"][index][key], value, rtol=0, atol=0)


def test_shared_config_validation_and_legacy_checkpoint_compatibility():
    config = load_experiment_config()
    for value in (-1, 2, True):
        invalid = copy.deepcopy(config)
        invalid["model"]["num_shared_experts"] = value
        with pytest.raises(ValueError, match="num_shared_experts"):
            validate_experiment_config(invalid)
    for value in (0, -0.5, 0.1, float("nan")):
        invalid = copy.deepcopy(config)
        invalid["model"]["shared_expert_ratio"] = value
        with pytest.raises(ValueError, match="shared_expert_ratio"):
            validate_experiment_config(invalid)

    legacy = {"model": {"mlp_type": "moe"}, "training": {"total_steps": 20}}
    current = copy.deepcopy(legacy)
    current["model"].update(num_shared_experts=0, shared_expert_ratio=0.5)
    checkpoint = {"format_version": 1, "resolved_config": legacy}
    validate_checkpoint_config(checkpoint, current)


def test_e256k6_configs_resolve_target_geometry_and_distinct_runs(cpu_grouped_gemm):
    root = Path(__file__).resolve().parents[1] / "configs"
    shared = load_experiment_config(root / "moe_e256k6_r0.5_shared_r0.5.toml")
    unshared = load_experiment_config(root / "moe_e256k6_r0.5.toml")
    assert shared["run_name"] != unshared["run_name"]
    for config in (shared, unshared):
        model = config["model"]
        assert (model["model_dim"], model["num_experts"], model["top_k"]) == (768, 256, 6)
        assert model["mlp_ratio"] == 0.5
        assert model["model_dim"] * model["mlp_ratio"] == 384
        assert model["moe_backend"] == "grouped_gemm"
        assert model["moe_parameter_layout"] == "packed"
        assert model["moe_backward"] == "standard"
    assert shared["model"]["num_shared_experts"] == 1
    assert shared["model"]["model_dim"] * shared["model"]["shared_expert_ratio"] == 384
    assert (shared["model"]["top_k"] + shared["model"]["num_shared_experts"]) * 0.5 == 3.5
    assert unshared["model"]["num_shared_experts"] == 0

    # Exercise the exact E/K dispatch geometry at a unit-test width without
    # allocating the production model's multi-gigabyte optimizer state.
    smoke = MoE(
        8, 256, 6, hidden_dim=4, shared_expert_hidden_dim=4,
        moe_backend="grouped_gemm", moe_parameter_layout="packed",
    )
    x = torch.randn(1, 3, 8, requires_grad=True)
    smoke(x).sum().backward()
    assert x.grad is not None
    assert smoke.fc_weight.grad.shape == (256, 8, 4)


def test_training_compile_boundary_accepts_shared_expert():
    model = GPT(
        vocab_size=37, num_layers=1, model_dim=128, mlp_type="moe",
        mlp_ratio=0.5, num_experts=4, top_k=2,
        num_shared_experts=1, shared_expert_ratio=0.5,
    )
    model.proj.weight.data.normal_(std=0.02)
    inputs = torch.randint(0, 37, (2, 4))
    targets = torch.randint(0, 37, (2, 4))
    prefix = eager_prefix(model, inputs)
    expected = make_head_loss(model)(prefix, targets)
    compiled = torch.compile(make_head_loss(model), fullgraph=True, dynamic=False)
    actual = compiled(prefix, targets)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    actual.backward()
    shared = model.blocks[0].mlp.shared_expert
    assert all(parameter.grad is not None for parameter in shared.parameters())
