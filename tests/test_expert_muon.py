"""Compass factors on the unchanged local Muon baseline (CPU tests)."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modded_nanogpt_moe import optim
from modded_nanogpt_moe.config import load_experiment_config
from modded_nanogpt_moe.model import MoE


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "all_gather", lambda *args: None)
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "extension")
    monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(ops=SimpleNamespace(gmm=None)))


def make_model(packed, experts=8, dtype=torch.float32):
    model = nn.Module()
    model.embed = nn.Embedding(7, 4)
    model.proj = nn.Linear(4, 7, bias=False)
    model.blocks = nn.ModuleList([
        MoE(4, experts, min(experts, 2), hidden_dim=2,
            moe_backend="grouped_gemm" if packed else "loop",
            moe_parameter_layout="packed" if packed else "modulelist")
        for _ in range(2)])
    # Same shape as expert FC1, but this non-expert must not receive factors.
    model.blocks.append(nn.Linear(4, 2))
    return model.to(dtype)


def build(model, compass=None, router="muon"):
    config = load_experiment_config()["optimizers"]
    config["adamw"]["fused"] = False
    config["router_optimizer"] = router
    if compass is not None:
        config["muon"]["compass"] = compass
    return optim.build_optimizers(model, config)


def assert_exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("experts", [1, 8, 64])
def test_unit_factors_exactly_match_baseline_updates_and_momentum(monkeypatch, packed, dtype, experts):
    monkeypatch.setattr(optim, "_compass_factors",
                        lambda directions, gradients, ratio:
                        [(1., u.new_tensor(1.)) for u in directions])
    torch.manual_seed(421)
    baseline = make_model(packed, experts, dtype)
    matched = copy.deepcopy(baseline)
    old, new = build(baseline), build(matched, True)
    assert len(old) == len(new) == 2
    for step, ratio in enumerate((1., .4, 0.)):
        for index, (a, b) in enumerate(zip(baseline.parameters(), matched.parameters())):
            a.grad = torch.randn_like(a)
            if index % 3 == 0 and step != 1:
                a.grad.zero_()
            b.grad = a.grad.clone()
        for pair in (old, new):
            for opt in pair:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * ratio
                opt.step()
        assert_exact(baseline.state_dict(), matched.state_dict())
        for a, b in zip(old, new):
            assert_exact(a.state_dict(), b.state_dict())


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("router", ["muon", "adamw"])
def test_real_factors_change_only_experts_and_resume_exactly(packed, router):
    torch.manual_seed(74)
    baseline = make_model(packed)
    matched = copy.deepcopy(baseline)
    old, new = build(baseline, router=router), build(matched, True, router)
    assert len(new[1].compass_families) == 4  # Two layers, distinct FC1/FC2 families.
    names = {id(p): name for name, p in matched.named_parameters()}
    expert_names = {names[id(p)] for p in new[1].compass_parameters}
    # Both optimizers retain baseline parameter order, hyperparameters and ownership.
    for a, b in zip(old, new):
        assert_exact(a.state_dict(), b.state_dict())
    for step in range(3):
        for a, b in zip(baseline.parameters(), matched.parameters()):
            a.grad = torch.randn_like(a)
            b.grad = a.grad.clone()
        for pair in (old, new):
            for opt in pair:
                opt.step()
        for name, p in baseline.named_parameters():
            if name not in expert_names:
                assert_exact(p, dict(matched.named_parameters())[name])
        assert_exact(old[1].state_dict(), new[1].state_dict())
    assert any(not torch.equal(p, dict(matched.named_parameters())[name])
               for name, p in baseline.named_parameters() if name in expert_names)
    resumed = copy.deepcopy(matched)
    restored = build(resumed, True, router)
    for a, b in zip(restored, new):
        a.load_state_dict(copy.deepcopy(b.state_dict()))
    for a, b in zip(resumed.parameters(), matched.parameters()):
        a.grad = torch.randn_like(a)
        b.grad = a.grad.clone()
    for pair in (restored, new):
        for opt in pair:
            opt.step()
    assert_exact(resumed.state_dict(), matched.state_dict())


@pytest.mark.parametrize("ratio", [1., .3, 0.])
def test_factor_formulas_match_read_only_reference(monkeypatch, ratio):
    source = Path(__file__).resolve().parents[2] / "ExpertMuon" / "src"
    if not source.is_dir():
        pytest.skip("factor parity requires the read-only sibling ExpertMuon checkout")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(source))
    from expertmuon_compass.optim.literal_reference import reference_geometry
    torch.manual_seed(23)
    directions = [torch.randn(3, 5) for _ in range(8)]
    gradients = [torch.randn(3, 5) for _ in range(8)]
    gradients[0].zero_()
    directions[1].zero_()
    expected = reference_geometry(directions, gradients, ratio)
    actual = optim._compass_factors(directions, gradients, ratio)
    for (gate, radius), (_, ref_gate, _, _, ref_radius) in zip(actual, expected):
        assert gate == ref_gate
        assert_exact(radius, ref_radius)


def test_disabled_config_and_grouping_unchanged(tmp_path):
    path = tmp_path / "compass.toml"
    path.write_text('run_name = "test"\n[optimizers.muon]\ncompass = true\n')
    resolved = load_experiment_config(path)
    assert resolved["optimizers"]["muon"] == dict(lr=.025, mu=.95, weight_decay=.05, compass=True)
    assert "compass" not in load_experiment_config()["optimizers"]["muon"]
    path.write_text('run_name = "test"\n[optimizers.muon]\ncompass = false\n')
    assert "compass" not in load_experiment_config(path)["optimizers"]["muon"]
    model = make_model(True)
    for a, b in zip(build(model), build(model, False)):
        assert_exact(a.state_dict(), b.state_dict())
    path.write_text('run_name = "test"\n[optimizers.muon]\ncompass = "yes"\n')
    with pytest.raises(ValueError, match="boolean"):
        load_experiment_config(path)
    path.write_text('run_name = "test"\n[optimizers.expert_muon]\n')
    with pytest.raises(ValueError, match="unknown"):
        load_experiment_config(path)


def test_multirank_compass_rejected(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="one rank"):
        build(make_model(True), True)
