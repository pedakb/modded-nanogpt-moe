"""CPU parity with the read-only sibling ExpertMuon reference (f829248)."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from modded_nanogpt_moe.config import load_experiment_config
from modded_nanogpt_moe.model import GPT
from modded_nanogpt_moe.optim import ExpertMuon, build_optimizers


@pytest.fixture
def reference(monkeypatch):
    source = Path(__file__).resolve().parents[2] / "ExpertMuon" / "src"
    if not source.is_dir():
        pytest.skip("reference parity requires the read-only sibling ExpertMuon checkout")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(source))
    from expertmuon_compass.config import OptimizerConfig, TelemetryConfig
    from expertmuon_compass.optim.state_owner import WholeMatrixStateOwnerOptimizer
    from expertmuon_compass.registry import ParameterRegistry

    def make(named):
        config = OptimizerConfig(method="compass", matrix_lr=0.0005)
        registry = ParameterRegistry.build(named, 1, config, TelemetryConfig())
        return WholeMatrixStateOwnerOptimizer(named, registry, config, total_steps=10)
    return make


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("experts", [3, 8, 64])
def test_reference_updates_and_momentum(reference, packed, dtype, experts):
    torch.manual_seed(421)
    # Tall and wide projections are distinct families. Packed storage is [E,in,out].
    refs = [[torch.nn.Parameter(torch.randn(*shape).to(dtype)) for _ in range(experts)]
            for shape in ((3, 5), (5, 3))]
    names = [[f"layers.0.experts.{i}.fc{role + 1}.weight" for i in range(experts)]
             for role in range(2)]
    oracle = reference([(name, p) for group_names, group in zip(names, refs)
                        for name, p in zip(group_names, group)])
    families = [dict(params=[torch.nn.Parameter(torch.stack([p.mT for p in group]).contiguous())],
                     transposed=True) if packed else
                dict(params=[torch.nn.Parameter(p.detach().clone()) for p in group])
                for group in refs]
    port = ExpertMuon(families)
    for step, ratio in enumerate((1.0, 0.7, 0.2, 0.0, 0.5)):
        oracle.set_lr(0.0005 * ratio, 0.00002 * ratio)
        for family in port.param_groups:
            family["lr"] = family["initial_lr"] * ratio
        for group, family in zip(refs, port.param_groups):
            for i, p in enumerate(group):
                p.grad = torch.randn_like(p)
                if i == 0 and step != 1:
                    p.grad.zero_()  # Both initially zero and zero after nonzero momentum.
            if packed:
                family["params"][0].grad = torch.stack([p.grad.mT for p in group]).contiguous()
            else:
                for p, other in zip(group, family["params"]):
                    other.grad = p.grad.clone()
        oracle.step()
        port.step()
        reference_state = oracle.state_dict()["manual_state"]
        for group_names, group, family in zip(names, refs, port.param_groups):
            actual = family["params"]
            weights = actual[0].mT.unbind() if packed else actual
            buffers = (port.state[actual[0]]["momentum_buffer"].unbind() if packed else
                       [port.state[p]["momentum_buffer"] for p in actual])
            for name, expected, weight, momentum in zip(group_names, group, weights, buffers):
                torch.testing.assert_close(weight, expected, rtol=0, atol=0)
                torch.testing.assert_close(momentum, reference_state[name]["momentum_buffer"], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_state_reload_preserves_fp32_momentum_and_next_update(dtype):
    torch.manual_seed(77)
    p = torch.nn.Parameter(torch.randn(3, 5, 2).to(dtype))
    optimizer = ExpertMuon([dict(params=[p], transposed=True)])
    for _ in range(3):
        p.grad = torch.randn_like(p)
        optimizer.step()
    restored_p = torch.nn.Parameter(p.detach().clone())
    restored = ExpertMuon([dict(params=[restored_p], transposed=True)])
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert restored.state[restored_p]["momentum_buffer"].dtype == torch.float32
    p.grad = torch.randn_like(p)
    restored_p.grad = p.grad.clone()
    optimizer.step()
    restored.step()
    torch.testing.assert_close(p, restored_p, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[p]["momentum_buffer"],
                               restored.state[restored_p]["momentum_buffer"], rtol=0, atol=0)


def test_missing_gradient_skips_state_and_decay():
    p = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = ExpertMuon([dict(params=[p])])
    optimizer.step()
    assert not optimizer.state
    assert torch.equal(p, torch.ones_like(p))


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("router", ["muon", "adamw"])
def test_builder_changes_only_expert_ownership(monkeypatch, packed, router):
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "extension")
    monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(ops=SimpleNamespace(gmm=None)))
    model = GPT(vocab_size=17, num_layers=1, model_dim=128, mlp_type="moe",
                num_experts=3, top_k=2, mlp_ratio=0.5,
                moe_backend="grouped_gemm" if packed else "loop",
                moe_parameter_layout="packed" if packed else "modulelist")
    config = load_experiment_config()["optimizers"]
    assert "expert_muon" not in config
    config["router_optimizer"] = router
    old_adam, old_muon = build_optimizers(model, config)
    config["expert_muon"] = {}
    adam, muon, expert_muon = build_optimizers(model, config)
    assert type(expert_muon) is ExpertMuon
    assert adam.state_dict() == old_adam.state_dict()
    assert [g["params"] for g in adam.param_groups] == [g["params"] for g in old_adam.param_groups]
    moe = model.blocks[0].mlp
    expected = ({moe.fc_weight, moe.proj_weight} if packed else
                {p for expert in moe.experts for p in (expert.fc.weight, expert.proj.weight)})
    actual = {p for group in expert_muon.param_groups for p in group["params"]}
    assert actual == expected
    assert len(expert_muon.param_groups) == 2
    assert [p for p in old_muon.param_groups[0]["params"] if p not in expected] == muon.param_groups[0]["params"]
    assigned = [p for opt in (adam, muon, expert_muon) for group in opt.param_groups for p in group["params"]]
    assert len(assigned) == len(set(assigned)) == len(list(model.parameters()))


def test_optional_config(tmp_path):
    path = tmp_path / "expert.toml"
    path.write_text('run_name = "expert"\n[optimizers.expert_muon]\nlr = 0.001\n')
    config = load_experiment_config(path)
    assert config["optimizers"]["expert_muon"] == dict(lr=0.001, momentum=0.95, weight_decay=0.01)
    assert "expert_muon" not in load_experiment_config()["optimizers"]
    for setting in ('lr = 0', 'momentum = 1', 'weight_decay = -1', 'lr = nan'):
        path.write_text('run_name = "expert"\n[optimizers.expert_muon]\n' + setting)
        with pytest.raises(ValueError):
            load_experiment_config(path)


def test_multirank_is_rejected_without_new_ownership_logic(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="one rank"):
        ExpertMuon([dict(params=[torch.nn.Parameter(torch.ones(2, 3))])])
