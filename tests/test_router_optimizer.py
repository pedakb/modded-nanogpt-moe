"""Router ownership ablation; no change to model backward or Muon mathematics."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from modded_nanogpt_moe.checkpoint import (
    CHECKPOINT_FORMAT_VERSION, make_training_checkpoint,
    restore_training_checkpoint, validate_checkpoint_config,
)
from modded_nanogpt_moe.config import load_experiment_config
from modded_nanogpt_moe.model import GPT, MoE
from modded_nanogpt_moe.optim import build_optimizers, moe_router_weights


def optimizer_config(owner="muon"):
    config = load_experiment_config()["optimizers"]
    config["router_optimizer"] = owner
    config["adamw"]["fused"] = False
    return config


def make_model(layout="modulelist", backward="standard", experts=8, k=2, dense=False):
    return GPT(vocab_size=37, num_layers=2, model_dim=128, mlp_ratio=0.5,
               mlp_type="dense" if dense else "moe", num_experts=experts, top_k=k,
               moe_backend="grouped_gemm" if layout == "packed" or backward == "grad_em" else "loop",
               moe_parameter_layout=layout, moe_backward=backward, grad_em_eta=0.01)


@pytest.fixture
def mock_extension(monkeypatch):
    # These tests construct optimizers only; no CUDA extension or forward is needed.
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "extension")
    monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(
        ops=SimpleNamespace(gmm=None)))


def groups(optimizers):
    return [[group["params"] for group in opt.param_groups] for opt in optimizers]


def assert_exact(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_exact(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_exact(a, b)
    else:
        assert actual == expected


def test_default_and_toml_router_optimizer(tmp_path):
    defaults = load_experiment_config()["optimizers"]
    assert defaults["router_optimizer"] == "muon"
    assert defaults["router_adamw_lr"] is None
    path = tmp_path / "router.toml"
    path.write_text(
        'run_name = "test"\n[optimizers]\n'
        'router_optimizer = "adamw"\nrouter_adamw_lr = 0.003\n')
    optimizers = load_experiment_config(path)["optimizers"]
    assert optimizers["router_optimizer"] == "adamw"
    assert optimizers["router_adamw_lr"] == 0.003


@pytest.mark.parametrize("value", ['"sgd"', '"AdamW"', "true", "1", "[]"])
def test_invalid_router_optimizer(tmp_path, value):
    path = tmp_path / "invalid.toml"
    path.write_text(f'run_name = "test"\n[optimizers]\nrouter_optimizer = {value}\n')
    with pytest.raises(ValueError, match="optimizers.router_optimizer"):
        load_experiment_config(path)
    with pytest.raises(ValueError, match="optimizers.router_optimizer"):
        build_optimizers(make_model(), optimizer_config("sgd"))


@pytest.mark.parametrize("value", ["0", "-0.1", "true", '"0.003"', "inf", "nan", "[]"])
def test_invalid_router_adamw_lr(tmp_path, value):
    path = tmp_path / "invalid.toml"
    path.write_text(
        f'run_name = "test"\n[optimizers]\nrouter_adamw_lr = {value}\n')
    with pytest.raises(ValueError, match="optimizers.router_adamw_lr"):
        load_experiment_config(path)

    config = optimizer_config("adamw")
    config["router_adamw_lr"] = 0
    with pytest.raises(ValueError, match="optimizers.router_adamw_lr"):
        build_optimizers(make_model(), config)


def test_dedicated_router_adamw_lr_preserves_all_other_ownership_and_lrs():
    model = make_model(layout="packed")
    muon = build_optimizers(model, optimizer_config("muon"))
    shared = build_optimizers(model, optimizer_config("adamw"))
    dedicated_config = optimizer_config("adamw")
    dedicated_config["router_adamw_lr"] = 0.003
    dedicated = build_optimizers(model, dedicated_config)

    routers = set(moe_router_weights(model))
    biases = {module.router.bias for module in model.modules() if isinstance(module, MoE)}
    assert len(shared[0].param_groups) == 3
    assert routers <= set(shared[0].param_groups[2]["params"])
    assert shared[0].param_groups[2]["lr"] == 0.015
    assert len(dedicated[0].param_groups) == 4
    assert set(dedicated[0].param_groups[3]["params"]) == routers
    assert dedicated[0].param_groups[3]["lr"] == 0.003
    assert biases <= set(dedicated[0].param_groups[2]["params"])
    assert dedicated[0].param_groups[2]["lr"] == 0.015

    muon_params = set(muon[1].param_groups[0]["params"])
    dedicated_muon_params = set(dedicated[1].param_groups[0]["params"])
    assert muon_params - routers == dedicated_muon_params
    for group_index in (0, 1):
        assert dedicated[0].param_groups[group_index]["params"] == muon[0].param_groups[group_index]["params"]
        assert dedicated[0].param_groups[group_index]["lr"] == muon[0].param_groups[group_index]["lr"]
    assert set(dedicated[0].param_groups[2]["params"]) == set(muon[0].param_groups[2]["params"])

    for optimizers in (muon, shared, dedicated):
        assigned = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
        assert len(assigned) == len(set(assigned)) == len(list(model.parameters()))
        assert set(assigned) == set(model.parameters())


def test_router_adamw_lr_does_not_affect_muon_ownership():
    model = make_model()
    baseline = build_optimizers(model, optimizer_config("muon"))
    configured = optimizer_config("muon")
    configured["router_adamw_lr"] = 0.003
    with_router_lr = build_optimizers(model, configured)
    assert groups(baseline) == groups(with_router_lr)
    for a, b in zip(baseline, with_router_lr):
        assert_exact(a.state_dict(), b.state_dict())


@pytest.mark.parametrize("layout", ["modulelist", "packed"])
@pytest.mark.parametrize("backward", ["standard", "grad_em"])
@pytest.mark.parametrize("experts,k", [(8, 2), (64, 8)])
def test_only_router_weights_move_and_all_parameters_occur_once(mock_extension, layout, backward, experts, k):
    model = make_model(layout, backward, experts, k)
    default_config = optimizer_config()
    del default_config["router_optimizer"]  # Legacy direct builder calls.
    default = build_optimizers(model, default_config)
    muon = build_optimizers(model, optimizer_config("muon"))
    adamw = build_optimizers(model, optimizer_config("adamw"))
    assert groups(default) == groups(muon)
    for a, b in zip(default, muon):
        assert_exact(a.state_dict(), b.state_dict())

    routers = set(moe_router_weights(model))
    assert len(routers) == 2
    assert {name for name, p in model.named_parameters() if p in routers} == {
        "blocks.0.mlp.router.weight", "blocks.1.mlp.router.weight"}
    before, after = groups(muon), groups(adamw)
    assert before[0][:2] == after[0][:2]  # Embedding and head groups.
    assert [p for p in after[0][2] if p not in routers] == before[0][2]
    assert [p for p in before[1][0] if p not in routers] == after[1][0]
    assert set(after[0][2]) - set(before[0][2]) == routers
    assert set(before[1][0]) - set(after[1][0]) == routers
    for opt_before, opt_after in zip(muon, adamw):
        for a, b in zip(opt_before.param_groups, opt_after.param_groups):
            assert {key: v for key, v in a.items() if key != "params"} == {
                key: v for key, v in b.items() if key != "params"}
    for optimizers, owner in ((muon, 1), (adamw, 0)):
        all_params = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
        assert len(all_params) == len(set(all_params)) == len(list(model.parameters()))
        assert set(all_params) == set(model.parameters())
        assert routers <= {p for group in optimizers[owner].param_groups for p in group["params"]}
        for module in model.modules():
            if isinstance(module, MoE):
                assert module.router.bias in set(optimizers[0].param_groups[2]["params"])
    assert muon[1].transposed_params == adamw[1].transposed_params


def test_dense_groups_unchanged():
    model = make_model(dense=True)
    assert moe_router_weights(model) == []
    before = build_optimizers(model, optimizer_config("muon"))
    after = build_optimizers(model, optimizer_config("adamw"))
    assert groups(before) == groups(after)
    for a, b in zip(before, after):
        assert_exact(a.state_dict(), b.state_dict())


def test_duplicate_ownership_is_rejected():
    model = make_model()
    model.embed.weight = model.blocks[0].mlp.router.weight
    with pytest.raises(AssertionError, match="ownership must be exclusive"):
        build_optimizers(model, optimizer_config("muon"))


@pytest.mark.parametrize("saved_owner", [None, "muon", "adamw"])
def test_router_ownership_checkpoint_restore(tmp_path, saved_owner):
    model = make_model()
    config = optimizer_config(saved_owner or "muon")
    if saved_owner is None:
        del config["router_optimizer"]
        del config["router_adamw_lr"]
    optimizers = build_optimizers(model, config)
    # Exercise real AdamW moments (including router weights in ablation mode).
    for group in optimizers[0].param_groups:
        group["lr"] = group["initial_lr"] * 0.5
        for p in group["params"]:
            p.grad = torch.randn_like(p)
    optimizers[0].step()
    optimizers[0].zero_grad(set_to_none=True)
    for p in optimizers[1].param_groups[0]["params"]:
        optimizers[1].state[p]["momentum"] = torch.randn_like(p)
    loader = SimpleNamespace(state_dict=lambda: {"token_offset": 8})
    resolved = {"optimizers": config}
    checkpoint = make_training_checkpoint(
        model, optimizers, 1, 8, resolved, loader, "test", 0, 1., 1., 0, {})
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    checkpoint = torch.load(path, weights_only=False)
    restored_model = make_model()
    current = {"optimizers": optimizer_config(saved_owner or "muon")}
    restored_optimizers = build_optimizers(restored_model, current["optimizers"])
    loader_state = {}
    restored_loader = SimpleNamespace(load_state_dict=loader_state.update)
    restore_training_checkpoint(checkpoint, current, restored_model,
                                restored_optimizers, restored_loader, 20, 8)
    assert loader_state == {"token_offset": 8}
    assert_exact(restored_model.state_dict(), model.state_dict())
    for a, b in zip(restored_optimizers, optimizers):
        assert_exact(a.state_dict(), b.state_dict())
    assert ("router_optimizer" in checkpoint["resolved_config"]["optimizers"]) == (saved_owner is not None)
    # The next AdamW update must use restored moments and scheduled LR, not restart.
    for a, b in zip(restored_optimizers[0].param_groups, optimizers[0].param_groups):
        for restored_p, original_p in zip(a["params"], b["params"]):
            original_p.grad = torch.randn_like(original_p)
            restored_p.grad = original_p.grad.clone()
    optimizers[0].step()
    restored_optimizers[0].step()
    assert_exact(restored_model.state_dict(), model.state_dict())
    assert_exact(restored_optimizers[0].state_dict(), optimizers[0].state_dict())


@pytest.mark.parametrize("saved_owner,current_owner", [(None, "adamw"), ("muon", "adamw"), ("adamw", "muon")])
def test_resume_rejects_changed_ownership(saved_owner, current_owner):
    saved = {"optimizers": optimizer_config(saved_owner)}
    if saved_owner is None:
        del saved["optimizers"]["router_optimizer"]
        del saved["optimizers"]["router_adamw_lr"]
    checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION, "resolved_config": saved}
    with pytest.raises(ValueError, match="configuration is incompatible"):
        validate_checkpoint_config(checkpoint, {"optimizers": optimizer_config(current_owner)})


def test_legacy_checkpoint_without_router_adamw_lr_is_compatible():
    saved = {"optimizers": optimizer_config("adamw")}
    del saved["optimizers"]["router_adamw_lr"]
    checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION, "resolved_config": saved}
    validate_checkpoint_config(
        checkpoint, {"optimizers": optimizer_config("adamw")})


def test_resume_rejects_changed_dedicated_router_lr():
    saved = optimizer_config("adamw")
    saved["router_adamw_lr"] = 0.003
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "resolved_config": {"optimizers": saved},
    }
    with pytest.raises(ValueError, match="configuration is incompatible"):
        validate_checkpoint_config(
            checkpoint, {"optimizers": optimizer_config("adamw")})


def test_ablation_config_changes_only_requested_settings():
    root = Path(__file__).resolve().parents[1] / "configs"
    reference = load_experiment_config(root / "moe_e64k8_r0.5_gradem_eta0.1.toml")
    expected = copy.deepcopy(reference)
    expected["run_name"] = "moe-e64k8-r0.5-gradem-eta0.01-router-adamw"
    expected["model"]["grad_em_eta"] = 0.01
    expected["optimizers"]["router_optimizer"] = "adamw"
    expected["checkpoint"]["interval"] = 50
    actual = load_experiment_config(root / "moe_e64k8_r0.5_gradem_eta0.01_router_adamw.toml")
    assert actual == expected
    assert actual["training"]["total_steps"] == 3250


def test_dedicated_lr_config_changes_only_router_lr_and_run_name():
    root = Path(__file__).resolve().parents[1] / "configs"
    reference = load_experiment_config(
        root / "moe_e64k8_r0.5_gradem_eta0.01_router_adamw.toml")
    expected = copy.deepcopy(reference)
    expected["run_name"] = "ge-eta01-adamw-lr003"
    expected["optimizers"]["router_adamw_lr"] = 0.003
    actual = load_experiment_config(
        root / "moe_e64k8_r0.5_gradem_eta0.01_router_adamw_lr0.003.toml")
    assert actual == expected
