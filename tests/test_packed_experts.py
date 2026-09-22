"""Packed experts versus the reference storage, including optimizer state."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modded_nanogpt_moe import optim as optim_module
from modded_nanogpt_moe.checkpoint import (
    CHECKPOINT_FORMAT_VERSION, atomic_save_checkpoint, make_training_checkpoint,
    restore_training_checkpoint, validate_checkpoint_config,
)
from modded_nanogpt_moe.config import load_experiment_config, validate_experiment_config
from modded_nanogpt_moe.model import GPT, MoE, initialize_model_parameters
from modded_nanogpt_moe.optim import Muon, build_optimizers

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))]


@pytest.fixture
def cpu_gmm(monkeypatch):
    def gmm(x, weight, counts, trans_b=False):
        assert not trans_b
        return torch.cat([segment @ matrix for segment, matrix in
                          zip(x.split(counts.tolist()), weight)])
    monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(ops=SimpleNamespace(gmm=gmm)))
    return gmm


def parameter_pairs(reference, packed):
    yield reference.router.weight, packed.router.weight
    yield reference.router.bias, packed.router.bias
    for index, expert in enumerate(reference.experts):
        yield expert.fc.weight, packed.fc_weight[index].mT
        yield expert.fc.bias, packed.fc_bias[index]
        yield expert.proj.weight, packed.proj_weight[index].mT
        yield expert.proj.bias, packed.proj_bias[index]


@torch.no_grad()
def copy_experts(reference, packed):
    for source, destination in parameter_pairs(reference, packed):
        destination.copy_(source)


def make_pair(e, k, d=16, h=8, device="cpu", dtype=torch.float32):
    reference = MoE(d, e, k, hidden_dim=h, moe_backend="grouped_gemm").to(device=device, dtype=dtype)
    packed = MoE(d, e, k, hidden_dim=h, moe_backend="grouped_gemm",
                 moe_parameter_layout="packed").to(device=device, dtype=dtype)
    copy_experts(reference, packed)
    return reference, packed


@pytest.mark.parametrize("e,k,ratio", [(8, 2, 2), (64, 8, 0.5)])
@pytest.mark.parametrize("device", DEVICES)
def test_packed_initialization_matches_reference_and_rng(cpu_gmm, e, k, ratio, device):
    kwargs = dict(vocab_size=37, num_layers=2, model_dim=128, mlp_type="moe",
                  mlp_ratio=ratio, num_experts=e, top_k=k, moe_backend="grouped_gemm")
    torch.manual_seed(123)
    reference = GPT(**kwargs).to(device)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    packed = GPT(**kwargs, moe_parameter_layout="packed").to(device)
    assert torch.equal(rng, torch.get_rng_state())
    device_rng = torch.cuda.get_rng_state if device == "cuda" else torch.get_rng_state
    for reinitialize in (False, True):
        if reinitialize:
            torch.manual_seed(456)
            initialize_model_parameters(reference)
            rng = device_rng().clone()
            torch.manual_seed(456)
            initialize_model_parameters(packed)
            assert torch.equal(rng, device_rng())
        for ref_block, packed_block in zip(reference.blocks, packed.blocks):
            for source, destination in parameter_pairs(ref_block.mlp, packed_block.mlp):
                torch.testing.assert_close(destination, source, rtol=0, atol=0)
        for name, value in reference.state_dict().items():
            if ".mlp." not in name:
                torch.testing.assert_close(packed.state_dict()[name], value, rtol=0, atol=0)


def check_forward_backward(reference, packed, x, atol=0, rtol=0):
    with torch.no_grad():
        reference.router.bias[-1] = -1e4  # Empty expert with zero, present gradients.
    copy_experts(reference, packed)
    x_ref, x_packed = x.clone().requires_grad_(), x.clone().requires_grad_()
    output_ref, output_packed = reference(x_ref), packed(x_packed)
    grad_output = torch.randn_like(output_ref)
    output_ref.backward(grad_output)
    output_packed.backward(grad_output)
    torch.testing.assert_close(output_packed, output_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(x_packed.grad, x_ref.grad, atol=atol, rtol=rtol)
    for source, destination in ((reference.router.weight, packed.router.weight),
                                (reference.router.bias, packed.router.bias)):
        torch.testing.assert_close(destination.grad, source.grad, atol=atol, rtol=rtol)
    for i, expert in enumerate(reference.experts):
        for old, new in ((expert.fc.weight.grad, packed.fc_weight.grad[i].mT),
                         (expert.proj.weight.grad, packed.proj_weight.grad[i].mT),
                         (expert.fc.bias.grad, packed.fc_bias.grad[i]),
                         (expert.proj.bias.grad, packed.proj_bias.grad[i])):
            torch.testing.assert_close(new, old, atol=atol, rtol=rtol)
            if i == len(reference.experts) - 1:
                assert torch.count_nonzero(new) == 0


@pytest.mark.parametrize("e,k,h", [(8, 2, 32), (64, 8, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_forward_gradients_cpu(cpu_gmm, e, k, h, dtype):
    torch.manual_seed(7)
    reference, packed = make_pair(e, k, h=h)
    check_forward_backward(reference, packed, torch.randn(2, 9, 16, dtype=dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and grouped_gemm")
@pytest.mark.parametrize("e,k,h", [(8, 2, 1536), (64, 8, 384)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32_master", "bf16"])
def test_packed_forward_gradients_cuda(e, k, h, dtype):
    pytest.importorskip("grouped_gemm")
    torch.manual_seed(7)
    reference, packed = make_pair(e, k, d=768, h=h, device="cuda", dtype=dtype)
    check_forward_backward(reference, packed,
                           torch.randn(1, 32, 768, device="cuda", dtype=torch.bfloat16),
                           atol=2e-2, rtol=2e-2)


def test_packed_forward_uses_storage_without_stacking(cpu_gmm, monkeypatch):
    _, packed = make_pair(8, 2)
    pointers = []
    def gmm(x, weight, counts, trans_b=False):
        pointers.append(weight.data_ptr())
        assert weight.is_contiguous()
        return cpu_gmm(x, weight, counts, trans_b)
    packed._gmm = gmm
    def forbidden(*args, **kwargs):
        pytest.fail("packed forward reconstructed weights")
    monkeypatch.setattr(torch, "stack", forbidden)
    monkeypatch.setattr(torch.Tensor, "contiguous", forbidden)
    monkeypatch.setattr(torch.Tensor, "transpose", forbidden)
    packed(torch.randn(1, 8, 16))
    assert pointers == [packed.fc_weight.data_ptr(), packed.proj_weight.data_ptr()]


def test_packed_config_and_optimizer_assignment(cpu_gmm):
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/moe_e64k8_r0.5.toml")
    assert config["run_name"] == "moe-e64k8-r0.5"
    assert config["model"]["moe_parameter_layout"] == "packed"
    assert config["diagnostics"]["scalar_interval"] == 25
    model = GPT(vocab_size=37, num_layers=1, model_dim=128, mlp_type="moe",
                num_experts=8, top_k=2, moe_backend="grouped_gemm",
                moe_parameter_layout="packed")
    adamw, muon = build_optimizers(model)
    adam_params = {p for group in adamw.param_groups for p in group["params"]}
    muon_params = {p for group in muon.param_groups for p in group["params"]}
    moe = model.blocks[0].mlp
    assert {moe.fc_bias, moe.proj_bias} <= adam_params
    assert {moe.fc_weight, moe.proj_weight} <= muon_params
    assert adam_params.isdisjoint(muon_params)
    assert muon.transposed_params == {moe.fc_weight, moe.proj_weight}


@pytest.mark.parametrize("mlp_type,backend,layout", [
    ("dense", "grouped_gemm", "packed"), ("moe", "loop", "packed"),
    ("moe", "grouped_gemm", "invalid"),
])
def test_invalid_layout_configuration(mlp_type, backend, layout):
    config = load_experiment_config()
    config["model"].update(mlp_type=mlp_type, moe_backend=backend, moe_parameter_layout=layout)
    with pytest.raises(ValueError):
        validate_experiment_config(config)


def test_checkpoint_layout_is_explicit_and_incompatible():
    legacy = {"model": {"moe_backend": "grouped_gemm"}}
    packed = copy.deepcopy(legacy)
    packed["model"]["moe_parameter_layout"] = "packed"
    for old, new in ((legacy, packed), (packed, legacy)):
        checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION, "resolved_config": old}
        validate_checkpoint_config(checkpoint, copy.deepcopy(old))
        with pytest.raises(ValueError, match="incompatible"):
            validate_checkpoint_config(checkpoint, new)


def optimizer_holder(moe):
    model = nn.Module()
    model.embed = nn.Embedding(4, 16)
    model.proj = nn.Linear(16, 4)
    model.blocks = nn.ModuleList([moe])
    return model


@pytest.mark.parametrize("e,k", [(8, 2), (64, 8)])
@pytest.mark.parametrize("device", DEVICES)
def test_packed_adamw_bias_step_and_state(cpu_gmm, e, k, device):
    torch.manual_seed(11)
    reference, packed = make_pair(e, k, device=device)
    old_opt = build_optimizers(optimizer_holder(reference))[0]
    new_opt = build_optimizers(optimizer_holder(packed))[0]
    # Both cold-start and established moment estimates, with an empty expert.
    for _ in range(2):
        for attr, layer in (("fc_bias", "fc"), ("proj_bias", "proj")):
            parameter = getattr(packed, attr)
            parameter.grad = torch.randn_like(parameter)
            parameter.grad[-1].zero_()
            for i, expert in enumerate(reference.experts):
                getattr(expert, layer).bias.grad = parameter.grad[i].clone()
        old_opt.step()
        new_opt.step()
        for attr, layer in (("fc_bias", "fc"), ("proj_bias", "proj")):
            parameter = getattr(packed, attr)
            for i, expert in enumerate(reference.experts):
                old_parameter = getattr(expert, layer).bias
                torch.testing.assert_close(parameter[i], old_parameter, rtol=0, atol=0)
                for name in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(new_opt.state[parameter][name][i],
                                               old_opt.state[old_parameter][name], rtol=0, atol=0)
                torch.testing.assert_close(new_opt.state[parameter]["step"],
                                           old_opt.state[old_parameter]["step"], rtol=0, atol=0)


@pytest.mark.parametrize("e,k,h", [(8, 2, 32), (64, 8, 8)])
@pytest.mark.parametrize("device", DEVICES)
def test_packed_muon_step_and_momentum(cpu_gmm, monkeypatch, e, k, h, device):
    torch.manual_seed(12)
    reference, packed = make_pair(e, k, h=h, device=device)
    old_parameters = [p for expert in reference.experts
                      for p in (expert.fc.weight, expert.proj.weight)]
    new_parameters = [packed.fc_weight, packed.proj_weight]
    old_opt = Muon(old_parameters, lr=0.025, weight_decay=0.05)
    new_opt = Muon(new_parameters, lr=0.025, weight_decay=0.05,
                   transposed_params=new_parameters)
    monkeypatch.setattr(optim_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(optim_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(optim_module.dist, "all_gather", lambda outputs, value: None)
    # Use the actual compiled muon_update on both layouts.
    for _ in range(2):
        for attr, layer in (("fc_weight", "fc"), ("proj_weight", "proj")):
            parameter = getattr(packed, attr)
            parameter.grad = torch.randn_like(parameter)
            parameter.grad[-1].zero_()
            for i, expert in enumerate(reference.experts):
                getattr(expert, layer).weight.grad = parameter.grad[i].mT.contiguous()
        old_opt.step()
        new_opt.step()
        for attr, layer in (("fc_weight", "fc"), ("proj_weight", "proj")):
            parameter = getattr(packed, attr)
            for i, expert in enumerate(reference.experts):
                old_parameter = getattr(expert, layer).weight
                torch.testing.assert_close(parameter[i].mT, old_parameter, rtol=0, atol=0)
                torch.testing.assert_close(new_opt.state[parameter]["momentum"][i].mT,
                                           old_opt.state[old_parameter]["momentum"], rtol=0, atol=0)
    # State loading keeps the packed momentum and orientation metadata usable.
    restored_parameters = [nn.Parameter(p.detach().clone()) for p in new_parameters]
    restored = Muon(restored_parameters, transposed_params=restored_parameters)
    restored.load_state_dict(copy.deepcopy(new_opt.state_dict()))
    for p, restored_p in zip(new_parameters, restored_parameters):
        torch.testing.assert_close(restored.state[restored_p]["momentum"],
                                   new_opt.state[p]["momentum"], rtol=0, atol=0)
        p.grad = torch.randn_like(p)
        restored_p.grad = p.grad.clone()
    restored.step()
    new_opt.step()
    for p, restored_p in zip(new_parameters, restored_parameters):
        torch.testing.assert_close(restored_p, p, rtol=0, atol=0)


def test_packed_muon_distributed_ownership_and_gather(monkeypatch):
    # CPU simulation checks rank ownership and gather ordering, not NCCL itself.
    torch.manual_seed(19)
    values = [torch.randn(3, 8, 4) for _ in range(3)] + [torch.randn(4, 8), torch.randn(5, 7)]
    gradients = [torch.randn_like(value) for value in values]
    current_rank = 0
    world_size = 1
    gathers = [[], []]
    monkeypatch.setattr(optim_module.dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(optim_module.dist, "get_rank", lambda: current_rank)
    monkeypatch.setattr(optim_module.dist, "all_gather", lambda outputs, value: None)

    def run():
        params = [nn.Parameter(value.clone()) for value in values]
        opt = Muon(params, transposed_params=params[:3])
        for p, grad in zip(params, gradients):
            p.grad = grad.clone()
        opt.step()
        return opt, params

    _, reference = run()
    world_size = 2
    def gather(outputs, value):
        assert outputs[current_rank] is value
        gathers[current_rank].append((outputs, value))
    monkeypatch.setattr(optim_module.dist, "all_gather", gather)
    ranks = []
    for current_rank in range(world_size):
        opt, params = run()
        assert set(opt.state) == set(opt.param_groups[0]["params"][current_rank::world_size])
        ranks.append(params)
    assert len(gathers[0]) == len(gathers[1]) == 3
    with torch.no_grad():
        for first, second in zip(*gathers):
            sources = [first[1].clone(), second[1].clone()]
            for outputs, _ in (first, second):
                for output, source in zip(outputs, sources):
                    output.copy_(source)
    for params in ranks:
        for p, expected in zip(params, reference):
            torch.testing.assert_close(p, expected, rtol=0, atol=0)


def test_packed_training_checkpoint_restores_model_and_optimizer_layout(cpu_gmm, tmp_path):
    class Loader:
        state = {"cursor": 7}
        def state_dict(self):
            return self.state
        def load_state_dict(self, state):
            self.state = state

    def construct():
        moe = MoE(16, 8, 2, hidden_dim=8, moe_backend="grouped_gemm",
                  moe_parameter_layout="packed")
        model = optimizer_holder(moe)
        return model, build_optimizers(model)

    original, optimizers = construct()
    # Materialize AdamW moments and nonzero packed Muon state.
    for p in original.parameters():
        p.grad = torch.randn_like(p)
    optimizers[0].step()
    for p in optimizers[1].param_groups[0]["params"]:
        optimizers[1].state[p]["momentum"] = torch.randn_like(p)
    original.zero_grad(set_to_none=True)
    config = {"model": {"moe_parameter_layout": "packed"}, "training": {"total_steps": 20}}
    checkpoint = make_training_checkpoint(
        original, optimizers, completed_updates=2, batch_size=16,
        resolved_config=config, train_loader=Loader(), run_id="packed-test", trial_idx=0,
        training_time=1.0, current_segment_time=0.5, last_val_step=0,
        environment_metadata={})
    path = atomic_save_checkpoint(checkpoint, tmp_path)
    loaded = torch.load(path, weights_only=False)
    restored, restored_opts = construct()
    loader = Loader()
    loader.state = {}
    result = restore_training_checkpoint(loaded, config, restored, restored_opts,
                                         loader, train_steps=20, batch_size=16)
    assert result["completed_updates"] == 2
    assert loader.state == {"cursor": 7}
    for name, p in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], p, rtol=0, atol=0)
    for old, new in zip(optimizers, restored_opts):
        old_state, new_state = old.state_dict(), new.state_dict()
        assert old_state["param_groups"] == new_state["param_groups"]
        for index, state in old_state["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(new_state["state"][index][key], value, rtol=0, atol=0)
    moe = restored.blocks[0]
    assert restored_opts[1].transposed_params == {moe.fc_weight, moe.proj_weight}
