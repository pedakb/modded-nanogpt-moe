"""Global interpolation and recursive multi-layer gradient propagation."""
import copy
from pathlib import Path

import pytest
import torch

from modded_nanogpt_moe.checkpoint import CHECKPOINT_FORMAT_VERSION, validate_checkpoint_config
from modded_nanogpt_moe.config import load_experiment_config, validate_experiment_config
from modded_nanogpt_moe.model import MoE
from test_grad_em_grouped_local_bp import (
    cpu_backend, make_models, run, canonical_grads, compare_pair, require_cuda_backend, CUDA,
)


def check_interpolation(dtype, e, k, normalize, device="cpu", need_x=True):
    _, _, bp, full = make_models(e=e, k=k, normalize=normalize, device=device)
    value = torch.randn(2, 5, 16, device=device, dtype=dtype)
    upstream = torch.randn_like(value)
    ordinary = run(bp, value, upstream, need_x)
    ge = run(full, value, upstream, need_x)
    for coefficient in (0., 0.5, 1.):
        mixed = copy.deepcopy(full)
        mixed.zero_grad(set_to_none=True)
        mixed.grad_em_lambda = coefficient
        actual = run(mixed, value, upstream, need_x)
        torch.testing.assert_close(actual[0], ge[0], atol=0, rtol=0)
        if need_x:
            if coefficient in (0., 1.):
                torch.testing.assert_close(actual[1], (ordinary if coefficient == 0 else ge)[1], atol=0, rtol=0)
            else:
                compare_pair(actual[1], ordinary[1] + coefficient * (ge[1] - ordinary[1]), dtype)
        for name, gradient in actual[2].items():
            expected = ordinary[2][name] + coefficient * (ge[2][name] - ordinary[2][name])
            if coefficient in (0., 1.):
                expected = (ordinary if coefficient == 0 else ge)[2][name]
                torch.testing.assert_close(gradient, expected, atol=0, rtol=0)
            else:
                compare_pair(gradient, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("e,k", [(8, 2), (64, 8)])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("need_x", [True, False])
def test_cpu_interpolation(cpu_backend, dtype, e, k, normalize, need_x):
    check_interpolation(dtype, e, k, normalize, need_x=need_x)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixed_combine_strides_and_zero_forward_weight(device, dtype):
    from modded_nanogpt_moe.grad_em import GradEMCombine, grad_em_reference
    torch.manual_seed(65)
    n, k, e, d = 7, 2, 8, 19
    logits = torch.randn(n, e * 2, device=device, dtype=dtype)[:, ::2].requires_grad_()
    indices = logits.detach().float().topk(k).indices
    weights = torch.rand(k, n, device=device, dtype=dtype).T
    weights[0, -1] = 0
    weights.requires_grad_()
    order = indices.flatten().argsort(stable=True)
    out = torch.randn(n*k, d*2, device=device, dtype=dtype)[:, ::2].requires_grad_()
    g = torch.randn(n, d*2, device=device, dtype=dtype)[:, ::2]
    result = GradEMCombine.apply(out, logits, weights, indices, order, 0.4, None, 0.5)
    actual = torch.autograd.grad(result, (out, logits, weights), g)
    selected = torch.empty_like(out)
    selected[order] = out.detach()
    selected = selected.view(n, k, d)
    ge = grad_em_reference(logits, indices, selected, g, 0.4)
    mixed = weights.float() + 0.5 * (ge.q - weights.float())
    expected = ((mixed[..., None] * g.float()[:, None]).flatten(0, 1)[order].to(dtype),
                ge.grad_logits.to(dtype) * 0.5,
                (selected * g[:, None]).sum(-1) * 0.5)
    for gradient, oracle in zip(actual, expected):
        assert torch.isfinite(gradient).all()
        compare_pair(gradient, oracle, dtype)


@CUDA
@pytest.mark.parametrize("implementation", ["torch", "extension"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("e,k", [(8, 2), (64, 8)])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("need_x", [True, False])
def test_cuda_interpolation(monkeypatch, implementation, dtype, e, k, normalize, need_x):
    require_cuda_backend(monkeypatch, implementation, dtype)
    check_interpolation(dtype, e, k, normalize, device="cuda", need_x=need_x)


@CUDA
@pytest.mark.parametrize("implementation", ["torch", "extension"])
def test_cuda_interpolation_gemm_counts(monkeypatch, implementation):
    from tools.benchmark_grad_em import count_grouped_calls
    require_cuda_backend(monkeypatch, implementation, torch.bfloat16)
    _, _, _, model = make_models(device="cuda")
    x = torch.randn(2, 7, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(x)
    expected = {f"{phase}.{kind}": 1 for phase in ("fc1", "fc2")
                for kind in ("forward", "dx", "dw")}
    for coefficient in (0., 0.5, 1.):
        model.grad_em_lambda = coefficient
        assert count_grouped_calls(model, x, g) == expected


def reference_mixed_run(template, value, upstream):
    """Independent PyTorch signal oracle on an ordinary BP forward graph."""
    from modded_nanogpt_moe.grad_em import grad_em_reference
    model = copy.deepcopy(template)
    model.moe_backward = "standard"

    def install(logits, indices, weights, sorted_outputs, order, output, x):
        signals = {}

        def capture(g):
            selected = torch.empty_like(sorted_outputs)
            selected[order] = sorted_outputs.detach()
            selected = selected.view(*indices.shape, sorted_outputs.shape[-1])
            g = g.flatten(0, 1)
            ge = grad_em_reference(logits, indices, selected, g, model.grad_em_eta)
            mixed = weights.float() + 0.5 * (ge.q - weights.float())
            signals["expert"] = (mixed[..., None] * g.float()[:, None, :]).flatten(0, 1)[order].to(sorted_outputs.dtype)
            signals["router"] = ge.grad_logits.to(logits.dtype) * 0.5

        output.register_hook(capture)
        sorted_outputs.register_hook(lambda _: signals["expert"])
        weights.register_hook(lambda g: g * 0.5)
        logits.register_hook(lambda g: g + signals["router"])

    model._grad_em_diagnostics = install
    return run(model, value, upstream)


def check_recursive(dtype, normalize, device="cpu"):
    _, local, bp, global_em = make_models(normalize=normalize, device=device)
    value = torch.randn(2, 5, 16, device=device, dtype=dtype)
    upstream = 3 * torch.randn_like(value)
    results = {}
    for label, template, coefficient in (("bp", bp, 1.), ("zero", global_em, 0.),
                                         ("half", global_em, 0.5), ("ge", global_em, 1.),
                                         ("local", local, 1.)):
        first, second = copy.deepcopy(template), copy.deepcopy(template)
        first.grad_em_lambda = second.grad_em_lambda = coefficient
        x = value.clone().requires_grad_()
        middle = first(x)
        middle.retain_grad()
        second(middle.tanh()).backward(upstream)
        results[label] = (x.grad, middle.grad, canonical_grads(first), canonical_grads(second))

    # Independently mix signals on a BP graph for the SAME incoming g; feed
    # the resulting VJP into the preceding layer. Mixing before GEMM matters
    # in BF16: a mixture of two already-rounded VJPs is not an exact oracle.
    with torch.no_grad():
        middle_value = bp(value)
    middle = middle_value.detach().requires_grad_()
    activated = middle.tanh()
    top = reference_mixed_run(bp, activated.detach(), upstream)
    incoming, = torch.autograd.grad(activated, middle, top[1])
    bottom = reference_mixed_run(bp, value, incoming)
    compare_pair(results["half"][0], bottom[1], dtype)
    compare_pair(results["half"][1], incoming, dtype)
    for index, expected in ((2, bottom), (3, top)):
        for name in expected[2]:
            compare_pair(results["half"][index][name], expected[2][name], dtype)
            torch.testing.assert_close(results["zero"][index][name], results["bp"][index][name], atol=0, rtol=0)
    for index in (0, 1):
        torch.testing.assert_close(results["zero"][index], results["bp"][index], atol=0, rtol=0)
        torch.testing.assert_close(results["local"][index], results["bp"][index], atol=0, rtol=0)
        assert not torch.allclose(results["half"][index].float(), results["local"][index].float())
        assert not torch.allclose(results["half"][index].float(), results["ge"][index].float())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
def test_cpu_recursive_global_not_local(cpu_backend, dtype, normalize):
    check_recursive(dtype, normalize)


@CUDA
@pytest.mark.parametrize("implementation", ["torch", "extension"])
@pytest.mark.parametrize("normalize", [False, True])
def test_cuda_recursive_global_not_local(monkeypatch, implementation, normalize):
    require_cuda_backend(monkeypatch, implementation, torch.bfloat16)
    check_recursive(torch.bfloat16, normalize, "cuda")


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), -float("inf"), True, "0.5", 10**400])
def test_invalid_lambda(value):
    config = load_experiment_config()
    config["model"]["grad_em_lambda"] = value
    with pytest.raises(ValueError, match="grad_em_lambda"):
        validate_experiment_config(config)
    with pytest.raises(ValueError, match="grad_em_lambda"):
        MoE(16, 8, 2, grad_em_lambda=value)


def test_lambda_defaults_loading_and_checkpoint_compatibility(tmp_path):
    reference = "configs/moe_e64k8_r0.5_gradem_local_bp_eta0.01.toml"
    config = load_experiment_config(reference)
    assert config["model"]["grad_em_lambda"] == 1.
    config["model"]["grad_em_mode"] = "global"
    legacy = copy.deepcopy(config)
    del legacy["model"]["grad_em_lambda"]
    checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION, "resolved_config": legacy}
    validate_checkpoint_config(checkpoint, config)
    path = tmp_path / "mixed.toml"
    path.write_text(Path(reference).read_text().replace(
        'grad_em_mode = "local_bp"', 'grad_em_mode = "global"').replace(
        'grad_em_eta = 0.01', 'grad_em_eta = 0.01\ngrad_em_lambda = 0.5'))
    mixed = load_experiment_config(path)
    assert mixed["model"]["grad_em_lambda"] == 0.5
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_config(checkpoint, mixed)
    mixed["model"]["grad_em_mode"] = "local_bp"
    with pytest.raises(ValueError, match="requires grad_em_mode"):
        validate_experiment_config(mixed)
