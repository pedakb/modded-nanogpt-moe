"""Two-coefficient local signal and recursive boundary contracts."""
import copy

import pytest
import torch
from torch import nn

from modded_nanogpt_moe.config import load_experiment_config, validate_experiment_config
from modded_nanogpt_moe.checkpoint import CHECKPOINT_FORMAT_VERSION, validate_checkpoint_config
from modded_nanogpt_moe.model import MoE
from test_grad_em_grouped_local_bp import (
    cpu_backend, make_models, run, canonical_grads, compare_pair, require_cuda_backend, CUDA,
)


CASES = [(0., 0.), (0., 1.), (1., 0.), (1., 1.), (0.5, 1.),
         (0.5, 0.), (1., 0.5), (0.5, 0.5)]


def signal_reference(template, value, g, coefficient):
    """Independent BP graph with explicitly replaced boundary signals."""
    from modded_nanogpt_moe.grad_em import grad_em_reference
    model = copy.deepcopy(template)
    model.moe_backward = "standard"
    model.zero_grad(set_to_none=True)
    if coefficient == 0:
        return run(model, value, g)

    def install(logits, indices, weights, outputs, order, output, x):
        signals = {}

        def capture(incoming):
            selected = torch.empty_like(outputs)
            selected[order] = outputs.detach()
            selected = selected.view(*indices.shape, outputs.shape[-1])
            incoming = incoming.flatten(0, 1)
            ge = grad_em_reference(logits, indices, selected, incoming, model.grad_em_eta)
            p = weights.float()
            mixed = p + coefficient * (ge.q - p)
            signals["expert"] = (mixed[..., None] * incoming.float()[:, None]).flatten(0, 1)[order].to(outputs.dtype)
            signals["router"] = ge.grad_logits.to(logits.dtype) * coefficient

        output.register_hook(capture)
        outputs.register_hook(lambda _: signals["expert"])
        weights.register_hook(lambda grad: grad * (1 - coefficient))
        logits.register_hook(lambda grad: grad + signals["router"])

    model._grad_em_diagnostics = install
    return run(model, value, g)


def check_recursion(dtype, normalize, coefficient, device="cpu"):
    _, _, bp, template = make_models(normalize=normalize, device=device)
    value = torch.randn(2, 5, 16, dtype=dtype, device=device)
    g = torch.randn_like(value)
    results = {}
    for alpha in (0., 0.5, 1.):
        first, second = copy.deepcopy(template), copy.deepcopy(template)
        first.grad_em_lambda = second.grad_em_lambda = coefficient
        first.grad_em_alpha = second.grad_em_alpha = alpha
        x = value.clone().requires_grad_()
        middle = first(x)
        middle.retain_grad()
        second(middle.tanh()).backward(g)
        results[alpha] = (x.grad, middle.grad)
        with torch.no_grad():
            middle_value = bp(value)
        middle_leaf = middle_value.detach().requires_grad_()
        activated = middle_leaf.tanh()
        top_boundary = signal_reference(template, activated.detach(), g, alpha * coefficient)
        incoming, = torch.autograd.grad(activated, middle_leaf, top_boundary[1])
        bottom_boundary = signal_reference(template, value, incoming, alpha * coefficient)
        compare_pair(x.grad, bottom_boundary[1], dtype)
        compare_pair(middle.grad, incoming, dtype)
        for model, inputs, downstream in ((first, value, incoming), (second, activated.detach(), g)):
            expected = signal_reference(template, inputs, downstream, coefficient)
            for name, gradient in canonical_grads(model).items():
                compare_pair(gradient, expected[2][name], dtype)
    x = value.clone().requires_grad_()
    first_bp, second_bp = copy.deepcopy(bp), copy.deepcopy(bp)
    middle = first_bp(x)
    middle.retain_grad()
    second_bp(middle.tanh()).backward(g)
    for index, ordinary in enumerate((x.grad, middle.grad)):
        torch.testing.assert_close(results[0.][index], ordinary, atol=0, rtol=0)
        assert not torch.allclose(results[0.5][index].float(), results[0.][index].float())
        assert not torch.allclose(results[0.5][index].float(), results[1.][index].float())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("coefficient", [0.5, 1.])
def test_cpu_recursion(cpu_backend, dtype, normalize, coefficient):
    check_recursion(dtype, normalize, coefficient)


@CUDA
@pytest.mark.parametrize("implementation", ["torch", "extension"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("coefficient", [0.5, 1.])
def test_cuda_recursion(monkeypatch, implementation, normalize, coefficient):
    require_cuda_backend(monkeypatch, implementation, torch.bfloat16)
    check_recursion(torch.bfloat16, normalize, coefficient, "cuda")


def check_plane(dtype, e, k, normalize, device="cpu", calls=None, need_x=True):
    _, local, bp, ge = make_models(e=e, k=k, normalize=normalize, device=device)
    value = torch.randn(2, 5, 16, dtype=dtype, device=device)
    g = torch.randn_like(value)
    bp_result = run(bp, value, g, need_x)
    ge_result = run(ge, value, g, need_x)
    local_result = run(local, value, g, need_x)
    legacy_half = copy.deepcopy(ge)
    legacy_half.zero_grad(set_to_none=True)
    legacy_half.grad_em_lambda = 0.5
    half_result = run(legacy_half, value, g, need_x)
    for coefficient, alpha in CASES:
        model = copy.deepcopy(ge)
        model.zero_grad(set_to_none=True)
        model.grad_em_lambda, model.grad_em_alpha = coefficient, alpha
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
        if calls is not None:
            calls.clear()
        actual = run(model, value, g, need_x)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        if cuda_rng is not None:
            torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, atol=0, rtol=0)
        if calls is not None:
            assert calls.count("forward") == 2
            assert calls.count("dw") == 2
            if need_x:
                assert calls.count("dx") == 2
        torch.testing.assert_close(actual[0], bp_result[0], atol=0, rtol=0)
        endpoint = (bp_result if coefficient == 0 else
                    local_result if (coefficient, alpha) == (1., 0.) else
                    ge_result if (coefficient, alpha) == (1., 1.) else
                    half_result if (coefficient, alpha) == (0.5, 1.) else None)
        for name, gradient in actual[2].items():
            assert torch.isfinite(gradient).all()
            if endpoint is not None:
                torch.testing.assert_close(gradient, endpoint[2][name], atol=0, rtol=0)
            else:
                compare_pair(gradient, bp_result[2][name] + coefficient * (
                    ge_result[2][name] - bp_result[2][name]), dtype)
        if need_x:
            if endpoint is not None:
                torch.testing.assert_close(actual[1], endpoint[1], atol=0, rtol=0)
            if alpha == 0 or coefficient == 0:
                torch.testing.assert_close(actual[1], bp_result[1], atol=0, rtol=0)
            else:
                compare_pair(actual[1], bp_result[1] + alpha * coefficient * (
                    ge_result[1] - bp_result[1]), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("e,k", [(8, 2), (64, 8)])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("need_x", [True, False])
def test_cpu_coefficient_plane(cpu_backend, dtype, e, k, normalize, need_x):
    check_plane(dtype, e, k, normalize, calls=cpu_backend, need_x=need_x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("normalize", [True, False])
def test_loop_explicit_signal_reference(cpu_backend, dtype, normalize):
    from modded_nanogpt_moe.grad_em import grad_em_reference
    loop, _, _, _ = make_models(normalize=normalize)
    x, g = torch.randn(2, 5, 16, dtype=dtype), torch.randn(2, 5, 16, dtype=dtype)
    for coefficient, alpha in CASES:
        model = copy.deepcopy(loop)
        model.grad_em_lambda, model.grad_em_alpha = coefficient, alpha
        actual = run(model, x, g)
        ref = copy.deepcopy(loop)
        leaf = x.clone().requires_grad_()
        out, logits, indices, selected = ref._forward_loop(leaf, return_local_components=True)
        ge = grad_em_reference(logits, indices, selected, g.flatten(0, 1), ref.grad_em_eta)
        bp_router, = torch.autograd.grad(out, logits, g, retain_graph=True)
        p = logits.float().softmax(-1).gather(1, indices)
        if normalize:
            p = p / p.sum(-1, keepdim=True)
        p = p.to(dtype).float()

        def vjp(strength, targets, retain_graph):
            if strength == 0:
                return torch.autograd.grad(out, targets, g, retain_graph=retain_graph, allow_unused=True)
            signal = ge.q if strength == 1 else p + strength * (ge.q - p)
            expert = (signal[..., None] * g.flatten(0, 1).float()[:, None]).to(dtype)
            router = ge.grad_logits if strength == 1 else bp_router.float() + strength * (ge.grad_logits - bp_router.float())
            return torch.autograd.grad((selected, logits), targets, (expert, router.to(dtype)),
                                       retain_graph=retain_graph, allow_unused=True)

        expected_dx, = vjp(alpha * coefficient, (leaf,), True)
        gradients = vjp(coefficient, tuple(ref.parameters()), False)
        for parameter, gradient in zip(ref.parameters(), gradients):
            parameter.grad = gradient
        torch.testing.assert_close(actual[0], out, atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected_dx, atol=0, rtol=0)
        for name, gradient in canonical_grads(ref).items():
            torch.testing.assert_close(actual[2][name], gradient, atol=0, rtol=0)


@pytest.mark.parametrize("freeze", ["router", "fc1", "fc2", "all"])
def test_partial_boundary_with_frozen_parameters(cpu_backend, freeze):
    _, _, _, model = make_models()
    for name, parameter in model.named_parameters():
        if (freeze == "all" or (freeze == "router" and name.startswith("router"))
                or (freeze == "fc1" and name.startswith("fc_"))
                or (freeze == "fc2" and name.startswith("proj_"))):
            parameter.requires_grad_(False)
    x = torch.randn(2, 5, 16, requires_grad=True)
    g = torch.randn_like(x)
    endpoints = []
    for backward in ("standard", "grad_em"):
        endpoint = copy.deepcopy(model)
        endpoint.moe_backward = backward
        endpoints.append(torch.autograd.grad(endpoint(x), x, g)[0])
    model.grad_em_lambda = model.grad_em_alpha = 0.5
    actual, = torch.autograd.grad(model(x), x, g)
    compare_pair(actual, endpoints[0] + 0.25 * (endpoints[1] - endpoints[0]), torch.float32)


@CUDA
@pytest.mark.parametrize("implementation", ["torch", "extension"])
@pytest.mark.parametrize("e,k", [(8, 2), (64, 8)])
@pytest.mark.parametrize("normalize", [True, False])
def test_cuda_coefficient_plane(monkeypatch, implementation, e, k, normalize):
    require_cuda_backend(monkeypatch, implementation, torch.bfloat16)
    check_plane(torch.bfloat16, e, k, normalize, device="cuda")


@CUDA
@pytest.mark.parametrize("implementation", ["torch", "extension"])
def test_cuda_gemm_counts(monkeypatch, implementation):
    from tools.benchmark_grad_em import count_grouped_calls
    require_cuda_backend(monkeypatch, implementation, torch.bfloat16)
    _, _, _, model = make_models(device="cuda")
    x = torch.randn(2, 5, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(x)
    expected = {f"{phase}.{kind}": 1 for phase in ("fc1", "fc2")
                for kind in ("forward", "dx", "dw")}
    for coefficient, alpha in CASES:
        model.grad_em_lambda, model.grad_em_alpha = coefficient, alpha
        assert count_grouped_calls(model, x, g) == expected


@pytest.mark.parametrize("mode,coefficient,expected", [("global", 1., 1.),
                                                      ("global", 0.5, 1.),
                                                      ("local_bp", 1., 0.)])
def test_legacy_alpha_defaults_and_checkpoint_compatibility(mode, coefficient, expected):
    config = load_experiment_config()
    config["model"].update(grad_em_mode=mode, grad_em_lambda=coefficient)
    legacy = copy.deepcopy(config)
    del legacy["model"]["grad_em_alpha"]
    current = copy.deepcopy(config)
    current["model"]["grad_em_alpha"] = expected
    checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION, "resolved_config": legacy}
    validate_checkpoint_config(checkpoint, config)
    validate_checkpoint_config(checkpoint, current)
    current["model"]["grad_em_alpha"] = 0.5
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_config(checkpoint, current)
    layer = MoE(16, 8, 2, grad_em_mode=mode, grad_em_lambda=coefficient)
    assert layer.grad_em_alpha == expected
    assert layer.grad_em_lambda == coefficient


@pytest.mark.parametrize("mode", ["global", "local_bp"])
def test_explicit_alpha_config_and_model_wiring(tmp_path, cpu_backend, mode):
    from modded_nanogpt_moe.model import GPT
    path = tmp_path / "config.toml"
    path.write_text(f'''run_name = "test-alpha"
[model]
mlp_type = "moe"
moe_backend = "grouped_gemm"
moe_parameter_layout = "packed"
num_layers = 1
model_dim = 128
vocab_size = 128
num_experts = 8
top_k = 2
moe_backward = "grad_em"
grad_em_mode = "{mode}"
grad_em_lambda = 0.5
grad_em_alpha = 0.5
''')
    config = load_experiment_config(path)
    model = GPT(**config["model"])
    assert model.blocks[0].mlp.grad_em_alpha == 0.5
    assert model.blocks[0].mlp.grad_em_lambda == 0.5


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), float("nan"), True, "0.5", 10**400])
def test_invalid_alpha(value):
    config = load_experiment_config()
    config["model"]["grad_em_alpha"] = value
    with pytest.raises(ValueError, match="grad_em_alpha"):
        validate_experiment_config(config)
    with pytest.raises(ValueError, match="grad_em_alpha"):
        MoE(16, 8, 2, grad_em_alpha=value)


def test_hidden_interpolation_zero_and_subnormal_weights():
    from modded_nanogpt_moe._local_bp import _activation_backward
    p = torch.tensor([[0., 0.25, 1e-40]])
    q = torch.full_like(p, 0.5)
    hidden = torch.tensor([[2.], [0.5], [2e-40]])
    pre = torch.ones_like(hidden)
    order = torch.arange(3)
    bp, ge = _activation_backward(hidden, pre, p, q, order, True)
    boundary, parameter = _activation_backward(hidden, pre, p, q, order, True, 0.5, 0.25)
    assert torch.isfinite(parameter).all() and torch.isfinite(boundary).all()
    torch.testing.assert_close(parameter, bp + 0.5 * (ge - bp), atol=0, rtol=0)
    torch.testing.assert_close(boundary, bp + 0.25 * (ge - bp), atol=0, rtol=0)
