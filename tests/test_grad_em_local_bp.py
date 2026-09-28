import copy
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modded_nanogpt_moe.grad_em import grad_em_reference
from modded_nanogpt_moe.model import MoE


@pytest.fixture
def cpu_gmm(monkeypatch):
    def gmm(x, weights, counts, trans_b=False):
        assert trans_b is False
        return torch.cat([
            segment @ weight for segment, weight in
            zip(x.split(counts.tolist()), weights)])

    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "extension")
    monkeypatch.setitem(
        sys.modules, "grouped_gemm", SimpleNamespace(ops=SimpleNamespace(gmm=gmm)))


def make_moe(mode, state=None, *, eta=0.4):
    kwargs = dict(dim=4, num_experts=3, top_k=2, hidden_dim=5,
                  moe_parameter_layout="modulelist")
    if mode == "standard":
        model = MoE(**kwargs, moe_backend="loop")
    elif mode == "global":
        model = MoE(**kwargs, moe_backend="grouped_gemm",
                    moe_backward="grad_em", grad_em_mode="global",
                    grad_em_eta=eta)
    elif mode == "local_bp":
        model = MoE(**kwargs, moe_backend="loop", moe_backward="grad_em",
                    grad_em_mode="local_bp", grad_em_eta=eta)
    else:
        raise AssertionError(mode)
    if state is not None:
        model.load_state_dict(state)
    return model.double()


def gradients(model, x, grad_output):
    output = model(x)
    values = torch.autograd.grad(
        output, (x, *model.parameters()), grad_output, allow_unused=True)
    return output, values[0], dict(zip(dict(model.named_parameters()), values[1:]))


def explicit_local_reference(model, x, grad_output, eta):
    output, logits, indices, selected = model._forward_loop(
        x, return_local_components=True)
    result = grad_em_reference(
        logits, indices, selected, grad_output.flatten(0, 1), eta)
    dx, = torch.autograd.grad(output, x, grad_output, retain_graph=True)
    parameters = tuple(model.parameters())
    parameter_grads = torch.autograd.grad(
        (selected, logits), parameters,
        (result.grad_expert.to(selected.dtype), result.grad_logits.to(logits.dtype)),
        allow_unused=True)
    parameter_grads = {
        name: torch.zeros_like(parameter) if gradient is None else gradient
        for (name, parameter), gradient in zip(model.named_parameters(), parameter_grads)
    }
    return output, dx, parameter_grads, result


def test_forward_input_and_local_parameter_gradient_contract(cpu_gmm):
    torch.manual_seed(43)
    standard = make_moe("standard")
    state = copy.deepcopy(standard.state_dict())
    global_grad_em = make_moe("global", state)
    local = make_moe("local_bp", state)
    reference = make_moe("standard", state)
    x_value = torch.randn(2, 3, 4, dtype=torch.float64)
    grad_output = torch.randn(2, 3, 4, dtype=torch.float64)

    results = {}
    for name, model in (("standard", standard), ("global", global_grad_em),
                        ("local", local)):
        x = x_value.clone().requires_grad_()
        results[name] = gradients(model, x, grad_output)
    x_reference = x_value.clone().requires_grad_()
    expected = explicit_local_reference(reference, x_reference, grad_output, 0.4)

    torch.testing.assert_close(results["global"][0], results["standard"][0],
                               rtol=0, atol=0)
    torch.testing.assert_close(results["local"][0], results["standard"][0],
                               rtol=0, atol=0)
    assert results["global"][0].square().sum() == results["standard"][0].square().sum()
    assert results["local"][0].square().sum() == results["standard"][0].square().sum()
    torch.testing.assert_close(results["local"][1], results["standard"][1],
                               rtol=0, atol=0)
    torch.testing.assert_close(results["local"][1], expected[1], rtol=0, atol=0)
    assert not torch.allclose(results["global"][1], results["standard"][1])

    local_grads = results["local"][2]
    standard_grads = results["standard"][2]
    for name, expected_gradient in expected[2].items():
        torch.testing.assert_close(
            local_grads[name], expected_gradient, rtol=1e-6, atol=1e-7,
            msg=name)
    assert any(not torch.allclose(local_grads[name], standard_grads[name])
               for name in local_grads if name.startswith("experts."))
    assert any(not torch.allclose(local_grads[name], standard_grads[name])
               for name in local_grads if name.startswith("router."))
    assert not torch.allclose(expected[3].q, expected[3].a)


def test_local_grad_em_converges_to_standard_for_small_eta(cpu_gmm):
    torch.manual_seed(9)
    standard = make_moe("standard")
    local = make_moe("local_bp", standard.state_dict(), eta=1e-3)
    x_value = torch.randn(2, 4, 4, dtype=torch.float64)
    grad_output = torch.randn_like(x_value)
    _, _, standard_grads = gradients(
        standard, x_value.clone().requires_grad_(), grad_output)
    _, _, local_grads = gradients(
        local, x_value.clone().requires_grad_(), grad_output)
    for name in standard_grads:
        torch.testing.assert_close(
            local_grads[name], standard_grads[name], rtol=2e-2, atol=3e-3,
            msg=name)


class TwoMoE(nn.Module):
    def __init__(self, mode, states=None, eta=0.4):
        super().__init__()
        self.first = make_moe(mode, None if states is None else states[0], eta=eta)
        self.second = make_moe(mode, None if states is None else states[1], eta=eta)

    def forward(self, x, capture):
        first_output = self.first(x)
        first_output.retain_grad()
        capture.append(first_output)
        return self.second(torch.tanh(first_output))


def test_two_moe_local_mode_anchors_each_layer_to_standard_bp(cpu_gmm):
    torch.manual_seed(71)
    standard = TwoMoE("standard")
    states = (copy.deepcopy(standard.first.state_dict()),
              copy.deepcopy(standard.second.state_dict()))
    local = TwoMoE("local_bp", states)
    global_grad_em = TwoMoE("global", states)
    x_value = torch.randn(2, 3, 4, dtype=torch.float64)
    grad_output = torch.randn_like(x_value)

    captures = {}
    parameter_grads = {}
    for name, model in (("standard", standard), ("local", local),
                        ("global", global_grad_em)):
        captured = []
        output = model(x_value.clone().requires_grad_(), captured)
        output.backward(grad_output)
        captures[name] = captured[0].grad
        parameter_grads[name] = {
            parameter_name: parameter.grad.clone()
            for parameter_name, parameter in model.named_parameters()}

    torch.testing.assert_close(captures["local"], captures["standard"],
                               rtol=0, atol=0)
    assert not torch.allclose(captures["global"], captures["standard"])
    assert any(not torch.allclose(parameter_grads["local"][name],
                                  parameter_grads["standard"][name])
               for name in parameter_grads["local"]
               if name.startswith("second.experts."))
    assert any(not torch.allclose(parameter_grads["local"][name],
                                  parameter_grads["standard"][name])
               for name in parameter_grads["local"]
               if name.startswith("second.router."))

    reference_first = make_moe("standard", states[0])
    reference_input = x_value.clone().requires_grad_()
    _, _, expected_grads, _ = explicit_local_reference(
        reference_first, reference_input, captures["standard"], 0.4)
    for name, expected in expected_grads.items():
        torch.testing.assert_close(
            parameter_grads["local"][f"first.{name}"], expected,
            rtol=1e-6, atol=1e-7, msg=name)


@pytest.mark.parametrize("layout", ["modulelist", "packed"])
def test_local_mode_accepts_grouped_layouts(cpu_gmm, layout):
    MoE(4, 3, 2, hidden_dim=5, moe_backend="grouped_gemm",
        moe_parameter_layout=layout, moe_backward="grad_em", grad_em_mode="local_bp")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_local_input_gradient_matches_standard_on_mps():
    torch.manual_seed(123)
    standard = make_moe("standard").float().to("mps")
    local = make_moe("local_bp", standard.state_dict()).float().to("mps")
    x_value = torch.randn(2, 3, 4, device="mps")
    grad_output = torch.randn_like(x_value)
    standard_x = x_value.clone().requires_grad_()
    local_x = x_value.clone().requires_grad_()
    standard_output = standard(standard_x)
    local_output = local(local_x)
    standard_dx, = torch.autograd.grad(standard_output, standard_x, grad_output)
    local_dx, = torch.autograd.grad(local_output, local_x, grad_output)
    torch.testing.assert_close(local_output, standard_output, rtol=0, atol=0)
    torch.testing.assert_close(local_dx, standard_dx, rtol=0, atol=0)
