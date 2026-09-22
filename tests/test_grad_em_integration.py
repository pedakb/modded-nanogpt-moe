"""Stage-2A CPU graph tests; the replacement rule is not forward gradcheck."""
import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from modded_nanogpt_moe import model as model_module
from modded_nanogpt_moe.grad_em import GradEMCombine, grad_em_reference, require_grad_em_device
from modded_nanogpt_moe.model import GPT, MoE, combine_expert_outputs


@pytest.fixture
def cpu_gmm(monkeypatch):
    # Test double only: production expert_gmm dispatch and graph remain intact.
    def gmm(x, weights, counts, trans_b=False):
        assert trans_b is False
        return torch.cat([segment @ weight for segment, weight in
                          zip(x.split(counts.tolist()), weights)])
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "extension")
    monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(ops=SimpleNamespace(gmm=gmm)))


@pytest.mark.parametrize("k", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_boundary_exact_forward_and_replacement_gradients(k, dtype):
    torch.manual_seed(31)
    z = torch.randn(5, 12, dtype=dtype, requires_grad=True)
    p = z.float().softmax(-1)
    weights, indices = p.topk(k, -1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(dtype)
    weights.retain_grad()
    order = indices.flatten().argsort(stable=True)
    sorted_outputs = torch.randn(5 * k, 7, dtype=dtype, requires_grad=True)
    g = torch.randn(5, 7, dtype=dtype)
    out = GradEMCombine.apply(sorted_outputs, z, weights, indices, order, 0.1)
    assert all(saved is original for saved, original in zip(
        out.grad_fn.saved_tensors, (sorted_outputs, z, indices, order)))
    assert len(out.grad_fn.saved_tensors) == 4
    assert torch.equal(out, combine_expert_outputs(sorted_outputs, weights, order))
    selected = torch.empty_like(sorted_outputs)
    selected[order] = sorted_outputs.detach()
    expected = grad_em_reference(z, indices, selected.view(5, k, 7), g)
    out.backward(g)
    assert torch.equal(sorted_outputs.grad, expected.grad_expert.flatten(0, 1)[order].to(dtype))
    assert torch.equal(z.grad, expected.grad_logits.to(dtype))
    assert weights.grad is None  # no ordinary softmax/top-k backward
    assert expected.q.grad_fn is None and not expected.q.requires_grad


@pytest.mark.parametrize("layout", ["modulelist", "packed"])
@pytest.mark.parametrize("experts,k", [(8, 1), (8, 2), (64, 8)])
@pytest.mark.parametrize("variant", ["normal", "frozen_router", "frozen_experts", "eta_zero", "g_zero"])
def test_actual_moe_graph_matches_explicit_gradient_injection(cpu_gmm, monkeypatch, layout, experts, k, variant):
    torch.manual_seed(82)
    standard = MoE(4, experts, k, hidden_dim=3, moe_backend="grouped_gemm",
                   moe_parameter_layout=layout)
    with torch.no_grad():
        standard.router.bias.copy_(torch.linspace(2, -2, experts))
        standard.router.bias[-1] = -100  # guaranteed empty expert
    grad_em = copy.deepcopy(standard)
    grad_em.moe_backward = "grad_em"
    grad_em.grad_em_eta = 0 if variant == "eta_zero" else 0.1
    if variant.startswith("frozen"):
        for module in (standard, grad_em):
            for name, parameter in module.named_parameters():
                is_router = name.startswith("router.")
                if is_router == (variant == "frozen_router"):
                    parameter.requires_grad_(False)
    x_ref = torch.randn(2, 5, 4, requires_grad=True)
    x = x_ref.detach().clone().requires_grad_()
    captured = {}
    original_combine = model_module.combine_expert_outputs
    def capture_combine(out_sorted, weights, order):
        captured["sorted"] = out_sorted
        captured["weights"] = weights
        captured["order"] = order
        out_sorted.retain_grad()
        weights.retain_grad()
        return original_combine(out_sorted, weights, order)
    monkeypatch.setattr(model_module, "combine_expert_outputs", capture_combine)
    def capture_router(module, args, output):
        captured["logits"] = output
        output.retain_grad()
    standard.router.register_forward_hook(capture_router)
    grad_em.router.register_forward_hook(capture_router)
    out_ref = standard(x_ref)
    ref = captured.copy()
    out = grad_em(x)
    actual = captured.copy()
    assert torch.equal(out, out_ref)
    assert list(grad_em.state_dict()) == list(standard.state_dict())
    indices = ref["logits"].float().softmax(-1).topk(k, -1).indices
    counts = torch.bincount(indices.flatten(), minlength=experts)
    assert counts[-1] == 0 and counts.max() > counts.min()
    selected = torch.empty_like(ref["sorted"])
    selected[ref["order"]] = ref["sorted"].detach()
    g = torch.zeros_like(out) if variant == "g_zero" else torch.randn_like(out)
    expected = grad_em_reference(ref["logits"], indices, selected.view(10, k, 4),
                                 g.flatten(0, 1), grad_em.grad_em_eta)
    grad_sorted = expected.grad_expert.flatten(0, 1)[ref["order"]]
    # Separate VJPs demonstrate that BOTH paths to x survive, even when their
    # parameters are frozen. Neither branch is detached by the custom boundary.
    expert_dx = torch.autograd.grad(ref["sorted"], x_ref, grad_sorted, retain_graph=True)[0]
    router_dx = torch.autograd.grad(ref["logits"], x_ref, expected.grad_logits, retain_graph=True)[0]
    assert torch.count_nonzero(router_dx) > 0
    if variant != "g_zero":
        assert torch.count_nonzero(expert_dx) > 0
    torch.autograd.backward((ref["sorted"], ref["logits"]), (grad_sorted, expected.grad_logits))
    out.backward(g)
    assert torch.equal(actual["sorted"].grad, grad_sorted)
    assert torch.equal(actual["logits"].grad, expected.grad_logits)
    assert actual["weights"].grad is None
    torch.testing.assert_close(x.grad, expert_dx + router_dx, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-6, atol=1e-7)
    for (name, p), (_, reference) in zip(grad_em.named_parameters(), standard.named_parameters()):
        if not p.requires_grad:
            assert p.grad is None and reference.grad is None
        else:
            torch.testing.assert_close(p.grad, reference.grad, rtol=0, atol=0, msg=name)
    if variant != "frozen_experts":
        empty = (list(grad_em.experts[-1].parameters()) if layout == "modulelist" else
                 [grad_em.fc_weight, grad_em.fc_bias, grad_em.proj_weight, grad_em.proj_bias])
        for parameter in empty:
            value = parameter.grad if layout == "modulelist" else parameter.grad[-1]
            assert value is not None and torch.count_nonzero(value) == 0


def test_default_standard_never_calls_custom_boundary(cpu_gmm, monkeypatch):
    torch.manual_seed(4)
    default = MoE(4, 8, 2, hidden_dim=3, moe_backend="grouped_gemm")
    torch.manual_seed(4)
    explicit = MoE(4, 8, 2, hidden_dim=3, moe_backend="grouped_gemm", moe_backward="standard")
    def forbidden(*args):
        pytest.fail("standard mode called Grad-EM")
    monkeypatch.setattr(GradEMCombine, "apply", forbidden)
    x = torch.randn(2, 3, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = default(x), explicit(y)
    assert torch.equal(a, b)
    a.sum().backward()
    b.sum().backward()
    assert torch.equal(x.grad, y.grad)
    for p, q in zip(default.parameters(), explicit.parameters()):
        assert torch.equal(p.grad, q.grad)


def test_config_reaches_each_moe_block(cpu_gmm):
    model = GPT(37, 2, 128, mlp_type="moe", num_experts=8, top_k=2,
                moe_backend="grouped_gemm", moe_backward="grad_em", grad_em_eta=0.7)
    assert all(block.mlp.moe_backward == "grad_em" and block.mlp.grad_em_eta == 0.7
               for block in model.blocks)


def test_supported_devices_without_allocating_gpu_tensors():
    require_grad_em_device(torch.device("cpu"))
    require_grad_em_device(torch.device("cuda"))


def test_moe_rejects_unsupported_device_before_router_or_expert_execution(cpu_gmm):
    moe = MoE(4, 8, 2, moe_backend="grouped_gemm", moe_backward="grad_em")
    # Only device is inspected: no actual GPU allocation or execution.
    with pytest.raises(NotImplementedError, match="only CPU and CUDA"):
        moe(SimpleNamespace(device=torch.device("mps")))


def test_loop_grad_em_rejected_explicitly():
    with pytest.raises(NotImplementedError, match="grouped_gemm combine boundary"):
        MoE(4, 8, 2, moe_backward="grad_em")
