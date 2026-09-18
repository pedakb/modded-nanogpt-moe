"""Combine-only parity; CPU checks do not establish Triton kernel correctness."""
import copy
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from modded_nanogpt_moe import model as model_module
from modded_nanogpt_moe.model import MoE, combine_expert_outputs


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


def reference_combine(x, weights, order):
    n, k = weights.shape
    unsorted = torch.empty_like(x)
    unsorted[order] = x
    return (unsorted.view(n, k, -1) * weights.unsqueeze(-1)).sum(1)


def routing_order(n, e, k, imbalanced, device):
    if imbalanced:
        selected = torch.arange(k).expand(n, k)
    else:
        selected = torch.randn(n, e).topk(k, dim=-1).indices
    return selected.flatten().argsort(stable=True).to(device)


def evaluate(implementation, x, weights, order, grad):
    x = x.detach().clone().requires_grad_()
    weights = weights.detach().clone().requires_grad_()
    out = implementation(x, weights, order)
    gx, gw = torch.autograd.grad(out, (x, weights), grad)
    return out.detach(), gx, gw


def check_rounded_sum(actual, expected, products, axis):
    # Compare both implementations to the sum of dtype-rounded products in FP64.
    exact = products.double().sum(axis)
    u = torch.finfo(torch.float32).eps / 2
    length = products.shape[axis]
    gamma = length * u / (1 - length * u)
    bound = gamma * products.double().abs().sum(axis)
    # Final output cast (including subnormal rounding), not a relaxed allclose.
    finfo = torch.finfo(actual.dtype)
    bound += finfo.eps / 2 * (exact.abs() + bound) + finfo.tiny * finfo.eps / 2
    for value in (actual, expected):
        assert torch.all((value.double() - exact).abs() <= bound)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("e,k", [(1, 1), (8, 2), (8, 4), (64, 8), (7, 3)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("imbalanced", [False, True])
def test_combine_outputs_and_both_gradients(device, e, k, dtype, imbalanced):
    torch.manual_seed(101)
    n, d = 19, 768
    order = routing_order(n, e, k, imbalanced, device)
    x = torch.randn(n * k, d, device=device, dtype=dtype)
    weights = torch.randn(n, k, device=device).softmax(-1).to(dtype)
    grad = torch.randn(n, d, device=device, dtype=dtype)
    expected = evaluate(reference_combine, x, weights, order, grad)
    actual = evaluate(combine_expert_outputs, x, weights, order, grad)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    unsorted = torch.empty_like(x)
    unsorted[order] = x
    values = unsorted.view(n, k, d)
    check_rounded_sum(actual[0], expected[0], values * weights[:, :, None], 1)
    check_rounded_sum(actual[2], expected[2], grad[:, None, :] * values, 2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_combine_exact_binary_sums_and_feature_tail(device, dtype):
    torch.manual_seed(102)
    n, k, d = 17, 8, 37
    order = routing_order(n, 64, k, False, device)
    x = torch.randint(-8, 9, (n * k, d), device=device).to(dtype) / 8
    weights = torch.randint(0, 9, (n, k), device=device).to(dtype) / 16
    grad = torch.randint(-8, 9, (n, d), device=device).to(dtype) / 8
    expected = evaluate(reference_combine, x, weights, order, grad)
    actual = evaluate(combine_expert_outputs, x, weights, order, grad)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_combine_rounds_products_before_both_reductions(device, dtype):
    u = torch.finfo(dtype).eps
    order = torch.tensor([1, 0], device=device)
    # Rounded products cancel exactly; an incorrectly fused FP32 dot gives u**2.
    x = torch.tensor([[-(1 + 2*u)], [1 + u]], device=device, dtype=dtype)
    weights = torch.tensor([[1 + u, 1]], device=device, dtype=dtype)
    actual = combine_expert_outputs(x, weights, order)
    assert torch.count_nonzero(actual) == 0
    assert (x[1].float() * weights[0, 0].float() + x[0].float()) != 0
    x = torch.tensor([[1 + u, -(1 + 2*u)]], device=device, dtype=dtype)
    weights = torch.ones(1, 1, device=device, dtype=dtype, requires_grad=True)
    out = combine_expert_outputs(x, weights, torch.tensor([0], device=device))
    out.backward(torch.tensor([[1 + u, 1]], device=device, dtype=dtype))
    assert torch.count_nonzero(weights.grad) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("need_x,need_weights", [(True, False), (False, True), (True, True)])
def test_combine_strided_inputs_expanded_gradient(device, need_x, need_weights):
    n, k, d = 5, 4, 37
    torch.manual_seed(103)
    order = routing_order(n, 8, k, False, device)
    x = torch.randn(d, n * k, device=device).T.requires_grad_(need_x)
    weights = torch.randn(k, n, device=device).T.requires_grad_(need_weights)
    params = tuple(p for p in (x, weights) if p.requires_grad)
    expected_out = reference_combine(x, weights, order)
    expected = torch.autograd.grad(expected_out.sum(), params)
    actual_out = combine_expert_outputs(x, weights, order)
    actual = torch.autograd.grad(actual_out.sum(), params)
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-5, atol=1e-5)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_combine_empty_and_double_gradcheck(device):
    x = torch.empty(0, 37, device=device, requires_grad=True)
    w = torch.empty(0, 4, device=device, requires_grad=True)
    out = combine_expert_outputs(x, w, torch.empty(0, device=device, dtype=torch.int64))
    out.sum().backward()
    assert out.shape == (0, 37) and x.grad.shape == x.shape and w.grad.shape == w.shape
    x = torch.randn(6, 3, device=device, dtype=torch.float64, requires_grad=True)
    w = torch.randn(2, 3, device=device, dtype=torch.float64, requires_grad=True)
    order = torch.tensor([4, 0, 1, 5, 3, 2], device=device)
    assert torch.autograd.gradcheck(lambda a, b: combine_expert_outputs(a, b, order), (x, w))


def mock_triton_module(monkeypatch, launches):
    class Kernel:
        def __init__(self, fn):
            self.name = fn.__name__
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.name, grid))
                if self.name != "_combine_assignment_rows":
                    assert kwargs["enable_fp_fusion"] is False
            return launch
    triton = ModuleType("triton")
    language = ModuleType("triton.language")
    language.constexpr = object()
    triton.language, triton.jit = language, Kernel
    triton.cdiv = lambda x, y: (x + y - 1) // y
    triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = Path(__file__).resolve().parents[1] / "modded_nanogpt_moe/_combine.py"
    spec = importlib.util.spec_from_file_location("combine_launcher_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_combine_launch_plumbing_and_no_activation_scratch(monkeypatch):
    launches, allocations = [], []
    module = mock_triton_module(monkeypatch, launches)
    n, k, d = 17, 8, 768
    x, weights = torch.empty(n*k, d), torch.empty(n, k)
    order = torch.arange(n*k)
    allocate = torch.empty
    def record_empty(shape, **kwargs):
        allocations.append((tuple(shape), kwargs["dtype"]))
        return allocate(shape, **kwargs)
    class Context:
        needs_input_grad = (True, True, False)
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors
    ctx = Context()
    monkeypatch.setattr(torch, "empty", record_empty)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    out = module._FusedCombine.forward(ctx, x, weights, order)
    assert allocations == [((n*k,), torch.int64), ((n, d), x.dtype)]
    with torch.no_grad():
        gx, gw, _ = module._FusedCombine.backward(ctx, torch.ones_like(out))
    assert gx.shape == x.shape and gw.shape == weights.shape
    assert launches == [("_combine_assignment_rows", (1,)),
                         ("_combine_forward", (n, 6)),
                         ("_combine_backward", (34,))]


def test_fused_combine_higher_order_fallback(monkeypatch):
    module = mock_triton_module(monkeypatch, [])
    x = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    grad = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    order = torch.tensor([4, 0, 1, 5, 3, 2])
    rows = torch.empty_like(order)
    rows[order] = torch.arange(6)
    ctx = SimpleNamespace(saved_tensors=(x, weights, rows), needs_input_grad=(True, True, False))
    expected = torch.autograd.grad(reference_combine(x, weights, order), (x, weights),
                                   grad, create_graph=True)
    actual = module._FusedCombine.backward(ctx, grad)[:2]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    expected_second = torch.autograd.grad(sum(t.sum() for t in expected), (x, weights, grad))
    actual_second = torch.autograd.grad(sum(t.sum() for t in actual), (x, weights, grad))
    for a, b in zip(actual_second, expected_second):
        # Different second-order graphs change addition order (observed ~4e-16),
        # so allow only a small FP64-roundoff bound, not BF16/model tolerances.
        torch.testing.assert_close(a, b, rtol=32*torch.finfo(torch.float64).eps,
                                   atol=32*torch.finfo(torch.float64).eps)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", ["modulelist", "packed"])
@pytest.mark.parametrize("e,k", [(8, 2), (8, 4), (64, 8)])
@pytest.mark.parametrize("imbalanced", [False, True])
def test_moe_combine_preserves_router_and_expert_gradients(monkeypatch, device, layout, e, k, imbalanced):
    if device == "cuda":
        pytest.importorskip("grouped_gemm")
    else:
        def gmm(x, weights, counts, trans_b=False):
            return torch.cat([segment @ weight for segment, weight in
                              zip(x.split(counts.tolist()), weights)])
        monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(ops=SimpleNamespace(gmm=gmm)))
    torch.manual_seed(104)
    d = 768 if device == "cuda" else 16
    model = MoE(d, e, k, hidden_dim=d//2, moe_backend="grouped_gemm",
                 moe_parameter_layout=layout).to(device)
    if imbalanced:
        with torch.no_grad():
            model.router.weight.zero_()
            model.router.bias.copy_(-torch.arange(e, device=device) / e)
    updated = copy.deepcopy(model)
    base = torch.randn(2, 19, d, device=device, dtype=torch.bfloat16)
    gradient = torch.randn_like(base)
    results = []
    for moe, implementation in ((model, reference_combine), (updated, combine_expert_outputs)):
        monkeypatch.setattr(model_module, "combine_expert_outputs", implementation)
        x = base.clone().requires_grad_()
        out = moe(x)
        out.backward(gradient)
        results.append([out.detach(), x.grad, *(p.grad for p in moe.parameters())])
    tolerance = dict(rtol=2e-2, atol=2e-2) if device == "cuda" else dict(rtol=0, atol=0)
    for a, b in zip(results[1], results[0]):
        assert a is not None and b is not None
        torch.testing.assert_close(a, b, **tolerance)


@CUDA
def test_cuda_combine_kernel_counts():
    n, k, d = 65, 8, 768
    order = routing_order(n, 64, k, False, "cuda")
    x = torch.randn(n*k, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(n, k, device="cuda", dtype=x.dtype, requires_grad=True)
    gradient = torch.randn(n, d, device="cuda", dtype=x.dtype)
    combine_expert_outputs(x, weights, order).backward(gradient)  # Warm JIT outside capture.
    x.grad = weights.grad = None
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        combine_expert_outputs(x, weights, order).backward(gradient)
        torch.cuda.synchronize()
    kernels = [event.name for event in profile.events()
               if event.device_type == torch.autograd.DeviceType.CUDA]
    for name in ("_combine_assignment_rows", "_combine_forward", "_combine_backward"):
        assert sum(name in event for event in kernels) == 1, kernels
    assert len(kernels) == 3, kernels
