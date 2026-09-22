"""CUDA acceptance tests exercising the public Grad-EM path without bypasses."""
import copy
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from modded_nanogpt_moe import grad_em as em
from modded_nanogpt_moe import model as model_module

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")


@pytest.fixture
def candidate():
    pytest.importorskip("triton")
    from modded_nanogpt_moe import _grad_em_cuda
    return _grad_em_cuda


def tolerance(dtype):
    # One low-precision rounding bin plus small FP32 reduction/softmax error.
    return {torch.bfloat16: dict(rtol=8e-3, atol=2e-5),
            torch.float16: dict(rtol=1e-3, atol=2e-6),
            torch.float32: dict(rtol=2e-5, atol=2e-6)}[dtype]


@CUDA
@pytest.mark.parametrize("e,k", [(8, 1), (8, 2), (64, 8), (7, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", ["random", "imbalanced", "eta_zero", "g_zero"])
def test_cuda_oracle_and_direct_autograd(candidate, e, k, dtype, case):
    torch.manual_seed(41)
    n, d = 19, 768
    z = torch.randn(n, e, device="cuda", dtype=dtype)
    if case == "imbalanced":
        z[:, k:] = -100  # all assignments go to K experts; remaining experts empty
    z.requires_grad_()
    w, ids = z.float().softmax(-1).topk(k, -1)
    w = (w / w.sum(-1, keepdim=True)).to(dtype)
    w.retain_grad()
    order = ids.flatten().argsort(stable=True)
    x = (torch.randn(n*k, d, device="cuda", dtype=dtype) * 0.2).requires_grad_()
    g = torch.randn(n, d, device="cuda", dtype=dtype) * 0.1
    if case == "g_zero":
        g.zero_()
    eta = 0 if case == "eta_zero" else 0.1
    selected = torch.empty_like(x)
    selected[order] = x.detach()
    ref = em.grad_em_reference(z, ids, selected.view(n, k, d), g, eta)
    output, rows = candidate.cuda_forward(x, z, w, ids, order)
    gx, gz, q, v = candidate.cuda_backward(x, z, ids, rows, g, eta, save_v=True)
    torch.testing.assert_close(v, ref.v, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(q, ref.q, rtol=2e-5, atol=2e-6)
    assert q.dtype == v.dtype == torch.float32 and q.grad_fn is None
    torch.testing.assert_close(gx, ref.grad_expert.flatten(0, 1)[order].to(dtype), **tolerance(dtype))
    torch.testing.assert_close(gz, ref.grad_logits.to(dtype), **tolerance(dtype))
    standard = model_module.combine_expert_outputs(x, w, order)
    torch.testing.assert_close(output, standard, rtol=0, atol=0)
    actual = em.GradEMCombine.apply(x, z, w, ids, order, eta)
    torch.testing.assert_close(actual, standard, rtol=0, atol=0)
    actual.backward(g)
    torch.testing.assert_close(x.grad, gx, rtol=0, atol=0)
    torch.testing.assert_close(z.grad, gz, rtol=0, atol=0)
    assert w.grad is None
    if case == "g_zero":
        assert torch.count_nonzero(gx) == 0 and torch.count_nonzero(gz) > 0


@CUDA
@pytest.mark.parametrize("need_x,need_z", [(True, False), (False, True), (True, True)])
def test_cuda_strides_feature_tail_and_empty(candidate, need_x, need_z):
    for n in (0, 5):
        k, e, d = 3, 8, 37
        z = torch.randn(e, n, device="cuda").T.requires_grad_(need_z)
        w, ids = z.softmax(-1).topk(k, -1)
        w = w / w.sum(-1, keepdim=True)
        order = ids.flatten().argsort(stable=True)
        x = torch.randn(d, n*k, device="cuda").T.requires_grad_(need_x)
        selected = torch.empty_like(x)
        selected[order] = x.detach()
        g = torch.ones(1, 1, device="cuda").expand(n, d)
        ref = em.grad_em_reference(z, ids, selected.reshape(n, k, d), g)
        em.GradEMCombine.apply(x, z, w, ids, order, 0.1).backward(g)
        if need_x:
            torch.testing.assert_close(x.grad, ref.grad_expert.flatten(0, 1)[order], **tolerance(x.dtype))
        if need_z:
            torch.testing.assert_close(z.grad, ref.grad_logits, **tolerance(z.dtype))


@CUDA
@pytest.mark.parametrize("layout", ["modulelist", "packed"])
@pytest.mark.parametrize("e,k", [(8, 2), (64, 8)])
def test_cuda_full_expert_and_router_graph(candidate, monkeypatch, layout, e, k):
    # Use the actual GH200/native or extension backend selected by the caller.
    if model_module.implementation_from_environment() == "extension":
        pytest.importorskip("grouped_gemm")
    torch.manual_seed(13)
    base = model_module.MoE(768, e, k, hidden_dim=1536 if e == 8 else 384, moe_backend="grouped_gemm",
                            moe_parameter_layout=layout).cuda()
    with torch.no_grad():
        base.router.bias[k:] = -100
    actual = copy.deepcopy(base)
    actual.moe_backward = "grad_em"
    x0 = torch.randn(2, 16, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_()
    captured = {}
    def router_hook(module, args, output):
        captured["z"] = output
    base.router.register_forward_hook(router_hook)
    combine = model_module.combine_expert_outputs
    def capture(x, w, order):
        captured.update(x=x, order=order)
        return combine(x, w, order)
    monkeypatch.setattr(model_module, "combine_expert_outputs", capture)
    y0 = base(x0)
    y1 = actual(x1)
    torch.testing.assert_close(y1, y0, rtol=0, atol=0)
    g = torch.randn_like(y0) * 0.1
    z, sorted_x, order = captured["z"], captured["x"], captured["order"]
    ids = z.float().softmax(-1).topk(k, -1).indices
    selected = torch.empty_like(sorted_x)
    selected[order] = sorted_x.detach()
    ref = em.grad_em_reference(z, ids, selected.view(32, k, 768), g.view(32, 768))
    gx = ref.grad_expert.flatten(0, 1)[order].to(sorted_x.dtype)
    gz = ref.grad_logits.to(z.dtype)
    expert_dx = torch.autograd.grad(sorted_x, x0, gx, retain_graph=True)[0]
    router_dx = torch.autograd.grad(z, x0, gz, retain_graph=True)[0]
    torch.autograd.backward((sorted_x, z), (gx, gz))
    y1.backward(g)
    # Two BF16 GEMM-backward paths amplify one-bin boundary rounding: same
    # 2% envelope as existing CUDA full-MoE parity tests, not a v/q tolerance.
    torch.testing.assert_close(x1.grad, expert_dx + router_dx, rtol=2e-2, atol=2e-2)
    for p, r in zip(actual.parameters(), base.parameters()):
        torch.testing.assert_close(p.grad, r.grad, rtol=2e-2, atol=2e-2)


def test_candidate_launches_and_compact_scratch(monkeypatch):
    """CPU plumbing only: does not compile or establish kernel correctness."""
    launches, allocations = [], []
    class Kernel:
        def __init__(self, function):
            self.name = function if isinstance(function, str) else function.__name__
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.name, grid))
            return launch
    triton, tl = ModuleType("triton"), ModuleType("triton.language")
    triton.jit, triton.language, tl.constexpr = Kernel, tl, object()
    triton.cdiv = lambda a, b: (a+b-1)//b
    triton.next_power_of_2 = lambda a: 1 << (a-1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)
    monkeypatch.setitem(sys.modules, "modded_nanogpt_moe._combine", SimpleNamespace(
        _combine_assignment_rows=Kernel("_combine_assignment_rows"), _combine_forward=Kernel("_combine_forward")))
    path = Path(__file__).resolve().parents[1] / "modded_nanogpt_moe/_grad_em_cuda.py"
    spec = importlib.util.spec_from_file_location("modded_nanogpt_moe.candidate_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    allocate = torch.empty
    def fake(shape, dtype=torch.float32):
        strides = allocate(shape).stride()
        return SimpleNamespace(shape=torch.Size(shape), dtype=dtype, device=torch.device("cuda"),
                               is_cuda=True, ndim=len(shape), stride=lambda *axis: strides[axis[0]] if axis else strides)
    def record(shape, **kwargs):
        allocations.append((tuple(shape), kwargs["dtype"]))
        return allocate(shape, dtype=kwargs["dtype"])
    monkeypatch.setattr(torch, "empty", record)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    n, k, d, e = 17, 8, 768, 64
    x, z, w = fake((n*k, d), torch.bfloat16), fake((n, e)), fake((n, k), torch.bfloat16)
    ids, order = fake((n, k), torch.int64), fake((n*k,), torch.int64)
    _, rows = module.cuda_forward(x, z, w, ids, order)
    gx, gz, q, v = module.cuda_backward(x, z, ids, rows, fake((n, d)), 0.1)
    assert allocations == [((n*k,), torch.int64), ((n, d), torch.bfloat16),
                           ((n*k, d), torch.bfloat16), ((n, e), torch.float32), ((n, k), torch.float32)]
    assert v is None and q.shape == (n, k)
    assert launches == [("_combine_assignment_rows", (1,)), ("_combine_forward", (n, 6)),
                        ("_grad_em_expert_backward", (n,)), ("_grad_em_router_backward", (n,))]


@CUDA
def test_cuda_kernel_launch_count(candidate):
    n, k, d, e = 65, 8, 768, 64
    z = torch.randn(n, e, device="cuda", requires_grad=True)
    w, ids = z.detach().softmax(-1).topk(k, -1)
    w = (w / w.sum(-1, keepdim=True)).bfloat16()
    order = ids.flatten().argsort(stable=True)
    x = torch.randn(n*k, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(n, d, device="cuda", dtype=x.dtype)
    em.GradEMCombine.apply(x, z, w, ids, order, 0.1).backward(g)
    x.grad = z.grad = None
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        em.GradEMCombine.apply(x, z, w, ids, order, 0.1).backward(g)
        torch.cuda.synchronize()
    kernels = [event.name for event in profile.events()
               if event.device_type == torch.autograd.DeviceType.CUDA]
    for name in ("_combine_assignment_rows", "_combine_forward",
                 "_grad_em_expert_backward", "_grad_em_router_backward"):
        assert sum(name in event for event in kernels) == 1, kernels
    assert len(kernels) == 4, kernels
