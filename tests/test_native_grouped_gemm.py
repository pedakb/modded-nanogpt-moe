"""Native SM90 grouped GEMM versus the extension, without touching bias/combine."""
import copy
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from modded_nanogpt_moe import _grouped_gemm as gemm
from modded_nanogpt_moe import model
from tools.benchmark_expert_gemm import routing_counts


@pytest.mark.parametrize("e,k", [(8, 2), (64, 8)])
@pytest.mark.parametrize("routing", ["balanced", "imbalanced"])
def test_benchmark_counts_are_valid_dropless_topk(e, k, routing):
    counts = routing_counts(e, k, 65536, routing)
    assert counts.sum() == 65536 * k
    assert counts.max() <= 65536
    if routing == "balanced":
        assert counts.tolist() == [65536 * k // e] * e
    else:
        assert counts.tolist() == [65536] * k + [0] * (e - k)


def reference_mm(a, b, *, offs):
    starts = [0] + offs.tolist()
    if b.ndim == 3:
        return torch.cat([a[start:end] @ b[e]
                          for e, (start, end) in enumerate(zip(starts, starts[1:]))])
    return torch.stack([a[:, start:end] @ b[start:end]
                        for start, end in zip(starts, starts[1:])])


@pytest.mark.parametrize("e", [8, 64])
@pytest.mark.parametrize("empty", [False, True])
def test_native_autograd_call_shapes_and_values_cpu(e, empty):
    # Exercise the actual torch API's CPU fallback and transpose/offset contract.
    # This does not test its CUDA kernel; production explicitly rejects CPU.
    counts = torch.arange(1, e + 1) % 5 if empty else torch.full((e,), 3)
    counts[0] = 37
    offsets = counts.cumsum(0, dtype=torch.int32)
    torch.manual_seed(31)
    a = torch.randn(int(counts.sum()), 16, requires_grad=True)
    b = torch.randn(e, 16, 8, requires_grad=True)
    expected = reference_mm(a, b, offs=offsets)
    actual = gemm._NativeGroupedGemm.apply(a, b, offsets, "fc1")
    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (a, b), grad)
    actual_grads = torch.autograd.grad(actual, (a, b), grad)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for x, y in zip(actual_grads, expected_grads):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert torch.count_nonzero(actual_grads[1][counts == 0]) == 0


def test_selection_default_errors_and_native_cpu_rejection(monkeypatch):
    monkeypatch.delenv("MOE_GMM_IMPLEMENTATION", raising=False)
    assert gemm.implementation_from_environment() == "extension"
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "torch")
    assert gemm.implementation_from_environment() == "torch"
    with pytest.raises(ValueError, match="SM90"):
        gemm.native_gmm(torch.zeros(8, 16), torch.zeros(2, 16, 8), None, "fc1")
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "typo")
    with pytest.raises(ValueError, match="MOE_GMM_IMPLEMENTATION"):
        gemm.implementation_from_environment()


@pytest.mark.parametrize("implementation", ["extension", "torch"])
@pytest.mark.parametrize("requires_a,requires_b", [(True, True), (True, False), (False, True)])
def test_phase_ranges_and_frozen_parameters(monkeypatch, implementation, requires_a, requires_b):
    calls, ranges = [], []
    @contextmanager
    def nvtx(enabled, name):
        if enabled:
            ranges.append(("push", name))
        try:
            yield
        finally:
            if enabled:
                ranges.append(("pop", name))
    monkeypatch.setattr(model, "nsys_range", nvtx)
    monkeypatch.setattr(model, "_moe_nsys_capture_active", True)
    def raw(a, b, counts, trans_a=False, trans_b=False):
        calls.append((trans_a, trans_b))
        return reference_mm(a.mT if trans_a else a, b.mT if trans_b else b,
                            offs=counts.cumsum(0))
    monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(backend=SimpleNamespace(gmm=raw)))
    monkeypatch.setattr(gemm.F, "grouped_mm", reference_mm)
    a = torch.randn(7, 16, requires_grad=requires_a)
    b = torch.randn(2, 16, 8, requires_grad=requires_b)
    counts = torch.tensor([3, 4])
    function = gemm._ProfiledExtensionGemm if implementation == "extension" else gemm._NativeGroupedGemm
    function.apply(a, b, counts if implementation == "extension" else counts.cumsum(0), "fc2").sum().backward()
    phases = ["forward"] + (["dx"] if requires_a else []) + (["dw"] if requires_b else [])
    assert ranges == [(action, f"grouped_gemm.fc2.{phase}") for phase in phases for action in ("push", "pop")]
    if implementation == "extension":
        assert calls == [(False, False)] + ([(False, True)] if requires_a else []) + ([(True, False)] if requires_b else [])


def test_baseline_inactive_not_wrapped(monkeypatch):
    def baseline(a, b, counts, trans_b=False):
        assert trans_b is False
        return "original"
    baseline.__module__ = "grouped_gemm.ops"
    assert gemm.expert_gmm(None, None, None, None, "extension", baseline, "fc1", False) == "original"


@pytest.mark.parametrize("e,k,h", [(8, 2, 32), (64, 8, 8)])
def test_model_integration_cpu(monkeypatch, e, k, h):
    def baseline(a, b, counts, trans_b=False):
        return reference_mm(a, b, offs=counts.cumsum(0))
    monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(ops=SimpleNamespace(gmm=baseline)))
    monkeypatch.setattr(gemm, "validate_native_inputs", lambda a, b: None)
    monkeypatch.setattr(gemm.F, "grouped_mm", reference_mm)
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "extension")
    old = model.MoE(16, e, k, hidden_dim=h, moe_backend="grouped_gemm", moe_parameter_layout="packed")
    with torch.no_grad():
        old.proj_weight.normal_(std=0.05)
        old.router.bias[-1] = -1e4
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "torch")
    new = model.MoE(16, e, k, hidden_dim=h, moe_backend="grouped_gemm", moe_parameter_layout="packed")
    new.load_state_dict(old.state_dict())
    assert list(new.state_dict()) == list(old.state_dict())
    x = torch.randn(1, 17, 16)
    _check_models(old, new, x, atol=0, rtol=0)


def _check_models(old, new, x, atol, rtol):
    x_old, x_new = x.clone().requires_grad_(), x.clone().requires_grad_()
    y_old, y_new = old(x_old), new(x_new)
    grad = torch.randn_like(y_old)
    y_old.backward(grad)
    y_new.backward(grad)
    torch.testing.assert_close(y_new, y_old, atol=atol, rtol=rtol)
    torch.testing.assert_close(x_new.grad, x_old.grad, atol=atol, rtol=rtol)
    for (old_name, old_p), (new_name, new_p) in zip(old.named_parameters(), new.named_parameters()):
        assert old_name == new_name
        torch.testing.assert_close(new_p.grad, old_p.grad, atol=atol, rtol=rtol, msg=lambda msg: f"{old_name}: {msg}")
    for name in ("fc_weight", "fc_bias", "proj_weight", "proj_bias"):
        assert torch.count_nonzero(getattr(new, name).grad[-1]) == 0


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM90")


def require_sm90():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("native candidate is restricted to SM90; extension remains available on LS6")


@CUDA
@pytest.mark.parametrize("e,h", [(8, 1536), (64, 384)])
@pytest.mark.parametrize("routing", ["balanced", "imbalanced", "empty"])
@pytest.mark.parametrize("phase", ["fc1", "fc2"])
def test_native_gemm_cuda(e, h, routing, phase):
    require_sm90()
    extension = pytest.importorskip("grouped_gemm")
    torch.manual_seed(72)
    counts = torch.full((e,), 129, dtype=torch.int64)
    if routing == "imbalanced":
        counts[1:] = 1
        counts[0] = 4093
    if routing == "empty":
        counts[::2] = 0
    d_in, d_out = (768, h) if phase == "fc1" else (h, 768)
    a = (torch.randn(int(counts.sum()), d_in, device="cuda", dtype=torch.bfloat16) / d_in**0.5).requires_grad_()
    b = (torch.randn(e, d_in, d_out, device="cuda", dtype=torch.bfloat16) / d_in**0.5).requires_grad_()
    offsets = counts.to("cuda").cumsum(0, dtype=torch.int32)
    old = extension.ops.gmm(a, b, counts, trans_b=False)
    new = gemm.native_gmm(a, b, offsets, phase)
    grad = torch.randn_like(old) / d_out**0.5
    old_grads = torch.autograd.grad(old, (a, b), grad)
    new_grads = torch.autograd.grad(new, (a, b), grad)
    # Same established BF16 full-layer tolerances; dtype checking stays on.
    for actual, expected in zip((new, *new_grads), (old, *old_grads)):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(new_grads[1][counts.to("cuda") == 0]) == 0
    # Independent FP32 dot-product reference, not merely agreement with another backend.
    ref_a, ref_b = a.detach().float().requires_grad_(), b.detach().float().requires_grad_()
    ref = reference_mm(ref_a, ref_b, offs=counts.cumsum(0))
    ref_grads = torch.autograd.grad(ref, (ref_a, ref_b), grad.float())
    for actual, expected in zip((new, *new_grads), (ref, *ref_grads)):
        torch.testing.assert_close(actual, expected.to(a.dtype), atol=2e-2, rtol=2e-2)


@CUDA
def test_native_cuda_exact_binary_products_and_empty_dw():
    require_sm90()
    # Products and short sums are exactly representable in BF16: no tolerance.
    counts = torch.tensor([0, 3, 7, 0, 17, 5, 1, 0])
    offsets = counts.cuda().cumsum(0, dtype=torch.int32)
    a = torch.full((int(counts.sum()), 16), 0.125, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.full((8, 16, 8), 0.125, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = gemm.native_gmm(a, b, offsets, "fc1")
    da, db = torch.autograd.grad(y, (a, b), torch.full_like(y, 0.125))
    torch.testing.assert_close(y, torch.full_like(y, 0.25), atol=0, rtol=0)
    torch.testing.assert_close(da, torch.full_like(da, 0.125), atol=0, rtol=0)
    expected_dw = (counts.to(device="cuda", dtype=a.dtype) / 64)[:, None, None].expand_as(db)
    torch.testing.assert_close(db, expected_dw, atol=0, rtol=0)


@CUDA
@pytest.mark.parametrize("e,k,h", [(8, 2, 1536), (64, 8, 384)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_full_moe_native_cuda(monkeypatch, e, k, h, dtype):
    require_sm90()
    pytest.importorskip("grouped_gemm")
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "extension")
    torch.manual_seed(73)
    old = model.MoE(768, e, k, hidden_dim=h, moe_backend="grouped_gemm", moe_parameter_layout="packed").to(device="cuda", dtype=dtype)
    with torch.no_grad():
        old.proj_weight.normal_(std=0.02)
        old.router.bias[-1] = -1e4
        old.router.bias[0] = 2  # Uneven routing plus a guaranteed empty expert.
    new = copy.deepcopy(old)
    new.gmm_implementation = "torch"
    _check_models(old, new, torch.randn(1, 257, 768, device="cuda", dtype=torch.bfloat16), atol=2e-2, rtol=2e-2)
