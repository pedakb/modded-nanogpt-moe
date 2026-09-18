"""Opt-in PyTorch SM90 grouped GEMM; the nv-grouped-gemm baseline stays default.

Both implementations compute only contiguous routed segments, with BF16 outputs
and gradients and FP32 accumulation. No expert padding or parameter copies.
See docs/grouped_gemm.md for pinned source analysis and validation commands.
"""

import os

import torch
import torch.nn.functional as F


def implementation_from_environment():
    implementation = os.environ.get("MOE_GMM_IMPLEMENTATION", "extension")
    if implementation not in ("extension", "torch"):
        raise ValueError("MOE_GMM_IMPLEMENTATION must be 'extension' or 'torch'")
    return implementation


def validate_native_inputs(a, b):
    # Restrict the experiment to the inspected fast path, not PyTorch's hidden
    # per-expert fallback (which copies offsets to CPU). Never silently fall back.
    if not a.is_cuda or torch.version.hip or torch.cuda.get_device_capability(a.device) != (9, 0):
        raise ValueError("MOE_GMM_IMPLEMENTATION=torch requires NVIDIA SM90 CUDA")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("native grouped GEMM requires BF16 activations and weights")
    if a.ndim != 2 or b.ndim != 3 or a.shape[1] != b.shape[1]:
        raise ValueError("expected [rows,in] and [experts,in,out]")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("native grouped GEMM requires contiguous forward operands")
    if b.shape[1] % 8 or b.shape[2] % 8 or not 0 < b.shape[0] < 1024:
        raise ValueError("native grouped GEMM requires widths divisible by 8 and 1..1023 experts")
    if a.shape[0] > torch.iinfo(torch.int32).max:
        raise ValueError("routed row count exceeds native int32 offsets")


def _range(phase, kind):
    from . import model
    return model.nsys_range(model._moe_nsys_capture_active, f"grouped_gemm.{phase}.{kind}")


def native_phase(a, b, offsets, kind, grad=None):
    """The three actual native calls, also used by the isolated phase benchmark.

    dW uses column-major A / row-major B, so arbitrary segment lengths satisfy
    TMA address alignment: each boundary advances by a full aligned row, not
    by a single BF16 element. The transpose operations are views, not copies.
    """
    if kind == "forward":
        return F.grouped_mm(a, b, offs=offsets)
    if kind == "dx":
        return F.grouped_mm(grad, b.mT, offs=offsets)
    if kind == "dw":
        return F.grouped_mm(a.mT, grad, offs=offsets)
    raise ValueError(f"unknown GEMM phase: {kind}")


class _NativeGroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, offsets, phase):
        ctx.save_for_backward(a, b, offsets)
        ctx.phase = phase
        with _range(phase, "forward"):
            return native_phase(a, b, offsets, "forward")

    @staticmethod
    def backward(ctx, grad):
        a, b, offsets = ctx.saved_tensors
        grad = grad.contiguous()
        da = db = None
        if ctx.needs_input_grad[0]:
            with _range(ctx.phase, "dx"):
                da = native_phase(a, b, offsets, "dx", grad)
        if ctx.needs_input_grad[1]:
            with _range(ctx.phase, "dw"):
                db = native_phase(a, b, offsets, "dw", grad)
        return da, db, None, None


def native_gmm(a, b, offsets, phase):
    validate_native_inputs(a, b)
    return _NativeGroupedGemm.apply(a, b, offsets, phase)


class _ProfiledExtensionGemm(torch.autograd.Function):
    """Exact trans_b=False ops.py call sequence, with separate enqueue ranges.

    Used only during capture; normal baseline execution retains ops.gmm.
    """
    @staticmethod
    def forward(ctx, a, b, counts, phase):
        from grouped_gemm import backend
        assert torch.count_nonzero(counts) != 0, "Input batch_sizes should not be all zeros!"
        ctx.save_for_backward(a, b, counts)
        ctx.phase = phase
        with _range(phase, "forward"):
            return backend.gmm(a, b, counts, trans_a=False, trans_b=False)

    @staticmethod
    def backward(ctx, grad):
        from grouped_gemm import backend
        grad = grad.contiguous()
        a, b, counts = ctx.saved_tensors
        da = db = None
        if ctx.needs_input_grad[0]:
            with _range(ctx.phase, "dx"):
                da = backend.gmm(grad, b, counts, trans_a=False, trans_b=True)
        if ctx.needs_input_grad[1]:
            with _range(ctx.phase, "dw"):
                db = backend.gmm(a, grad, counts, trans_a=True, trans_b=False)
        return da, db, None, None


def expert_gmm(a, b, counts, offsets, implementation, baseline, phase, capture):
    if implementation == "torch":
        return native_gmm(a, b, offsets, phase)
    # Preserve injected test/diagnostic callables, and never import the extension
    # or wrap its autograd when capture is disabled.
    if capture and getattr(baseline, "__module__", None) == "grouped_gemm.ops":
        return _ProfiledExtensionGemm.apply(a, b, counts, phase)
    return baseline(a, b, counts, trans_b=False)
