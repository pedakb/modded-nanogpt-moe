"""Grad-EM kernels with selected-support KL router gradients.

Imported lazily, never by standard mode or the CPU reference. No atomics,
activation-sized scratch, or additional expert computation.
"""
import torch
import triton
import triton.language as tl

from ._combine import _combine_assignment_rows, _combine_forward
from .config import (validate_grad_em_eta, validate_grad_em_score_normalization,
                     validate_grad_em_score_norm_eps)


@triton.jit
def _grad_em_responsibilities(selected, values, valid,
                              ETA: tl.constexpr,
                              SCORE_NORMALIZATION: tl.constexpr,
                              SCORE_NORM_EPS: tl.constexpr):
    selected = tl.where(valid, selected, -float("inf"))
    scores = values
    base = tl.full(values.shape, 0., tl.float32)
    if SCORE_NORMALIZATION:
        base_exp = tl.exp(selected - tl.max(selected, axis=0))
        base = base_exp / tl.sum(base_exp, axis=0)
        # As in the CPU oracle, shifting by slot 0 gives exact constant-score
        # centering despite FP32 probability-sum roundoff; no extra reduction.
        origin = tl.gather(values, tl.full((1,), 0, tl.int32), axis=0)
        shifted = values - origin
        mean = tl.sum(tl.where(valid, base * shifted, 0.), axis=0)
        centered = shifted - mean
        variance = tl.sum(
            tl.where(valid, base * centered * centered, 0.), axis=0)
        scale = tl.maximum(tl.sqrt(variance), SCORE_NORM_EPS)
        scores = centered / scale
    tilted = tl.where(valid, selected - ETA * scores, -float("inf"))
    exp_tilted = tl.exp(tilted - tl.max(tilted, axis=0))
    return exp_tilted / tl.sum(exp_tilted, axis=0), base


@triton.jit
def _grad_em_expert_backward(X, Z, Indices, Rows, Grad, GradX, Q, V,
                             D: tl.constexpr, K: tl.constexpr,
                             XR: tl.constexpr, XC: tl.constexpr,
                             ZR: tl.constexpr, ZC: tl.constexpr,
                             IR: tl.constexpr, IC: tl.constexpr,
                             GR: tl.constexpr, GC: tl.constexpr,
                             ETA: tl.constexpr, NEED_X: tl.constexpr,
                             SCORE_NORMALIZATION: tl.constexpr,
                             SAVE_V: tl.constexpr,
                             SLOTS: tl.constexpr, COLS: tl.constexpr,
                             Weights=None, GradWeights=None,
                             WR: tl.constexpr = 0, WC: tl.constexpr = 0,
                             MIX_LAMBDA: tl.constexpr = 1.0,
                             NEED_WEIGHTS: tl.constexpr = False,
                             Base=None, SAVE_BASE: tl.constexpr = False,
                             SCORE_NORM_EPS: tl.constexpr = 1e-6):
    token = tl.program_id(0)
    slots, columns = tl.arange(0, SLOTS), tl.arange(0, COLS)
    rows = tl.load(Rows + token * K + slots, slots < K, other=0)
    ids = tl.load(Indices + token * IR + slots * IC, slots < K, other=0)
    grad = tl.load(Grad + token * GR + columns * GC, columns < D, other=0).to(tl.float32)
    values = tl.load(X + rows[:, None] * XR + columns[None, :] * XC,
                     (slots[:, None] < K) & (columns[None, :] < D), other=0).to(tl.float32)
    # Unlike standard combine's weight gradient, do NOT round products to BF16.
    v = tl.sum(values * grad[None, :], axis=1)
    selected = tl.load(Z + token * ZR + ids * ZC, slots < K, other=0).to(tl.float32)
    q, base = _grad_em_responsibilities(
        selected, v, slots < K, ETA, SCORE_NORMALIZATION, SCORE_NORM_EPS)
    if SAVE_BASE:
        tl.store(Base + token * K + slots, base, slots < K)
    tl.store(Q + token * K + slots, q, slots < K)
    if SAVE_V:
        tl.store(V + token * K + slots, v, slots < K)
    if NEED_WEIGHTS:
        # Match ordinary combine: activation-dtype product and reduction result.
        products = (values * grad[None, :]).to(X.dtype.element_ty).to(tl.float32)
        bp_weight = tl.sum(products, axis=1).to(Weights.dtype.element_ty).to(tl.float32)
        tl.store(GradWeights + token * K + slots, (1. - MIX_LAMBDA) * bp_weight, slots < K)
    if NEED_X:
        mixed = q
        if MIX_LAMBDA != 1:
            p = tl.load(Weights + token * WR + slots * WC, slots < K, other=0).to(tl.float32)
            mixed = p + MIX_LAMBDA * (q - p)
        # A permutation gives each sorted row exactly one token/slot writer.
        tl.store(GradX + rows[:, None] * D + columns[None, :],
                 mixed[:, None] * grad[None, :],
                 (slots[:, None] < K) & (columns[None, :] < D))


@triton.jit
def _grad_em_router_backward(Z, Indices, Q, GradZ,
                             ETA: tl.constexpr,
                             E: tl.constexpr, K: tl.constexpr,
                             ZR: tl.constexpr, ZC: tl.constexpr,
                             IR: tl.constexpr, IC: tl.constexpr,
                             EXPERTS: tl.constexpr, SLOTS: tl.constexpr,
                             Base=None, REUSE_BASE: tl.constexpr = False):
    token = tl.program_id(0)
    experts, slots = tl.arange(0, EXPERTS), tl.arange(0, SLOTS)
    ids = tl.load(Indices + token * IR + slots * IC, slots < K, other=-1)
    if REUSE_BASE:
        a = tl.load(Base + token * K + slots, slots < K, other=0)
    else:
        logits = tl.load(Z + token * ZR + ids * ZC, slots < K,
                         other=-float("inf")).to(tl.float32)
        exps = tl.exp(logits - tl.max(logits, axis=0))
        a = exps / tl.sum(exps, axis=0)
    q = tl.load(Q + token * K + slots, slots < K, other=0)
    selected_grad = (a - q) / ETA
    # Register-local matching gives exactly zero outside the unique support.
    # One store per dense output entry; no overlapping zero/scatter writes,
    # atomics, full-E softmax, or global dense intermediate.
    grad_z = tl.sum(tl.where(experts[:, None] == ids[None, :], selected_grad[None, :], 0.), axis=1)
    tl.store(GradZ + token * E + experts, grad_z, experts < E)


def cuda_forward(out_sorted, logits, weights, indices, order):
    """Reuse standard forward kernels/settings, retaining inverse rows once."""
    n, k = weights.shape
    d = out_sorted.shape[1]
    tensors = (out_sorted, logits, weights, indices, order)
    if not out_sorted.is_cuda or any(t.device != out_sorted.device for t in tensors):
        raise ValueError("Grad-EM CUDA inputs must share a CUDA device")
    if (out_sorted.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or weights.dtype != out_sorted.dtype
            or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError("Grad-EM CUDA requires FP16/BF16/FP32 activations and logits")
    if (indices.dtype != torch.int64 or order.dtype != torch.int64
            or indices.shape != (n, k) or logits.ndim != 2 or logits.shape[0] != n
            or out_sorted.shape != (n*k, d) or order.shape != (n*k,)):
        raise ValueError("invalid Grad-EM CUDA routing shapes/dtypes")
    if not (1 <= k <= min(32, logits.shape[1]) and 1 <= d <= 4096
            and logits.shape[1] <= 1024
            and triton.next_power_of_2(k) * triton.next_power_of_2(d) <= 32768):
        raise ValueError("unsupported Grad-EM CUDA tile geometry (K<=32, D<=4096, E<=1024, Ktile*Dtile<=32768)")
    rows = torch.empty((n*k,), device=order.device, dtype=torch.int64)
    output = torch.empty((n, d), device=out_sorted.device, dtype=out_sorted.dtype)
    if n:
        with torch.cuda.device(out_sorted.device):
            _combine_assignment_rows[(triton.cdiv(n*k, 256),)](
                order, rows, n*k, order.stride(0), 256)
            _combine_forward[(n, triton.cdiv(d, 128))](
                out_sorted, weights, rows, output, d, k,
                *out_sorted.stride(), *weights.stride(),
                triton.next_power_of_2(k), 128, num_warps=4, enable_fp_fusion=False)
    return output, rows


def cuda_backward(out_sorted, logits, indices, rows, grad_output, eta,
                  need_x=True, need_logits=True, *, save_v=False,
                  weights=None, mix_lambda=1.0, grad_weights=None,
                  score_normalization="none", score_norm_eps=1e-6):
    """Return gradients plus compact q (and optional test-only FP32 v)."""
    validate_grad_em_eta(eta)
    validate_grad_em_score_normalization(score_normalization)
    validate_grad_em_score_norm_eps(score_norm_eps)
    n, k = indices.shape
    d, e = out_sorted.shape[1], logits.shape[1]
    gx = torch.empty(out_sorted.shape, device=out_sorted.device, dtype=out_sorted.dtype) if need_x else None
    gz = torch.empty(logits.shape, device=logits.device, dtype=logits.dtype) if need_logits else None
    q = torch.empty((n, k), device=logits.device, dtype=torch.float32)
    reuse_base = score_normalization == "std" and need_logits
    # Compact scratch only in normalized mode; raw mode is unchanged.
    base = torch.empty((n, k), device=logits.device, dtype=torch.float32) if reuse_base else None
    v = torch.empty((n, k), device=logits.device, dtype=torch.float32) if save_v else None
    if n and (need_x or need_logits or save_v or grad_weights is not None):
        with torch.cuda.device(out_sorted.device):
            _grad_em_expert_backward[(n,)](
                out_sorted, logits, indices, rows, grad_output,
                gx if need_x else out_sorted, q, v if save_v else q,
                d, k, *out_sorted.stride(), *logits.stride(), *indices.stride(),
                *grad_output.stride(), eta, need_x,
                score_normalization == "std", save_v,
                triton.next_power_of_2(k), triton.next_power_of_2(d),
                weights, grad_weights, *(weights.stride() if weights is not None else (0, 0)),
                mix_lambda, grad_weights is not None,
                Base=base, SAVE_BASE=reuse_base, SCORE_NORM_EPS=score_norm_eps,
                num_warps=4, enable_fp_fusion=False)
            if need_logits:
                _grad_em_router_backward[(n,)](
                    logits, indices, q, gz, eta, e, k, *logits.stride(), *indices.stride(),
                    triton.next_power_of_2(e), triton.next_power_of_2(k),
                    Base=base, REUSE_BASE=reuse_base,
                    num_warps=4, enable_fp_fusion=False)
    return gx, gz, q, v
