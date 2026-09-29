"""Mixed pointwise kernels; forward and alpha=1 global kernels stay unchanged."""
import torch
import triton
import triton.language as tl

from ._grad_em_cuda import _grad_em_router_backward


@triton.jit
def _mixed_combine_backward(X, Z, Weights, Indices, Rows, Grad,
                            BP, GE, Q, GradWeights, V,
                            D: tl.constexpr, K: tl.constexpr,
                            XR: tl.constexpr, XC: tl.constexpr,
                            ZR: tl.constexpr, ZC: tl.constexpr,
                            WR: tl.constexpr, WC: tl.constexpr,
                            IR: tl.constexpr, IC: tl.constexpr,
                            GR: tl.constexpr, GC: tl.constexpr,
                            ETA: tl.constexpr, NEED_GE: tl.constexpr,
                            NEED_WEIGHTS: tl.constexpr, SAVE_V: tl.constexpr,
                            SLOTS: tl.constexpr, COLS: tl.constexpr,
                            MIX_LAMBDA: tl.constexpr = 1.0):
    token = tl.program_id(0)
    slots, columns = tl.arange(0, SLOTS), tl.arange(0, COLS)
    rows = tl.load(Rows + token * K + slots, slots < K, other=0)
    ids = tl.load(Indices + token * IR + slots * IC, slots < K, other=0)
    valid = (slots[:, None] < K) & (columns[None, :] < D)
    grad = tl.load(Grad + token * GR + columns * GC, columns < D, other=0).to(tl.float32)
    values = tl.load(X + rows[:, None] * XR + columns[None, :] * XC, valid, other=0).to(tl.float32)
    # GE sensitivity uses FP32 products; ordinary BP's mixing-weight VJP
    # rounds products to the activation dtype BEFORE the FP32 reduction.
    products = values * grad[None, :]
    v = tl.sum(products, axis=1)
    selected = tl.load(Z + token * ZR + ids * ZC, slots < K, other=0).to(tl.float32)
    scores = tl.where(slots < K, selected - ETA * v, -float("inf"))
    exps = tl.exp(scores - tl.max(scores, axis=0))
    q = exps / tl.sum(exps, axis=0)
    tl.store(Q + token * K + slots, q, slots < K)
    if SAVE_V:
        tl.store(V + token * K + slots, v, slots < K)
    if NEED_WEIGHTS:
        rounded = products.to(X.dtype.element_ty).to(tl.float32)
        tl.store(GradWeights + token * K + slots, tl.sum(rounded, axis=1), slots < K)
    if NEED_GE:
        # q*g directly avoids a second rounding through rho*(p*g).
        mixed = q
        if MIX_LAMBDA != 1:
            p = tl.load(Weights + token * WR + slots * WC, slots < K, other=0).to(tl.float32)
            mixed = p + MIX_LAMBDA * (q - p)
        tl.store(GE + rows[:, None] * D + columns[None, :], mixed[:, None] * grad[None, :], valid)
    p = tl.load(Weights + token * WR + slots * WC, slots < K, other=0).to(tl.float32)
    safe_p = tl.where(p == 0, 1., p)
    tl.store(BP + rows[:, None] * D + columns[None, :], safe_p[:, None] * grad[None, :], valid)


@triton.jit
def _mixed_activation_backward(Hidden, Pre, Weights, Q, Order, BP, GE,
                               SIZE: tl.constexpr, H: tl.constexpr,
                               NEED_GE: tl.constexpr, BLOCK: tl.constexpr,
                               MIX_LAMBDA: tl.constexpr = 1.0,
                               BOUNDARY_MIX: tl.constexpr = 0.0):
    elements = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = elements < SIZE
    assignment = tl.load(Order + elements // H, valid, other=0)
    p = tl.load(Weights + assignment, valid, other=1).to(tl.float32)
    pre = tl.load(Pre + elements, valid, other=0).to(tl.float32)
    hidden = tl.load(Hidden + elements, valid, other=0).to(tl.float32)
    # Preserve ordinary square -> ReLU backward's two rounding points.
    derivative = (2. * tl.maximum(pre, 0.)).to(Pre.dtype.element_ty).to(tl.float32)
    work = (hidden * derivative).to(Pre.dtype.element_ty).to(tl.float32)
    work = tl.where(pre <= 0, 0., work)
    bp = tl.where(p == 0, 0., work)
    if NEED_GE or BOUNDARY_MIX != 0:
        q = tl.load(Q + assignment, valid, other=0)
        safe_p = tl.where(p == 0, 1., p)
        # Do not form an overflowing q/p. div_rn preserves subnormal p,
        # unlike approximate reciprocal multiplication with flush-to-zero.
        scaled = tl.div_rn(work, safe_p) * q
        if NEED_GE:
            parameter = scaled
            if MIX_LAMBDA != 1:
                parameter = bp + MIX_LAMBDA * (scaled - bp)
            tl.store(GE + elements, parameter, valid)
        if BOUNDARY_MIX != 0:
            bp = bp + BOUNDARY_MIX * (scaled - bp)
    tl.store(BP + elements, bp, valid)


def combine_backward(out, logits, weights, indices, rows, grad, eta,
                     need_ge, need_logits, need_weights, save_v, mix_lambda=1.0):
    n, k = weights.shape
    d, e = out.shape[1], logits.shape[1]
    bp = torch.empty_like(out)
    ge = torch.empty_like(out) if need_ge else None
    gz = torch.empty_like(logits) if need_logits else None
    gw = torch.empty_like(weights) if need_weights else None
    q = torch.empty((n, k), device=out.device, dtype=torch.float32)
    v = torch.empty_like(q) if save_v else None
    if n:
        with torch.cuda.device(out.device):
            _mixed_combine_backward[(n,)](
                out, logits, weights, indices, rows, grad, bp, ge, q, gw, v,
                d, k, *out.stride(), *logits.stride(), *weights.stride(),
                *indices.stride(), *grad.stride(), eta, need_ge, need_weights, save_v,
                triton.next_power_of_2(k), triton.next_power_of_2(d),
                mix_lambda,
                num_warps=4, enable_fp_fusion=False)
            if need_logits:
                _grad_em_router_backward[(n,)](
                    logits, indices, q, gz, eta, e, k, *logits.stride(), *indices.stride(),
                    triton.next_power_of_2(e), triton.next_power_of_2(k),
                    num_warps=4, enable_fp_fusion=False)
    return bp, ge, gz, q, gw, v


def activation_backward(hidden, pre, weights, q, order, need_ge,
                        mix_lambda=1.0, boundary_mix=0.0):
    bp = torch.empty_like(pre)
    ge = torch.empty_like(pre) if need_ge else None
    if pre.numel():
        with torch.cuda.device(pre.device):
            _mixed_activation_backward[(triton.cdiv(pre.numel(), 1024),)](
                hidden, pre, weights, q, order, bp, ge, pre.numel(), pre.shape[1],
                need_ge, 1024, mix_lambda, boundary_mix, num_warps=4, enable_fp_fusion=False)
    return bp, ge
