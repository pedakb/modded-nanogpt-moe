"""Lazy CUDA-only, atomics-free MoE combine and its two input gradients."""
import torch
import triton
import triton.language as tl


@triton.jit
def _combine_assignment_rows(Order, Rows, SIZE: tl.constexpr,
                             STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    sorted_row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    assignment = tl.load(Order + sorted_row * STRIDE, sorted_row < SIZE, other=0)
    # order is a permutation: every assignment has exactly one writer.
    tl.store(Rows + assignment, sorted_row, sorted_row < SIZE)


@triton.jit
def _combine_forward(X, Weights, Rows, Output,
                     D: tl.constexpr, K: tl.constexpr,
                     XR: tl.constexpr, XC: tl.constexpr,
                     WR: tl.constexpr, WC: tl.constexpr,
                     SLOTS: tl.constexpr, COLS: tl.constexpr):
    token = tl.program_id(0)
    columns = tl.program_id(1) * COLS + tl.arange(0, COLS)
    slots = tl.arange(0, SLOTS)
    sorted_rows = tl.load(Rows + token * K + slots, slots < K, other=0)
    weights = tl.load(Weights + token * WR + slots * WC, slots < K, other=0).to(tl.float32)
    values = tl.load(X + sorted_rows[:, None] * XR + columns[None, :] * XC,
                     (slots[:, None] < K) & (columns[None, :] < D), other=0).to(tl.float32)
    # Eager PyTorch rounds the elementwise product BEFORE summing in FP32.
    products = (values * weights[:, None]).to(X.dtype.element_ty).to(tl.float32)
    result = tl.sum(products, axis=0)
    tl.store(Output + token * D + columns, result, columns < D)


@triton.jit
def _combine_backward(X, Weights, Rows, Grad, GradX, GradWeights,
                      ASSIGNMENTS: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
                      XR: tl.constexpr, XC: tl.constexpr,
                      WR: tl.constexpr, WC: tl.constexpr,
                      GR: tl.constexpr, GC: tl.constexpr,
                      NEED_X: tl.constexpr, NEED_WEIGHTS: tl.constexpr,
                      BLOCK_ROWS: tl.constexpr, COLS: tl.constexpr):
    assignments = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    token, slot = assignments // K, assignments % K
    columns = tl.arange(0, COLS)
    valid_rows = assignments < ASSIGNMENTS
    valid = valid_rows[:, None] & (columns[None, :] < D)
    sorted_rows = tl.load(Rows + assignments, valid_rows, other=0)
    grad = tl.load(Grad + token[:, None] * GR + columns[None, :] * GC,
                   valid, other=0).to(tl.float32)
    if NEED_X:
        weights = tl.load(Weights + token * WR + slot * WC, valid_rows, other=0).to(tl.float32)
        # Unique sorted-row ownership: no atomic scatter or unsorted gradient.
        tl.store(GradX + sorted_rows[:, None] * D + columns[None, :],
                 grad * weights[:, None], valid)
    if NEED_WEIGHTS:
        values = tl.load(X + sorted_rows[:, None] * XR + columns[None, :] * XC,
                         valid, other=0).to(tl.float32)
        products = (grad * values).to(X.dtype.element_ty).to(tl.float32)
        tl.store(GradWeights + assignments, tl.sum(products, axis=1), valid_rows)


class _FusedCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out_sorted, weights, order):
        n, k = weights.shape
        d = out_sorted.shape[1]
        # Only compact integer metadata, never [N*k,D] unsorted activations.
        rows = torch.empty(order.shape, device=order.device, dtype=torch.int64)
        output = torch.empty((n, d), device=out_sorted.device, dtype=out_sorted.dtype)
        if n:
            with torch.cuda.device(out_sorted.device):
                _combine_assignment_rows[(triton.cdiv(n * k, 256),)](
                    order, rows, n * k, order.stride(0), 256,
                )
                _combine_forward[(n, triton.cdiv(d, 128))](
                    out_sorted, weights, rows, output, d, k,
                    *out_sorted.stride(), *weights.stride(),
                    triton.next_power_of_2(k), 128, num_warps=4, enable_fp_fusion=False,
                )
        ctx.save_for_backward(out_sorted, weights, rows)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        out_sorted, weights, rows = ctx.saved_tensors
        n, k = weights.shape
        d = out_sorted.shape[1]
        need_x, need_weights = ctx.needs_input_grad[:2]
        if torch.is_grad_enabled():
            # Preserve higher-order autograd without using undifferentiable kernels.
            grad_x = None
            if need_x:
                assignments = (grad_output[:, None, :] * weights[:, :, None]).reshape(n * k, d)
                grad_x = torch.empty_like(out_sorted).index_copy(0, rows, assignments)
            grad_weights = None
            if need_weights:
                values = out_sorted.index_select(0, rows).view(n, k, d)
                grad_weights = (grad_output[:, None, :] * values).sum(dim=-1)
            return grad_x, grad_weights, None
        grad_x = torch.empty(out_sorted.shape, device=out_sorted.device,
                             dtype=out_sorted.dtype) if need_x else None
        grad_weights = torch.empty(weights.shape, device=weights.device,
                                   dtype=weights.dtype) if need_weights else None
        if n:
            with torch.cuda.device(out_sorted.device):
                _combine_backward[(triton.cdiv(n * k, 4),)](
                    out_sorted, weights, rows, grad_output,
                    grad_x if need_x else out_sorted,
                    grad_weights if need_weights else weights,
                    n * k, d, k, *out_sorted.stride(), *weights.stride(),
                    *grad_output.stride(), need_x, need_weights,
                    4, triton.next_power_of_2(d), num_warps=4, enable_fp_fusion=False,
                )
        return grad_x, grad_weights, None


def fused_combine(out_sorted, weights, order):
    return _FusedCombine.apply(out_sorted, weights, order)
