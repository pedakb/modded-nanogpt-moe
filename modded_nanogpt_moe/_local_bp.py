"""Mixed grouped local-BP VJP: BP dgrad and Grad-EM parameter gradients.

For each selected route p is the actual (activation-dtype) forward weight,
whereas q is the FP32 selected-logit Grad-EM posterior. Linearity gives
v_GE = (q/p) * v_BP in real arithmetic, including unnormalized Top-K. Rescaling
after a rounded GEMM changes GE FC1 rounding, but never the ordinary BP VJP.
"""

import torch

from ._grouped_gemm import expert_dgrad, expert_wgrad


class RouterInputOnly(torch.autograd.Function):
    """Route the ordinary logit VJP to x; the detached linear gets GE wgrad.

    Keeping softmax/top-k/normalization on their existing graph preserves the
    precise BP router derivative and dtype casts, including unnormalized Top-K.
    The parameter linear has no x edge, so it never computes GE router dgrad.
    """

    @staticmethod
    def forward(ctx, x, logits, weight):
        ctx.save_for_backward(weight)
        return logits

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        return grad @ weight, None, None


def _bias_grad(grad, counts):
    if grad.is_cuda:
        from ._segmented_bias import segmented_bias_grad
        return segmented_bias_grad(grad, counts)
    values = grad.float() if grad.dtype in (torch.float16, torch.bfloat16) else grad
    return torch.segment_reduce(values, "sum", lengths=counts, axis=0, unsafe=True).to(grad.dtype)


def _cpu_signals(out, logits, weights, indices, order, grad, eta):
    from .grad_em import grad_em_reference
    selected = torch.empty_like(out)
    selected[order] = out
    selected = selected.view(*indices.shape, out.shape[-1])
    result = grad_em_reference(logits, indices, selected, grad, eta)
    # A zero forward weight can still have nonzero q. Use an unweighted
    # signal for that row, then zero ONLY its BP VJP after the activation.
    safe_weights = torch.where(weights == 0, 1, weights)
    bp = (grad[:, None, :] * safe_weights[..., None]).flatten(0, 1)[order]
    ge = result.grad_expert.flatten(0, 1)[order].to(out.dtype)
    # Match the ordinary combine: round each product before reducing.
    grad_weights = (selected * grad[:, None, :]).sum(-1)
    return bp, ge, result.grad_logits.to(logits.dtype), result.q, grad_weights, result.v


def _activation_backward(grad_hidden, pre, weights, q, order, need_ge):
    if pre.is_cuda:
        from ._local_bp_cuda import activation_backward
        return activation_backward(grad_hidden, pre, weights, q, order, need_ge)
    work = (grad_hidden * (2 * pre.relu())).masked_fill_(pre <= 0, 0)
    p = weights.flatten()[order, None].float()
    ge = None
    if need_ge:
        # Divide the signal first: explicitly forming q/p can overflow for
        # subnormal p even when the final rescaled signal is representable.
        safe_p = torch.where(p == 0, 1, p)
        ge = (work.float() / safe_p * q.flatten()[order, None]).to(work.dtype)
    bp = work.masked_fill(p == 0, 0)
    return bp, ge


class MixedLocalBP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, x_sorted, fc_w, fc_b, proj_w, proj_b,
                h_pre, h_act, out_sorted, logits, weights, indices, order,
                counts, counts_device, offsets, eta, implementation, observer):
        if out_sorted.is_cuda:
            from ._grad_em_cuda import cuda_forward
            out, rows = cuda_forward(out_sorted, logits, weights, indices, order)
        else:
            from .model import combine_expert_outputs
            out = combine_expert_outputs(out_sorted, weights, order)
            rows = None
        ctx.save_for_backward(x_sorted, fc_w, proj_w, h_pre, h_act, out_sorted,
                              logits, weights, indices, order, counts, counts_device,
                              offsets, rows)
        ctx.eta, ctx.implementation, ctx.observer = eta, implementation, observer
        ctx.input_shape = x.shape
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        (x_sorted, fc_w, proj_w, pre, act, out, logits, weights, indices,
         order, counts, counts_device, offsets, rows) = ctx.saved_tensors
        needs = ctx.needs_input_grad
        need_fc1 = needs[2] or needs[3]
        if out.is_cuda:
            from ._local_bp_cuda import combine_backward
            bp_out, ge_out, ge_logits, q, bp_weights, v = combine_backward(
                out, logits, weights, indices, rows, grad, ctx.eta,
                needs[4] or needs[5], needs[9], needs[10], ctx.observer is not None)
        else:
            bp_out, ge_out, ge_logits, q, bp_weights, v = _cpu_signals(
                out, logits, weights, indices, order, grad, ctx.eta)
        if ctx.observer is not None:
            ctx.observer(v)
        # Exactly one wgrad per trainable expert weight; no GE dgrad.
        proj_grad = (expert_wgrad(act, ge_out, counts, offsets, ctx.implementation, "fc2")
                     if needs[4] else None)
        proj_bias = _bias_grad(ge_out, counts_device) if needs[5] else None
        del ge_out
        hidden = expert_dgrad(bp_out, proj_w, counts, offsets, ctx.implementation, "fc2")
        del bp_out
        bp_pre, ge_pre = _activation_backward(hidden, pre, weights, q, order, need_fc1)
        del hidden
        fc_grad = (expert_wgrad(x_sorted, ge_pre, counts, offsets, ctx.implementation, "fc1")
                   if needs[2] else None)
        fc_bias = _bias_grad(ge_pre, counts_device) if needs[3] else None
        del ge_pre
        grad_sorted = expert_dgrad(bp_pre, fc_w, counts, offsets, ctx.implementation, "fc1")
        grad_x = grad_sorted.new_zeros(ctx.input_shape)
        # Match the original gather's BF16 accumulation semantics.
        grad_x.index_put_((order // weights.shape[1],), grad_sorted, accumulate=True)
        return (grad_x, None, fc_grad, fc_bias, proj_grad, proj_bias,
                None, None, None, ge_logits if needs[9] else None,
                bp_weights if needs[10] else None, None, None, None, None, None,
                None, None, None)
