"""Mixed grouped VJP with independent parameter and boundary corrections.

For each selected route p is the actual (activation-dtype) forward weight,
whereas q is the FP32 selected-logit Grad-EM posterior. Linearity gives
v_GE = (q/p) * v_BP in real arithmetic, including unnormalized Top-K. Rescaling
after a rounded GEMM changes GE FC1 rounding. Mix strengths lambda and
alpha*lambda reuse that one FC2 dgrad; alpha=0 retains the ordinary BP VJP.
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


class MixedRouter(torch.autograd.Function):
    """Join BP/GE logit signals before one router wgrad and one dgrad."""

    @staticmethod
    def forward(ctx, x, logits, weight, mix_lambda, boundary_mix):
        ctx.save_for_backward(weight)
        ctx.mix_lambda, ctx.boundary_mix = mix_lambda, boundary_mix
        return logits.view_as(logits), logits.view_as(logits)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, ge, bp):
        (weight,) = ctx.saved_tensors
        parameter = ge if ctx.mix_lambda == 1 else (
            bp.float() + ctx.mix_lambda * (ge.float() - bp.float())).to(bp.dtype)
        boundary = bp if ctx.boundary_mix == 0 else (
            bp.float() + ctx.boundary_mix * (ge.float() - bp.float())).to(bp.dtype)
        return (boundary @ weight if ctx.needs_input_grad[0] else None,
                parameter if ctx.needs_input_grad[1] else None,
                None, None, None)


def _bias_grad(grad, counts):
    if grad.is_cuda:
        from ._segmented_bias import segmented_bias_grad
        return segmented_bias_grad(grad, counts)
    values = grad.float() if grad.dtype in (torch.float16, torch.bfloat16) else grad
    return torch.segment_reduce(values, "sum", lengths=counts, axis=0, unsafe=True).to(grad.dtype)


def _cpu_signals(out, logits, weights, indices, order, grad, eta, mix_lambda=1.0,
                 score_normalization="none", score_norm_eps=1e-6):
    from .grad_em import grad_em_reference
    selected = torch.empty_like(out)
    selected[order] = out
    selected = selected.view(*indices.shape, out.shape[-1])
    result = grad_em_reference(
        logits, indices, selected, grad, eta, score_normalization, score_norm_eps)
    # A zero forward weight can still have nonzero q. Use an unweighted
    # signal for that row, then zero ONLY its BP VJP after the activation.
    safe_weights = torch.where(weights == 0, 1, weights)
    bp = (grad[:, None, :] * safe_weights[..., None]).flatten(0, 1)[order]
    ge = result.grad_expert.flatten(0, 1)[order].to(out.dtype)
    if mix_lambda != 1:
        mixed = weights.float() + mix_lambda * (result.q - weights.float())
        ge = (mixed[..., None] * grad.float()[:, None, :]).flatten(0, 1)[order].to(out.dtype)
    # Match the ordinary combine: round each product before reducing.
    grad_weights = (selected * grad[:, None, :]).sum(-1)
    return bp, ge, result.grad_logits.to(logits.dtype), result.q, grad_weights, result.v


def _activation_backward(grad_hidden, pre, weights, q, order, need_ge,
                         mix_lambda=1.0, boundary_mix=0.0):
    if pre.is_cuda:
        from ._local_bp_cuda import activation_backward
        return activation_backward(grad_hidden, pre, weights, q, order, need_ge,
                                   mix_lambda, boundary_mix)
    work = (grad_hidden * (2 * pre.relu())).masked_fill_(pre <= 0, 0)
    p = weights.flatten()[order, None].float()
    ge = None
    bp = work.masked_fill(p == 0, 0)
    if need_ge or boundary_mix != 0:
        # Divide the signal first: explicitly forming q/p can overflow for
        # subnormal p even when the final rescaled signal is representable.
        safe_p = torch.where(p == 0, 1, p)
        full_ge = work.float() / safe_p * q.flatten()[order, None]
        if need_ge:
            ge = (full_ge if mix_lambda == 1 else
                  bp.float() + mix_lambda * (full_ge - bp.float())).to(work.dtype)
        if boundary_mix != 0:
            bp = (bp.float() + boundary_mix * (full_ge - bp.float())).to(work.dtype)
    return bp, ge


class MixedLocalBP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, x_sorted, fc_w, fc_b, proj_w, proj_b,
                h_pre, h_act, out_sorted, logits, weights, indices, order,
                counts, counts_device, offsets, eta, implementation, observer,
                mix_lambda=1.0, boundary_mix=0.0,
                score_normalization="none", score_norm_eps=1e-6):
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
        ctx.mix_lambda, ctx.boundary_mix = mix_lambda, boundary_mix
        ctx.score_normalization = score_normalization
        ctx.score_norm_eps = score_norm_eps
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
                needs[4] or needs[5], needs[9], needs[10], ctx.observer is not None,
                ctx.mix_lambda, ctx.score_normalization, ctx.score_norm_eps)
        else:
            bp_out, ge_out, ge_logits, q, bp_weights, v = _cpu_signals(
                out, logits, weights, indices, order, grad, ctx.eta,
                ctx.mix_lambda, ctx.score_normalization, ctx.score_norm_eps)
        if ctx.observer is not None:
            ctx.observer(v)
        # Exactly one wgrad per trainable expert weight; no GE dgrad.
        proj_grad = (expert_wgrad(act, ge_out, counts, offsets, ctx.implementation, "fc2")
                     if needs[4] else None)
        proj_bias = _bias_grad(ge_out, counts_device) if needs[5] else None
        del ge_out
        hidden = expert_dgrad(bp_out, proj_w, counts, offsets, ctx.implementation, "fc2")
        del bp_out
        bp_pre, ge_pre = _activation_backward(
            hidden, pre, weights, q, order, need_fc1, ctx.mix_lambda, ctx.boundary_mix)
        del hidden
        fc_grad = (expert_wgrad(x_sorted, ge_pre, counts, offsets, ctx.implementation, "fc1")
                   if needs[2] else None)
        fc_bias = _bias_grad(ge_pre, counts_device) if needs[3] else None
        del ge_pre
        grad_sorted = expert_dgrad(bp_pre, fc_w, counts, offsets, ctx.implementation, "fc1")
        grad_x = grad_sorted.new_zeros(ctx.input_shape)
        # Match the original gather's BF16 accumulation semantics.
        grad_x.index_put_((order // weights.shape[1],), grad_sorted, accumulate=True)
        gradients = (grad_x, None, fc_grad, fc_bias, proj_grad, proj_bias,
                None, None, None, ge_logits if needs[9] else None,
                bp_weights if needs[10] else None, None, None, None, None, None,
                None, None, None)
        return gradients + (None,) * (len(needs) - len(gradients))
