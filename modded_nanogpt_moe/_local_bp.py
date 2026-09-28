"""Input-only VJPs for grouped local-BP Grad-EM.

The ordinary grouped graph receives detached inputs and supplies all Grad-EM
parameter gradients. These forward identities reuse its activations/weights
without another expert or router forward. Their backwards supply only ordinary
BP input gradients, with no wgrad, bias reduction, or parameter-gradient edge.
"""

import torch

from ._grouped_gemm import expert_dgrad


class RouterInputOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, logits, weight):
        ctx.save_for_backward(weight)
        return logits

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        return grad @ weight, None, None


class ExpertInputOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_sorted, h_pre, fc_weight, proj_weight,
                counts, offsets, order, top_k, implementation):
        ctx.save_for_backward(h_pre, fc_weight, proj_weight, counts, offsets, order)
        ctx.top_k = top_k
        ctx.implementation = implementation
        ctx.input_shape = x.shape
        return out_sorted

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        h_pre, fc_weight, proj_weight, counts, offsets, order = ctx.saved_tensors
        grad_hidden = expert_dgrad(
            grad, proj_weight, counts, offsets, ctx.implementation, "fc2")
        # Match square -> ReLU backward, including activation-dtype rounding
        # after 2*relu(h_pre) and after multiplication by grad_hidden.
        grad_pre = (grad_hidden * (2 * h_pre.relu())).masked_fill_(h_pre <= 0, 0)
        grad_sorted = expert_dgrad(
            grad_pre, fc_weight, counts, offsets, ctx.implementation, "fc1")
        grad_x = grad_sorted.new_zeros(ctx.input_shape)
        # Match the advanced-index gather's IndexBackward, including BF16
        # accumulation semantics (index_add_ rounds differently on CPU).
        grad_x.index_put_((order // ctx.top_k,), grad_sorted, accumulate=True)
        return grad_x, None, None, None, None, None, None, None, None, None


class LocalBPOutput(torch.autograd.Function):
    """Keep the existing Grad-EM forward; send g into both disjoint VJPs."""

    @staticmethod
    def forward(ctx, parameter_output, input_output):
        return parameter_output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        return grad, grad
