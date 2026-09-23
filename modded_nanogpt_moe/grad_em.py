"""Grad-EM oracle and CPU/CUDA combine boundary.

Only selected expert outputs are supplied. All arithmetic/results are FP32,
detached, including products before the dot-product reduction. No activation-
dtype rounding or extra loss/accumulation scaling is applied in the oracle.
The boundary casts returned gradients to their original activation dtypes.
"""

from typing import NamedTuple

import torch

from .config import validate_grad_em_eta


def require_grad_em_device(device):
    if device.type not in ("cpu", "cuda"):
        raise NotImplementedError(
            "Grad-EM supports only CPU and CUDA devices")


class GradEMCombine(torch.autograd.Function):
    """Leave the expert graph intact with CPU reference and CUDA execution.

    order maps expert-sorted rows to flattened token/slot assignments. Only
    CPU backward materializes [T,K,D]; CUDA uses only compact routing scratch.
    """

    @staticmethod
    def forward(ctx, out_sorted, router_logits, topk_weights, topk_experts, order, eta):
        require_grad_em_device(out_sorted.device)
        validate_grad_em_eta(eta)
        ctx.is_cuda = out_sorted.is_cuda
        if ctx.is_cuda:
            from ._grad_em_cuda import cuda_forward
            output, rows = cuda_forward(out_sorted, router_logits, topk_weights, topk_experts, order)
            ctx.save_for_backward(out_sorted, router_logits, topk_experts, rows)
            ctx.eta = eta
            return output
        # Lazy import avoids a model/reference import cycle. Reuse the exact
        # existing forward, including activation-dtype mixing/rounding.
        from .model import combine_expert_outputs
        ctx.save_for_backward(out_sorted, router_logits, topk_experts, order)
        ctx.eta = eta
        return combine_expert_outputs(out_sorted, topk_weights, order)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        out_sorted, logits, indices, order = ctx.saved_tensors
        if ctx.is_cuda:
            from ._grad_em_cuda import cuda_backward
            gx, gz, _, _ = cuda_backward(out_sorted, logits, indices, order, grad_output,
                                         ctx.eta, *ctx.needs_input_grad[:2])
            return gx, gz, None, None, None, None
        selected = torch.empty_like(out_sorted)
        selected[order] = out_sorted
        selected = selected.view(*indices.shape, out_sorted.shape[-1])
        result = grad_em_reference(logits, indices, selected, grad_output, ctx.eta)
        grad_sorted = result.grad_expert.flatten(0, 1)[order].to(out_sorted.dtype)
        # Direct edge to original logits; NO edge through forward mixing weights.
        return grad_sorted, result.grad_logits.to(logits.dtype), None, None, None, None


class GradEMResult(NamedTuple):
    v: torch.Tensor
    q: torch.Tensor
    q_tilde: torch.Tensor
    grad_expert: torch.Tensor
    grad_logits: torch.Tensor


@torch.no_grad()
def grad_em_reference(router_logits, topk_idx, expert_outputs, grad_h, eta=0.1):
    """Evaluate the frozen replacement-gradient contract on fixed support.

    Inputs: logits [T,E], unique selected indices [T,K], selected expert
    outputs [T,K,D], incoming output gradient [T,D], fixed finite eta > 0.
    q = softmax(selected_logits - eta * <g,h_i>) is detached.
    Return q*g for selected experts and
    (softmax(logits) - q_tilde) / eta for logits.
    This deliberately is NOT the derivative of the ordinary MoE forward.
    Validation is for the reference, not a GPU hot-path implementation.
    """
    validate_grad_em_eta(eta)
    if (router_logits.ndim != 2 or topk_idx.ndim != 2
            or expert_outputs.ndim != 3 or grad_h.ndim != 2):
        raise ValueError("expected logits [T,E], indices [T,K], outputs [T,K,D], g [T,D]")
    tokens, experts = router_logits.shape
    k = topk_idx.shape[1]
    if (not 1 <= k <= experts or topk_idx.shape[0] != tokens
            or expert_outputs.shape[:2] != topk_idx.shape
            or grad_h.shape != (tokens, expert_outputs.shape[2])):
        raise ValueError("incompatible token, expert, support, or hidden dimensions")
    if topk_idx.dtype != torch.int64:
        raise ValueError("topk_idx must be int64")
    tensors = (router_logits, expert_outputs, grad_h)
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise ValueError("logits, expert outputs, and incoming gradient must be floating point")
    if any(tensor.device != topk_idx.device for tensor in tensors):
        raise ValueError("all inputs must be on the same device")
    sorted_idx = topk_idx.sort(dim=-1).values
    if ((topk_idx < 0).any() or (topk_idx >= experts).any()
            or (sorted_idx[:, 1:] == sorted_idx[:, :-1]).any()):
        raise ValueError("topk_idx must contain unique in-range experts per token")

    logits = router_logits.float()
    g = grad_h.float()[:, None, :]
    v = (g * expert_outputs.float()).sum(dim=-1)
    q = torch.softmax(logits.gather(1, topk_idx) - eta * v, dim=-1)
    p = torch.softmax(logits, dim=-1)
    q_tilde = torch.zeros_like(p).scatter_(1, topk_idx, q)
    return GradEMResult(v, q, q_tilde, q[..., None] * g, (p - q_tilde) / eta)
