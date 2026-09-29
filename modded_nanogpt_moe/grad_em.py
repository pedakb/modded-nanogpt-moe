"""Grad-EM oracle and CPU/CUDA combine boundary.

Only selected expert outputs are supplied. All arithmetic/results are FP32,
detached, including products before the dot-product reduction. No activation-
dtype rounding or extra loss/accumulation scaling is applied in the oracle.
The boundary casts returned gradients to their original activation dtypes.
"""

from typing import NamedTuple

import torch

from .config import validate_grad_em_eta, validate_grad_em_lambda


def require_grad_em_device(device):
    if device.type not in ("cpu", "cuda"):
        raise NotImplementedError(
            "Grad-EM supports only CPU and CUDA devices")


class LocalBPGradEM(torch.autograd.Function):
    """Use ordinary BP at the MoE input and Grad-EM for local parameters.

    The eager ModuleList forward is recorded once behind this boundary. Its
    ordinary output graph supplies only the input VJP. Separate VJPs from the
    same saved router logits and selected expert outputs supply parameter
    gradients, so replacement signals never cross the MoE input boundary.
    """

    @staticmethod
    def forward(ctx, x, module, *parameters):
        validate_grad_em_eta(module.grad_em_eta)
        # A fresh local leaf prevents the graph recorded inside the custom
        # Function from reaching upstream. backward() explicitly returns its
        # ordinary VJP to the original x input.
        inner_x = x.detach().requires_grad_(x.requires_grad)
        with torch.enable_grad():
            output, logits, indices, selected_outputs = module._forward_loop(
                inner_x, return_local_components=True)
        # Keep the inner graph tensors as Python attributes: saving the tensor
        # returned by this Function would expose its outer custom grad_fn in
        # backward instead of the ordinary graph recorded above.
        ctx.inner_x = inner_x
        ctx.output = output
        ctx.logits = logits
        ctx.indices = indices
        ctx.selected_outputs = selected_outputs
        ctx.parameters = parameters
        ctx.eta = module.grad_em_eta
        ctx.sensitivity_observer = (
            module._routing_diagnostics.observe_sensitivity
            if module._routing_diagnostics is not None else None)
        ctx.logit_gradient_observer = (
            module._routing_diagnostics.observe_logit_gradient
            if module._routing_diagnostics is not None else None)
        return output.detach()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        inner_x, output = ctx.inner_x, ctx.output
        logits, indices = ctx.logits, ctx.indices
        selected_outputs, parameters = ctx.selected_outputs, ctx.parameters
        result = grad_em_reference(
            logits, indices, selected_outputs, grad_output.flatten(0, 1), ctx.eta)
        if ctx.sensitivity_observer is not None:
            ctx.sensitivity_observer(result.v)
        if ctx.logit_gradient_observer is not None:
            ctx.logit_gradient_observer(result.grad_logits.to(logits.dtype))

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x, = torch.autograd.grad(
                output, inner_x, grad_output, retain_graph=True,
                create_graph=False, allow_unused=False)

        parameter_grads = torch.autograd.grad(
            (selected_outputs, logits), parameters,
            (result.grad_expert.to(selected_outputs.dtype),
             result.grad_logits.to(logits.dtype)),
            create_graph=False, allow_unused=True)
        parameter_grads = tuple(
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, parameter_grads))
        return (grad_x, None, *parameter_grads)


class GradEMCombine(torch.autograd.Function):
    """Leave the expert graph intact with CPU reference and CUDA execution.

    order maps expert-sorted rows to flattened token/slot assignments. Only
    CPU backward materializes [T,K,D]; CUDA uses only compact routing scratch.
    """

    @staticmethod
    def forward(ctx, out_sorted, router_logits, topk_weights, topk_experts, order, eta,
                sensitivity_observer=None, mix_lambda=1.0):
        require_grad_em_device(out_sorted.device)
        validate_grad_em_eta(eta)
        validate_grad_em_lambda(mix_lambda, "global")
        ctx.is_cuda = out_sorted.is_cuda
        ctx.sensitivity_observer = sensitivity_observer
        ctx.mix_lambda = mix_lambda
        extra = (topk_weights,) if mix_lambda != 1 else ()
        if ctx.is_cuda:
            from ._grad_em_cuda import cuda_forward
            output, rows = cuda_forward(out_sorted, router_logits, topk_weights, topk_experts, order)
            ctx.save_for_backward(out_sorted, router_logits, topk_experts, rows, *extra)
            ctx.eta = eta
            return output
        # Lazy import avoids a model/reference import cycle. Reuse the exact
        # existing forward, including activation-dtype mixing/rounding.
        from .model import combine_expert_outputs
        ctx.save_for_backward(out_sorted, router_logits, topk_experts, order, *extra)
        ctx.eta = eta
        return combine_expert_outputs(out_sorted, topk_weights, order)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        out_sorted, logits, indices, order = ctx.saved_tensors[:4]
        weights = ctx.saved_tensors[4] if ctx.mix_lambda != 1 else None
        grad_weights = None
        if ctx.is_cuda:
            from ._grad_em_cuda import cuda_backward
            if weights is not None and ctx.needs_input_grad[2]:
                grad_weights = torch.empty_like(weights, memory_format=torch.contiguous_format)
            gx, gz, _, v = cuda_backward(
                out_sorted, logits, indices, order, grad_output, ctx.eta,
                *ctx.needs_input_grad[:2], save_v=ctx.sensitivity_observer is not None,
                weights=weights, mix_lambda=ctx.mix_lambda, grad_weights=grad_weights)
            if gz is not None and ctx.mix_lambda != 1:
                gz = gz * ctx.mix_lambda
            if ctx.sensitivity_observer is not None:
                ctx.sensitivity_observer(v)
            gradients = (gx, gz, grad_weights, None, None, None)
            return gradients + (None,) * (len(ctx.needs_input_grad) - 6)
        selected = torch.empty_like(out_sorted)
        selected[order] = out_sorted
        selected = selected.view(*indices.shape, out_sorted.shape[-1])
        result = grad_em_reference(logits, indices, selected, grad_output, ctx.eta)
        if ctx.sensitivity_observer is not None:
            ctx.sensitivity_observer(result.v)
        grad_sorted = result.grad_expert.flatten(0, 1)[order].to(out_sorted.dtype)
        grad_logits = result.grad_logits.to(logits.dtype)
        if ctx.mix_lambda != 1:
            mixed = weights.float() + ctx.mix_lambda * (result.q - weights.float())
            grad_sorted = (mixed[..., None] * grad_output.float()[:, None, :])
            grad_sorted = grad_sorted.flatten(0, 1)[order].to(out_sorted.dtype)
            # Let the original softmax/top-k/casts differentiate BP, including
            # unnormalized routing. Both router edges sum before linear backward.
            grad_weights = (selected * grad_output[:, None, :]).sum(-1) * (1 - ctx.mix_lambda)
            grad_logits = grad_logits * ctx.mix_lambda
        gradients = (grad_sorted, grad_logits, grad_weights, None, None, None)
        return gradients + (None,) * (len(ctx.needs_input_grad) - 6)


class GradEMResult(NamedTuple):
    v: torch.Tensor
    q: torch.Tensor
    a: torch.Tensor
    grad_expert: torch.Tensor
    grad_logits: torch.Tensor


@torch.no_grad()
def grad_em_reference(router_logits, topk_idx, expert_outputs, grad_h, eta=0.1):
    """Evaluate the frozen replacement-gradient contract on fixed support.

    Inputs: logits [T,E], unique selected indices [T,K], selected expert
    outputs [T,K,D], incoming output gradient [T,D], fixed finite eta > 0.
    q = softmax(selected_logits - eta * <g,h_i>) is detached.
    Return q*g for selected experts and
    (a - q) / eta on selected logits, zero elsewhere, where a is the
    selected-logit softmax. This is the gradient of KL(q.detach() || a) / eta.
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
    selected_logits = logits.gather(1, topk_idx)
    a = torch.softmax(selected_logits, dim=-1)
    q = torch.softmax(selected_logits - eta * v, dim=-1)
    grad_logits = torch.zeros_like(logits).scatter_(1, topk_idx, (a - q) / eta)
    return GradEMResult(v, q, a, q[..., None] * g, grad_logits)
