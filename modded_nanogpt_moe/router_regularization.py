"""Ordinary-BP router objectives, isolated from every Grad-EM boundary."""
from contextlib import ExitStack

import torch

from .model import MoE


def router_losses(logits, top_k):
    """Token means; balanced load has auxiliary loss one, independent of E/K."""
    logits = logits.flatten(0, -2).float()
    probabilities = logits.softmax(-1)
    indices = probabilities.topk(top_k, dim=-1).indices
    counts = torch.bincount(indices.flatten(), minlength=logits.shape[-1]).detach()
    frequency = counts.float() * (logits.shape[-1] / indices.numel())
    aux = (frequency * probabilities.mean(0)).sum()
    z = logits.logsumexp(-1).square().mean()
    return aux, z


def model_router_losses(model, inputs):
    """Replay only the transformer with ordinary BP and average over MoE layers.

    A separate graph is necessary: adding a loss to router logits in the LM
    graph would send its upstream gradient through earlier Grad-EM Functions.
    No parameters, RNG draws, routing semantics or LM graph are changed here.
    Hooks and mode/diagnostic attributes are restored even if the pass fails.
    """
    values = []
    with ExitStack() as stack:
        for module in model.modules():
            if not isinstance(module, MoE):
                continue
            for name, value in (("moe_backward", "standard"),
                                ("_routing_diagnostics", None),
                                ("_grad_em_diagnostics", None)):
                old = getattr(module, name)
                stack.callback(setattr, module, name, old)
                setattr(module, name, value)
            def observe(router, args, logits, top_k=module.top_k):
                values.append(router_losses(logits, top_k))
            handle = module.router.register_forward_hook(observe)
            stack.callback(handle.remove)
        x = model.norm1(model.embed(inputs))
        for block in model.blocks:
            x = block(x)
    if not values:
        raise ValueError("router regularization requires MoE layers")
    return tuple(torch.stack([pair[i] for pair in values]).mean() for i in (0, 1))
