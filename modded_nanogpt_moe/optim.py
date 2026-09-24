"""Muon and the active trainer's optimizer construction."""

import math

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import AdamW
from .model import MoE

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, transposed_params=()):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        # Layout metadata comes from model construction, not optimizer state.
        self.transposed_params = set(transposed_params)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            local_params_by_shape = {}
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    bucket = (p.shape, p.dtype, p.device)
                    local_params_by_shape.setdefault(bucket, []).append(p)

            for same_shape_params in local_params_by_shape.values():
                momentums = []
                for p in same_shape_params:
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    momentums.append(state["momentum"])

                if same_shape_params[0].ndim == 3:
                    # Each packed parameter already batches E independent matrices.
                    # No restacking across layers. Use the reference contiguous
                    # [out,in] orientation for Muon's asymmetric aspect-ratio
                    # scale and BF16 reduction order. These optimizer workspaces
                    # also keep muon_update's in-place lerp off packed gradients.
                    for p, momentum in zip(same_shape_params, momentums):
                        transposed = p in self.transposed_params
                        grad = p.grad.mT.contiguous() if transposed else p.grad
                        momentum_batch = momentum.mT.contiguous() if transposed else momentum
                        update = muon_update(grad, momentum_batch, mu=group["mu"])
                        if transposed:
                            momentum.copy_(momentum_batch.mT)
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.mT if transposed else update, alpha=-group["lr"])
                    continue

                if len(same_shape_params) == 1:
                    p = same_shape_params[0]
                    update = muon_update(p.grad, momentums[0], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                    continue

                grad_batch = torch.stack([p.grad for p in same_shape_params])
                momentum_batch = torch.stack(momentums)
                update_batch = muon_update(
                    grad_batch, momentum_batch, mu=group["mu"])
                torch._foreach_copy_(momentums, momentum_batch.unbind())
                torch._foreach_mul_(
                    same_shape_params,
                    1 - group["lr"] * group["weight_decay"],
                )
                torch._foreach_add_(
                    same_shape_params, update_batch.unbind(), alpha=-group["lr"])

            params_pad = params + [torch.empty_like(params[-1])] * (
                (-len(params)) % world_size)
            for base_i in range(0, len(params), world_size):
                dist.all_gather(
                    params_pad[base_i:base_i + world_size],
                    params_pad[base_i + rank],
                )


def moe_router_weights(model):
    """Identify router weights by module identity, not a parameter-name substring."""
    return [module.router.weight for module in model.modules() if isinstance(module, MoE)]


def build_optimizers(model, config=None):
    config = config or {
        "adamw": {
            "group_lrs": [0.7, 0.004, 0.015],
            "betas": [0.8, 0.95],
            "eps": 1e-10,
            "weight_decay": 0.001,
            "fused": True,
        },
        "muon": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95},
    }
    adamw = config["adamw"]
    muon = config["muon"]
    router_optimizer = config.get("router_optimizer", "muon")
    if router_optimizer not in ("muon", "adamw"):
        raise ValueError("optimizers.router_optimizer must be 'muon' or 'adamw'")
    router_adamw_lr = config.get("router_adamw_lr")
    if router_adamw_lr is not None and (
            isinstance(router_adamw_lr, bool)
            or not isinstance(router_adamw_lr, (int, float))
            or not math.isfinite(router_adamw_lr)
            or router_adamw_lr <= 0):
        raise ValueError("optimizers.router_adamw_lr must be finite and positive")
    adamw_router_weights = set(moe_router_weights(model)) if router_optimizer == "adamw" else set()
    embed_lr, head_lr, scalar_lr = adamw["group_lrs"]
    packed_experts = [module for module in model.modules()
                      if isinstance(module, MoE) and module.moe_parameter_layout == "packed"]
    packed_biases = {p for module in packed_experts
                     for p in (module.fc_bias, module.proj_bias)}
    packed_weights = {p for module in packed_experts
                      for p in (module.fc_weight, module.proj_weight)}
    adamw_groups = [
        dict(params=[model.embed.weight], lr=embed_lr),
        dict(params=[model.proj.weight], lr=head_lr),
        dict(params=[p for p in model.parameters()
                     if p.ndim < 2 or p in packed_biases
                     or (p in adamw_router_weights and router_adamw_lr is None)],
             lr=scalar_lr),
    ]
    if adamw_router_weights and router_adamw_lr is not None:
        adamw_groups.append(dict(
            params=[p for p in model.parameters() if p in adamw_router_weights],
            lr=router_adamw_lr,
        ))
    optimizer1 = AdamW(adamw_groups,
                       betas=tuple(adamw["betas"]), eps=adamw["eps"],
                       weight_decay=adamw["weight_decay"], fused=adamw["fused"])
    optimizer2 = Muon([p for p in model.blocks.parameters()
                       if p.ndim >= 2 and p not in packed_biases
                       and p not in adamw_router_weights],
                      lr=muon["lr"], weight_decay=muon["weight_decay"], mu=muon["mu"],
                      transposed_params=packed_weights)
    optimizers = [optimizer1, optimizer2]
    assigned = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
    assert len(assigned) == len(set(assigned)), "optimizer parameter ownership must be exclusive"
    assert set(assigned) == set(model.parameters()), "optimizers must cover every model parameter"
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers
