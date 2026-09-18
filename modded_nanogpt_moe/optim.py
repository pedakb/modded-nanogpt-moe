"""Muon and the active trainer's optimizer construction."""

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import AdamW

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
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

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
    embed_lr, head_lr, scalar_lr = adamw["group_lrs"]
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=embed_lr),
                        dict(params=[model.proj.weight], lr=head_lr),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=scalar_lr)],
                       betas=tuple(adamw["betas"]), eps=adamw["eps"],
                       weight_decay=adamw["weight_decay"], fused=adamw["fused"])
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                      lr=muon["lr"], weight_decay=muon["weight_decay"], mu=muon["mu"])
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers
