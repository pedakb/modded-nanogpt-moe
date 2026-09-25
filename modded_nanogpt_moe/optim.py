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
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95, transposed_params=(),
                 compass_families=()):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)
        # Layout metadata comes from model construction, not optimizer state.
        self.transposed_params = set(transposed_params)
        # Optional factors only: retain Muon's parameter order, buckets and state.
        self.compass_families = tuple(tuple(family) for family in compass_families)
        self.compass_parameters = {p for family in self.compass_families for p in family}
        if self.compass_families:
            self._require_compass_single_rank()
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError("Muon Compass requires a finite positive base learning rate")
            self.param_groups[0]["initial_lr"] = lr

    @staticmethod
    def _require_compass_single_rank():
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("Muon Compass currently supports one rank; ownership logic is unchanged")

    def _apply_compass(self, parameters, updates, gradients, group):
        """Scale only expert updates, preserving the baseline update dtype."""
        ratio = group["lr"] / group["initial_lr"]
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("Muon Compass cooldown multiplier must be in [0, 1]")
        indices = {p: i for i, p in enumerate(parameters)}
        for family in self.compass_families:
            if family[0] not in indices:
                continue
            if not all(p in indices for p in family):
                raise ValueError("a Compass family must share a Muon shape/dtype/device bucket")
            directions, raw = [], []
            for p in family:
                i = indices[p]
                g = gradients[i].mT.contiguous() if p in self.transposed_params else gradients[i]
                directions.extend(updates[i].unbind() if p.ndim == 3 else [updates[i]])
                raw.extend(g.unbind() if p.ndim == 3 else [g])
            factors = _compass_factors([u.float() for u in directions], raw, ratio)
            for update, (gate, radius) in zip(directions, factors):
                # FP32 factor arithmetic, then baseline BF16 update storage.
                # Unit factors leave every stored update bit unchanged.
                update.copy_((update.float() * radius * gate).to(update.dtype))

    @torch.no_grad()
    def step(self):
        if self.compass_families:
            self._require_compass_single_rank()
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
                # muon_update may mutate its gradient input. Capture raw evidence
                # first, only for experts, and release it after this shape bucket.
                raw_gradients = ([p.grad.detach().float().clone() if p in self.compass_parameters
                                  else None for p in same_shape_params]
                                 if self.compass_families else None)
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
                    for index, (p, momentum) in enumerate(zip(same_shape_params, momentums)):
                        transposed = p in self.transposed_params
                        grad = p.grad.mT.contiguous() if transposed else p.grad
                        momentum_batch = momentum.mT.contiguous() if transposed else momentum
                        update = muon_update(grad, momentum_batch, mu=group["mu"])
                        if transposed:
                            momentum.copy_(momentum_batch.mT)
                        if p in self.compass_parameters:
                            self._apply_compass([p], [update], [raw_gradients[index]], group)
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.mT if transposed else update, alpha=-group["lr"])
                    continue

                if len(same_shape_params) == 1:
                    p = same_shape_params[0]
                    update = muon_update(p.grad, momentums[0], mu=group["mu"])
                    if p in self.compass_parameters:
                        self._apply_compass([p], [update], raw_gradients, group)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                    continue

                grad_batch = torch.stack([p.grad for p in same_shape_params])
                momentum_batch = torch.stack(momentums)
                update_batch = muon_update(
                    grad_batch, momentum_batch, mu=group["mu"])
                torch._foreach_copy_(momentums, momentum_batch.unbind())
                if self.compass_families:
                    self._apply_compass(same_shape_params, update_batch.unbind(), raw_gradients, group)
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


# Compass factors adapted from ExpertMuon-Compass, commit f829248,
# optim/literal_reference.py. The baseline Muon update above is unchanged.
# MIT License
# Copyright (c) 2026 ExpertMuon-Compass contributors
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def _compass_factors(directions, gradients, ratio):
    """Family gates and row radii on FP32 baseline Muon updates/gradients."""
    scalars = torch.stack([v for o, g in zip(directions, gradients)
                           for v in ((o * g).sum(), o.norm(), g.norm())]).tolist()
    raw = [max(0.0, scalars[i] / (scalars[i + 1] * scalars[i + 2] + 1e-12)) + 1e-6
           for i in range(0, len(scalars), 3)]
    mean = sum(raw) / len(raw)
    capped = [min(1.0, a / mean) for a in raw]
    capped_mean = sum(capped) / len(capped)
    factors = []
    for o, g, cap in zip(directions, gradients, capped):
        alignment = ((o * g).sum(dim=1) /
                     (o.norm(dim=1) * g.norm(dim=1) + 1e-12)).clamp_min(0.0)
        rows = alignment + 1e-6
        rows = (rows / rows.mean()).clamp_max(1.0)
        if ratio == 1.0:
            effective = rows
        elif ratio == 0.0:
            effective = torch.ones_like(rows)
        else:
            effective = torch.lerp(torch.ones_like(rows), rows, float(ratio))
        energy = o.square().sum()
        selective = (o.square() * effective.square().unsqueeze(1)).sum()
        radius = torch.where(energy > 0, (selective / energy.clamp_min(1e-30)).sqrt(),
                             torch.ones_like(energy))
        factors.append((cap / capped_mean, radius))
    return factors


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
    adamw_router_weights = set(moe_router_weights(model)) if router_optimizer == "adamw" else set()
    embed_lr, head_lr, scalar_lr = adamw["group_lrs"]
    packed_experts = [module for module in model.modules()
                      if isinstance(module, MoE) and module.moe_parameter_layout == "packed"]
    packed_biases = {p for module in packed_experts
                     for p in (module.fc_bias, module.proj_bias)}
    packed_weights = {p for module in packed_experts
                      for p in (module.fc_weight, module.proj_weight)}
    if "expert_muon" in config:
        raise ValueError("reference ExpertMuon was removed; use optimizers.muon.compass = true")
    compass = muon.get("compass", False)
    if not isinstance(compass, bool):
        raise ValueError("optimizers.muon.compass must be a boolean")
    families = []
    if compass:
        for module in model.modules():
            if not isinstance(module, MoE):
                continue
            if module.moe_parameter_layout == "packed":
                families.extend([p] for p in (module.fc_weight, module.proj_weight))
            else:
                families.extend([getattr(expert, role).weight for expert in module.experts]
                                for role in ("fc", "proj"))
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=embed_lr),
                        dict(params=[model.proj.weight], lr=head_lr),
                        dict(params=[p for p in model.parameters()
                                     if p.ndim < 2 or p in packed_biases
                                     or p in adamw_router_weights], lr=scalar_lr)],
                       betas=tuple(adamw["betas"]), eps=adamw["eps"],
                       weight_decay=adamw["weight_decay"], fused=adamw["fused"])
    optimizer2 = Muon([p for p in model.blocks.parameters()
                       if p.ndim >= 2 and p not in packed_biases
                       and p not in adamw_router_weights],
                      lr=muon["lr"], weight_decay=muon["weight_decay"], mu=muon["mu"],
                      transposed_params=packed_weights, compass_families=families)
    optimizers = [optimizer1, optimizer2]
    assigned = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
    assert len(assigned) == len(set(assigned)), "optimizer parameter ownership must be exclusive"
    assert set(assigned) == set(model.parameters()), "optimizers must cover every model parameter"
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers
