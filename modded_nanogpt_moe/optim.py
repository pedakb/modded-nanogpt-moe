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


# Compass arithmetic adapted from ExpertMuon-Compass, commit f829248,
# optim/literal_reference.py and optim/golden.py.
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


def _expert_muon_direction(source):
    """Original Compass five-step FP32 map (not this project's Muon map)."""
    if not torch.isfinite(source).all():
        raise FloatingPointError("Nonfinite ExpertMuon matrix direction")
    x = source / (source.norm() + 1e-7)
    transposed = source.size(0) > source.size(1)
    if transposed:
        x = x.mT
    for _ in range(5):
        gram = x @ x.mT
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
    return x.mT if transposed else x


class ExpertMuon(torch.optim.Optimizer):
    """Original Compass on expert families; each group is one layer/projection.

    A group has either ordered [out,in] parameters or one packed parameter with
    transposed=True for this project's [E,in,out] storage. No expert flattening.
    Scalar reductions intentionally follow the literal reference's host order.
    """
    def __init__(self, families, lr=0.0005, momentum=0.95, weight_decay=0.01):
        from .config import validate_expert_muon_config
        validate_expert_muon_config(lr, momentum, weight_decay)
        self._require_single_rank()
        super().__init__(families, dict(lr=lr, initial_lr=lr, momentum=momentum,
                                       weight_decay=weight_decay, transposed=False))

    @staticmethod
    def _require_single_rank():
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("ExpertMuon currently supports one rank; distributed ownership is unchanged")

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict would otherwise round FP32 state to BF16
        # parameter dtype. Preserve original momentum values before that cast.
        momentums = [state_dict["state"].get(index, {}).get("momentum_buffer")
                     for group in state_dict["param_groups"] for index in group["params"]]
        super().load_state_dict(state_dict)
        parameters = [p for group in self.param_groups for p in group["params"]]
        for p, momentum in zip(parameters, momentums):
            if momentum is not None:
                self.state[p]["momentum_buffer"] = momentum.to(device=p.device, dtype=torch.float32).clone()

    @torch.no_grad()
    def step(self):
        self._require_single_rank()
        for group in self.param_groups:
            lr = group["lr"]
            ratio = lr / group["initial_lr"]
            if not math.isfinite(ratio) or not 0 <= ratio <= 1:
                raise ValueError("ExpertMuon cooldown multiplier must be finite and in [0, 1]")
            pending = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim not in (2, 3):
                    raise ValueError("ExpertMuon requires 2D matrices or 3D packed experts")
                gradient = p.grad.detach().float()
                if group["transposed"]:
                    gradient = gradient.mT.contiguous()
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite ExpertMuon gradient")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient, dtype=torch.float32)
                momentum = state["momentum_buffer"]
                if momentum.dtype != torch.float32:
                    raise TypeError("ExpertMuon momentum must remain FP32")
                momentum.mul_(group["momentum"]).add_(gradient)
                source = gradient.add(momentum, alpha=group["momentum"])
                weight = p.mT if group["transposed"] else p
                matrices = [(weight, gradient, source)] if p.ndim == 2 else zip(weight, gradient, source)
                for w, g, value in matrices:
                    pending.append((w, g, _expert_muon_direction(value)))
            if not pending:
                continue
            scalars = torch.stack([v for _, g, o in pending
                                   for v in ((o * g).sum(), o.norm(), g.norm())]).tolist()
            raw = [max(0.0, scalars[i] / (scalars[i + 1] * scalars[i + 2] + 1e-12)) + 1e-6
                   for i in range(0, len(scalars), 3)]
            mean = sum(raw) / len(raw)
            capped = [min(1.0, a / mean) for a in raw]
            capped_mean = sum(capped) / len(capped)
            gates = [a / capped_mean for a in capped]
            for (w, g, o), gate in zip(pending, gates):
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
                update = o * radius
                scale = 0.2 * math.sqrt(max(w.shape))
                w.mul_(1.0 - lr * group["weight_decay"])
                w.add_(update.to(w.dtype), alpha=-(lr * gate) * scale)


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
    expert_optimizer = None
    expert_weights = set()
    if "expert_muon" in config:
        families = []
        for module in model.modules():
            if not isinstance(module, MoE):
                continue
            if module.moe_parameter_layout == "packed":
                families.extend(dict(params=[p], transposed=True)
                                for p in (module.fc_weight, module.proj_weight))
            else:
                families.extend(dict(params=[getattr(expert, role).weight for expert in module.experts])
                                for role in ("fc", "proj"))
        if families:
            expert_optimizer = ExpertMuon(families, **config["expert_muon"])
            expert_weights = {p for family in families for p in family["params"]}
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=embed_lr),
                        dict(params=[model.proj.weight], lr=head_lr),
                        dict(params=[p for p in model.parameters()
                                     if p.ndim < 2 or p in packed_biases
                                     or p in adamw_router_weights], lr=scalar_lr)],
                       betas=tuple(adamw["betas"]), eps=adamw["eps"],
                       weight_decay=adamw["weight_decay"], fused=adamw["fused"])
    optimizer2 = Muon([p for p in model.blocks.parameters()
                       if p.ndim >= 2 and p not in packed_biases
                       and p not in adamw_router_weights and p not in expert_weights],
                      lr=muon["lr"], weight_decay=muon["weight_decay"], mu=muon["mu"],
                      transposed_params=packed_weights - expert_weights)
    optimizers = [optimizer1, optimizer2]
    if expert_optimizer is not None:
        optimizers.append(expert_optimizer)
    assigned = [p for opt in optimizers for group in opt.param_groups for p in group["params"]]
    assert len(assigned) == len(set(assigned)), "optimizer parameter ownership must be exclusive"
    assert set(assigned) == set(model.parameters()), "optimizers must cover every model parameter"
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers
