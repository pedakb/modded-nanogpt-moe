"""Compact sampled TensorBoard diagnostics with no model/optimizer mutations.

Routing is rank-local; gradients are observed after the trainer's existing SUM
all-reduce. State is transient and never included in checkpoints. See
docs/diagnostics.md for definitions, scope, and memory costs.
"""
from contextlib import contextmanager
import math

import torch


MONITORED_LAYER_INDICES = (0, 5, 11)


def midpoint_median(values):
    """Selection-based even/odd median without a full sort or host read."""
    lower = values.kthvalue((values.numel() + 1) // 2).values
    upper = values.kthvalue(values.numel() // 2 + 1).values
    median = (lower + upper) * 0.5
    return torch.where(values.isnan().any(), float("nan"), median)


def _square_sum(tensor, dimensions=None):
    """FP32 squared L2 reduction, optionally independently over leading items."""
    return torch.linalg.vector_norm(
        tensor, dim=dimensions, dtype=torch.float32).square()


def _rms(square_sum, elements):
    return (square_sum / elements).sqrt()


def _ratio_from_squares(numerator, denominator):
    # Zero-initialized projections are common. Report an undefined ratio as NaN.
    return torch.where(denominator > 0, (numerator / denominator).sqrt(), float("nan"))


class RoutingStatistics:
    """Accumulate only the four retained routing metrics and logit-gradient RMS."""

    def __init__(self, experts, top_k):
        self.experts, self.top_k = experts, top_k
        self.tokens = 0
        self.entropy_sum = None
        self.counts = None
        self.margins = []
        self.logit_grad_squares = None
        self.logit_grad_elements = 0
        self.gradient_handles = []

    def attach_logit_gradient(self, module, inputs, logits):
        # The hook closes over scalar accumulators, never logits or the graph.
        if logits.requires_grad:
            self.gradient_handles.append(logits.register_hook(self.observe_logit_gradient))

    @torch.no_grad()
    def observe_logit_gradient(self, grad):
        squares = _square_sum(grad.detach())
        self.logit_grad_squares = (squares if self.logit_grad_squares is None
                                   else self.logit_grad_squares + squares)
        self.logit_grad_elements += grad.numel()
        # Returning None leaves the incoming gradient untouched.

    def remove_gradient_hooks(self):
        for handle in self.gradient_handles:
            handle.remove()
        self.gradient_handles.clear()

    @torch.no_grad()
    def observe(self, logits, probabilities, selected):
        logits, probabilities, selected = logits.detach(), probabilities.detach(), selected.detach()
        if not logits.shape[0]:
            return
        self.tokens += logits.shape[0]
        entropy = -(probabilities * probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny).log()).sum(-1).sum()
        self.entropy_sum = entropy if self.entropy_sum is None else self.entropy_sum + entropy
        counts = torch.bincount(selected.reshape(-1), minlength=self.experts)
        self.counts = counts if self.counts is None else self.counts + counts
        if self.top_k < self.experts:
            # Only sampled monitored layers: top-k+1, never a full expert sort.
            boundary = logits.float().topk(self.top_k + 1, dim=-1).values
            self.margins.append(boundary[:, self.top_k - 1] - boundary[:, self.top_k])

    @torch.no_grad()
    def finish(self):
        if self.tokens == 0:
            return {}
        normalizer = math.log(self.experts) if self.experts > 1 else 1.0
        q = self.counts.float() / (self.tokens * self.top_k)
        metrics = {
            "normalized_entropy": self.entropy_sum / self.tokens / normalizer,
            "load_cv": q.std(correction=0) / q.mean(),
            "zero_experts": (self.counts == 0).sum(),
        }
        if self.margins:
            metrics["topk_margin_median"] = midpoint_median(torch.cat(self.margins))
        if self.logit_grad_elements:
            metrics["dL_dlogits_rms"] = _rms(
                self.logit_grad_squares, self.logit_grad_elements)
        return metrics


class ParameterSet:
    """A small monitored set reduced as one parameter group."""

    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.numel = sum(parameter.numel() for parameter in self.parameters)

    def squares(self, per_parameter):
        values = [per_parameter[id(parameter)] for parameter in self.parameters
                  if id(parameter) in per_parameter]
        if values:
            return torch.stack(values).sum()
        return self.parameters[0].new_zeros((), dtype=torch.float32)


class ExpertSet:
    """FC1+FC2 weight reductions, combined independently for every expert."""

    def __init__(self, moe):
        self.packed = moe.moe_parameter_layout == "packed"
        if self.packed:
            self.parameters = [moe.fc_weight, moe.proj_weight]
            elements = moe.fc_weight[0].numel() + moe.proj_weight[0].numel()
            self.numel = torch.full(
                (moe.num_experts,), elements, device=moe.fc_weight.device,
                dtype=torch.float32)
        else:
            self.by_expert = [
                [expert.fc.weight, expert.proj.weight] for expert in moe.experts]
            self.parameters = [parameter for pair in self.by_expert for parameter in pair]
            self.numel = torch.tensor(
                [sum(parameter.numel() for parameter in pair) for pair in self.by_expert],
                device=self.parameters[0].device, dtype=torch.float32)

    def squares(self, per_parameter):
        if self.packed:
            values = [per_parameter.get(id(parameter)) for parameter in self.parameters]
            values = [value for value in values if value is not None]
            return (torch.stack(values).sum(0) if values
                    else self.numel.new_zeros(self.numel.shape))
        result = []
        for pair in self.by_expert:
            values = [per_parameter[id(parameter)] for parameter in pair
                      if id(parameter) in per_parameter]
            result.append(torch.stack(values).sum() if values else self.numel.new_zeros(()))
        return torch.stack(result)


class TrainingDiagnostics:
    def __init__(self, model, settings):
        from .model import MoE
        self.settings = settings
        self.parameters = [parameter for parameter in model.parameters()
                           if parameter.requires_grad]
        self.total_numel = sum(parameter.numel() for parameter in self.parameters)
        self.layers, self.routers, self.experts = {}, {}, {}
        self.reduction_dimensions = {}
        for index in MONITORED_LAYER_INDICES:
            if index >= len(model.blocks) or not isinstance(model.blocks[index].mlp, MoE):
                continue
            layer = f"l{index:02d}"
            moe = model.blocks[index].mlp
            self.layers[layer] = moe
            self.routers[layer] = ParameterSet(moe.router.parameters())
            self.experts[layer] = ExpertSet(moe)
            for parameter in self.routers[layer].parameters:
                self.reduction_dimensions[id(parameter)] = None
            for parameter in self.experts[layer].parameters:
                self.reduction_dimensions[id(parameter)] = (
                    (-2, -1) if self.experts[layer].packed else None)
        self.clear()

    def clear(self):
        self.routing, self.before, self.metrics = {}, [], {}

    def due(self, update):
        return update % self.settings["scalar_interval"] == 0

    @contextmanager
    def capture_routing(self):
        self.clear()
        router_handles = []
        try:
            for layer, moe in self.layers.items():
                stats = RoutingStatistics(moe.num_experts, moe.top_k)
                self.routing[layer] = stats
                moe._routing_diagnostics = stats.observe
                router_handles.append(moe.router.register_forward_hook(
                    stats.attach_logit_gradient))
            yield
        finally:
            for handle in router_handles:
                handle.remove()
            for stats in self.routing.values():
                stats.remove_gradient_hooks()
            for moe in self.layers.values():
                moe._routing_diagnostics = None

    def _collect_squares(self, getter):
        total = None
        monitored = {}
        for parameter in self.parameters:
            value = getter(parameter)
            if value is None:
                continue
            dimensions = self.reduction_dimensions.get(id(parameter))
            detail = _square_sum(value, dimensions)
            scalar = detail.sum() if detail.ndim else detail
            total = scalar if total is None else total + scalar
            if id(parameter) in self.reduction_dimensions:
                monitored[id(parameter)] = detail
        if total is None:
            total = self.parameters[0].new_zeros((), dtype=torch.float32)
        return total, monitored

    @torch.no_grad()
    def before_optimizers(self):
        param_total, param_monitored = self._collect_squares(lambda p: p.detach())
        grad_total, grad_monitored = self._collect_squares(
            lambda p: p.grad.detach() if p.grad is not None else None)
        self.metrics["global"] = (param_total, grad_total)
        for layer in self.layers:
            self.metrics[f"router/{layer}"] = (
                self.routers[layer].squares(param_monitored),
                self.routers[layer].squares(grad_monitored),
            )
            self.metrics[f"expert/{layer}"] = (
                self.experts[layer].squares(param_monitored),
                self.experts[layer].squares(grad_monitored),
            )
        # One snapshot per optimized parameter. Monitored groups reuse these
        # snapshots rather than cloning their parameters a second time.
        self.before = [parameter.detach().clone() for parameter in self.parameters]

    @torch.no_grad()
    def _update_squares(self):
        total = None
        monitored = {}
        for parameter, old in zip(self.parameters, self.before):
            old = old.float()
            old.sub_(parameter.detach())
            dimensions = self.reduction_dimensions.get(id(parameter))
            detail = _square_sum(old, dimensions)
            scalar = detail.sum() if detail.ndim else detail
            total = scalar if total is None else total + scalar
            if id(parameter) in self.reduction_dimensions:
                monitored[id(parameter)] = detail
        self.before.clear()
        return total, monitored

    @torch.no_grad()
    def after_optimizers(self, writer, update, extra_scalars=None):
        update_total, update_monitored = self._update_squares()
        param_total, grad_total = self.metrics["global"]
        scalars = dict(extra_scalars or {})
        scalars.update({
            "opt/global/param_rms": _rms(param_total, self.total_numel),
            "opt/global/grad_rms": _rms(grad_total, self.total_numel),
            "opt/global/update_rms": _rms(update_total, self.total_numel),
            "opt/global/update_ratio": _ratio_from_squares(update_total, param_total),
        })
        for layer in self.layers:
            param_square, grad_square = self.metrics[f"router/{layer}"]
            update_square = self.routers[layer].squares(update_monitored)
            prefix = f"opt/router/{layer}"
            scalars.update({
                f"{prefix}/param_rms": _rms(param_square, self.routers[layer].numel),
                f"{prefix}/grad_rms": _rms(grad_square, self.routers[layer].numel),
                f"{prefix}/update_rms": _rms(update_square, self.routers[layer].numel),
                f"{prefix}/update_ratio": _ratio_from_squares(update_square, param_square),
            })
            param_square, grad_square = self.metrics[f"expert/{layer}"]
            update_square = self.experts[layer].squares(update_monitored)
            prefix = f"opt/expert/{layer}"
            for name, value in (
                    ("param_rms", _rms(param_square, self.experts[layer].numel)),
                    ("grad_rms", _rms(grad_square, self.experts[layer].numel)),
                    ("update_rms", _rms(update_square, self.experts[layer].numel)),
                    ("update_ratio", _ratio_from_squares(update_square, param_square))):
                scalars[f"{prefix}/{name}_med"] = midpoint_median(value)
            routing = self.routing[layer].finish()
            for old, new in (
                    ("normalized_entropy", "entropy_norm"),
                    ("topk_margin_median", "topk_margin_med"),
                    ("load_cv", "load/cv"),
                    ("zero_experts", "load/zero")):
                if old in routing:
                    scalars[f"router/{layer}/{new}"] = routing[old]
            if "dL_dlogits_rms" in routing:
                scalars[f"opt/router/{layer}/dlogit_rms"] = routing["dL_dlogits_rms"]
        # One batched device-to-host transfer, including the ordinary training
        # loss on sampled steps when the trainer supplies it.
        values = torch.stack([value.float() for value in scalars.values()]).cpu().tolist()
        for tag, value in zip(scalars, values):
            writer.add_scalar(tag, value, update)
        self.clear()


def make_diagnostics(model, writer, settings, *, benchmark, nsys_profile, rank):
    # Return before inspecting parameters, attaching callbacks, or doing tensor work.
    if (writer is None or rank != 0 or benchmark or not settings["scalar_interval"]
            or (nsys_profile and not settings["during_nsys"])):
        return None
    return TrainingDiagnostics(model, settings)
