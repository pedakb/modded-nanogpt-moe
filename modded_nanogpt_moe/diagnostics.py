"""Sampled, rank-zero TensorBoard diagnostics; no optimizer or model mutations.

Routing/loss are rank-local; gradients are observed after the trainer's existing
SUM all-reduce. State is transient and never included in checkpoints. See
docs/diagnostics.md for denominators, undefined ratios, and memory costs.
"""
from contextlib import contextmanager
import math

import torch


def _compact_group(name):
    parts = {"experts": "expert", "attention": "attn", "embedding": "embed"}
    return "/".join("l" + part[6:] if part.startswith("layer_") else parts.get(part, part)
                    for part in name.split("/"))


_ROUTER_TAG_NAMES = {
    "normalized_entropy": "entropy_norm",
    "min_load_fraction": "load/min",
    "max_load_fraction": "load/max",
    "mean_load_fraction": "load/mean",
    "std_load_fraction": "load/std",
    "load_cv": "load/cv",
    "load_entropy": "load/entropy",
    "normalized_load_entropy": "load/entropy_norm",
    "zero_experts": "load/zero",
    "max_over_mean_load": "load/max_mean",
    "topk_margin_median": "topk_margin_med",
    "topk_margin_below_0.01": "margin/lt_001",
    "topk_margin_below_0.05": "margin/lt_005",
    "topk_margin_below_0.1": "margin/lt_01",
}


def ratio(numerator, denominator):
    # Zero-initialized projections are common. Do not hide undefined ratios with
    # a small epsilon or infinity; TensorBoard receives NaN until well-defined.
    return torch.where(denominator > 0, numerator / denominator, float("nan"))


def midpoint_median(values):
    """Selection-based even/odd median; scalar quantile validation is unnecessary."""
    lower = values.kthvalue((values.numel() + 1) // 2).values
    upper = values.kthvalue(values.numel() // 2 + 1).values
    median = (lower + upper) * 0.5
    return torch.where(values.isnan().any(), float("nan"), median)


def summary(values):
    return dict(min=values.min(), median=midpoint_median(values), mean=values.mean(),
                max=values.max(), std=values.std(correction=0))


def percentile(values, fraction):
    """Linear-interpolated percentile using selection, with no host scalar read."""
    position = (values.numel() - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    lo = values.kthvalue(lower + 1).values
    hi = values.kthvalue(upper + 1).values
    value = lo + (hi - lo) * (position - lower)
    return torch.where(values.isnan().any(), float("nan"), value)


class RoutingStatistics:
    def __init__(self, experts, top_k):
        self.experts, self.top_k = experts, top_k
        self.tokens = 0
        self.sums = None
        self.counts = None
        self.margins = []
        self.logit_grad_squares = None
        self.logit_grad_elements = 0
        self.gradient_handles = []

    def attach_logit_gradient(self, module, inputs, logits):
        # Module forward hook exists only inside sampled capture. The tensor
        # hook closes over this accumulator, never over logits/the graph.
        if logits.requires_grad:
            self.gradient_handles.append(logits.register_hook(self.observe_logit_gradient))

    @torch.no_grad()
    def observe_logit_gradient(self, grad):
        squares = torch.linalg.vector_norm(grad.detach(), dtype=torch.float32).square()
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
        logits, probabilities, selected = logits.detach().float(), probabilities.detach(), selected.detach()
        if not logits.shape[0]:
            return
        self.tokens += logits.shape[0]
        # Only sampled updates: top-k+1, never a full sort. Full softmax comes
        # from the model BEFORE selected-probability renormalization.
        z, indices = logits.float().topk(min(self.experts, max(2, self.top_k + 1)), dim=-1)
        p = probabilities.gather(-1, indices)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(-1)
        gap = p[:, 0] - p[:, 1] if self.experts > 1 else torch.zeros_like(p[:, 0])
        # Population variance is mean((z - mean(z))**2) per token. Aggregate
        # variances, not RMS values, so unequal microbatches are weighted correctly.
        centered_squares = logits.var(dim=-1, correction=0).sum()
        sums = torch.stack((entropy.sum(), p[:, 0].sum(), gap.sum(), centered_squares))
        counts = torch.bincount(selected.reshape(-1), minlength=self.experts)
        self.sums = sums if self.sums is None else self.sums + sums
        self.counts = counts if self.counts is None else self.counts + counts
        if self.top_k < self.experts:
            self.margins.append(z[:, self.top_k - 1] - z[:, self.top_k])

    @torch.no_grad()
    def finish(self):
        if self.tokens == 0:
            return {}, None
        entropy, top1, gap, logit_variance = self.sums / self.tokens
        q = self.counts.float() / (self.tokens * self.top_k)
        load_entropy = -(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum()
        normalizer = math.log(self.experts) if self.experts > 1 else 1.0
        metrics = dict(entropy=entropy, normalized_entropy=entropy / normalizer,
                       top1_prob=top1, max_prob=top1, logit_rms=logit_variance.sqrt(),
                       min_load_fraction=q.min(), max_load_fraction=q.max(),
                       mean_load_fraction=q.mean(), std_load_fraction=q.std(correction=0),
                       load_cv=q.std(correction=0) / q.mean(), load_entropy=load_entropy,
                       normalized_load_entropy=load_entropy / normalizer,
                       zero_experts=(self.counts == 0).sum(), max_over_mean_load=q.max() / q.mean())
        if self.experts > 1:
            metrics["top1_top2_gap"] = gap
        if self.logit_grad_elements:
            metrics["dL_dlogits_rms"] = (self.logit_grad_squares / self.logit_grad_elements).sqrt()
        if self.margins:
            margins = torch.cat(self.margins)
            metrics["topk_margin"] = margins.mean()
            # Exact midpoint median via selection, not a full token sort.
            metrics["topk_margin_median"] = midpoint_median(margins)
            for threshold in (0.01, 0.05, 0.1):
                metrics[f"topk_margin_below_{threshold:g}"] = (margins < threshold).float().mean()
        return metrics, q


def _norm(tensor, packed):
    return torch.linalg.vector_norm(tensor, dim=(-2, -1) if packed else None, dtype=torch.float32)


class ParameterGroup:
    def __init__(self, parameters, experts=False, packed=False):
        self.parameters = list(parameters)
        self.experts, self.packed = experts, packed
        if self.parameters:
            sizes = ([math.prod(self.parameters[0].shape[-2:])] if packed else
                     [p.numel() for p in self.parameters] if experts else
                     [sum(p.numel() for p in self.parameters)])
            # Tiny constants allocated once, not once per metric/update/expert.
            self.rms_divisor = torch.tensor([math.sqrt(size) for size in sizes],
                                           device=self.parameters[0].device, dtype=torch.float32)

    def rms(self, norms):
        return norms / self.rms_divisor

    def norms(self, values):
        norms = [_norm(value, self.packed) for value in values]
        if self.packed:
            return norms[0]
        values = torch.stack(norms)
        return values if self.experts else values.square().sum().sqrt().reshape(1)


class TrainingDiagnostics:
    def __init__(self, model, settings):
        from .model import MoE
        self.settings = settings
        self.groups, self.layers = {}, {}
        attention = []
        for index, block in enumerate(model.blocks):
            layer = f"layer_{index:02d}"
            attention.extend(p for name, p in block.attn.named_parameters() if name.endswith("weight"))
            if not isinstance(block.mlp, MoE):
                continue
            moe = block.mlp
            self.layers[layer] = moe
            self.groups[f"router/{layer}"] = ParameterGroup(moe.router.parameters())
            for fc, module_name in (("fc1", "fc"), ("fc2", "proj")):
                packed = moe.moe_parameter_layout == "packed"
                parameters = ([getattr(moe, f"{module_name}_weight")] if packed else
                              [getattr(expert, module_name).weight for expert in moe.experts])
                self.groups[f"experts/{layer}/{fc}"] = ParameterGroup(parameters, experts=True, packed=packed)
        if attention:
            self.groups["attention"] = ParameterGroup(attention)
        self.groups["embedding"] = ParameterGroup([model.embed.weight])
        self.groups["head"] = ParameterGroup([model.proj.weight])
        self.clear()

    def clear(self):
        self.routing, self.before, self.metrics, self.losses = {}, {}, {}, []

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
                router_handles.append(moe.router.register_forward_hook(stats.attach_logit_gradient))
            yield
        finally:
            for handle in router_handles:
                handle.remove()
            for stats in self.routing.values():
                stats.remove_gradient_hooks()
            for moe in self.layers.values():
                moe._routing_diagnostics = None

    def observe_loss(self, loss):
        self.losses.append(loss.detach())

    @torch.no_grad()
    def before_optimizers(self):
        # Once every sampled UPDATE, after gradient accumulation/all-reduce,
        # before either optimizer (Muon can mutate its incoming gradients).
        for name, group in self.groups.items():
            parameters = [p.detach() for p in group.parameters]
            grads = [p.grad.detach() if p.grad is not None else torch.zeros_like(p)
                     for p in group.parameters]
            param_norm, grad_norm = group.norms(parameters), group.norms(grads)
            self.metrics[name] = dict(param_norm=param_norm, grad_norm=grad_norm,
                                      grad_ratio=ratio(grad_norm, param_norm))
            self.before[name] = [p.clone() for p in parameters]

    @torch.no_grad()
    def after_optimizers(self, writer, update, local_tokens):
        scalars, histograms = {}, {}
        histogram_due = bool(self.settings["histogram_interval"] and
                             update % self.settings["histogram_interval"] == 0)
        for name, group in self.groups.items():
            old_values = self.before.pop(name)
            differences = []
            for old, parameter in zip(old_values, group.parameters):
                # Reuse FP32 snapshot storage. BF16 snapshots promote before
                # subtracting, so small representable updates aren't re-rounded.
                old = old.float()
                old.sub_(parameter.detach())
                differences.append(_norm(old, group.packed))
            if group.packed:
                update_norm = differences[0]
            else:
                update_norm = torch.stack(differences)
                if not group.experts:
                    update_norm = update_norm.square().sum().sqrt().reshape(1)
            metrics = self.metrics[name]
            metrics.update(update_norm=update_norm, update_ratio=ratio(update_norm, metrics["param_norm"]))
            for quantity in ("param", "grad", "update"):
                metrics[f"{quantity}_rms"] = group.rms(metrics[f"{quantity}_norm"])
            for metric, value in metrics.items():
                tag = f"opt/{_compact_group(name)}/{metric}"
                if group.experts:
                    # Keep old series; new RMS metrics need only the percentile
                    # band with median as the typical expert. No duplicate histograms.
                    stats = ({"median": midpoint_median(value)} if metric.endswith("_rms")
                             else summary(value))
                    stats.update(p10=percentile(value, 0.1), p90=percentile(value, 0.9))
                    scalars.update({f"{tag}_{'med' if stat == 'median' else stat}": result
                                    for stat, result in stats.items()})
                    if histogram_due and metric.endswith("_norm"):
                        histograms[tag] = value
                else:
                    scalars[tag] = value.squeeze(0)
            del old_values, old
        for layer in self.layers:
            router = self.metrics[f"router/{layer}"]
            for fc in ("fc1", "fc2"):
                expert = self.metrics[f"experts/{layer}/{fc}"]
                prefix = f"opt/compare/{_compact_group(layer)}/{fc}"
                scalars[f"{prefix}/router_over_expert_grad_norm"] = ratio(router["grad_norm"].squeeze(), midpoint_median(expert["grad_norm"]))
                scalars[f"{prefix}/router_over_expert_update_ratio"] = ratio(router["update_ratio"].squeeze(), midpoint_median(expert["update_ratio"]))
            routing, q = self.routing[layer].finish()
            for key, value in routing.items():
                if key == "max_prob":  # Duplicate of top1_prob; emit only the canonical tag.
                    continue
                if key == "dL_dlogits_rms":
                    tag = f"opt/router/{_compact_group(layer)}/dlogit_rms"
                else:
                    tag = f"router/{_compact_group(layer)}/{_ROUTER_TAG_NAMES.get(key, key)}"
                scalars[tag] = value
            if histogram_due and q is not None:
                histograms[f"router/{_compact_group(layer)}/load/fraction"] = q
        if self.losses:
            scalars["metric/loss/train"] = torch.stack(self.losses).sum() / local_tokens
        # One batched device->host scalar transfer per sampled update. Histogram
        # vectors use a second batched transfer only at their optional cadence.
        values = torch.stack([value.float() for value in scalars.values()]).cpu().tolist()
        for tag, value in zip(scalars, values):
            writer.add_scalar(tag, value, update)
        if histograms:
            sizes = [value.numel() for value in histograms.values()]
            host = torch.cat(list(histograms.values())).cpu()
            for tag, value in zip(histograms, host.split(sizes)):
                writer.add_histogram(tag, value.numpy(), update)
        self.clear()


def make_diagnostics(model, writer, settings, *, benchmark, nsys_profile, rank):
    # Return before walking parameters, attaching callbacks, or doing tensor work.
    if (writer is None or rank != 0 or benchmark or not settings["scalar_interval"]
            or (nsys_profile and not settings["during_nsys"])):
        return None
    return TrainingDiagnostics(model, settings)
