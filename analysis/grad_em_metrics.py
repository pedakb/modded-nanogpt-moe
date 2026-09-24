"""Vectorized CPU metrics for fixed-support Grad-EM diagnostic tensors.

For selected logits ``z`` and task scores ``v`` on a fixed Top-K support,
``a = softmax(z)`` and ``q = softmax(z - eta*v)``. The directional gradients
are ``g_BP = J_a v = a*(v-E_a[v])`` and ``g_GE = (a-q)/eta``. Centered ``v``
is used in curvature dot products so quantities that are mathematically
invariant under ``v -> v+c`` remain numerically stable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _array(value, *, dtype=None) -> np.ndarray:
    """Convert NumPy-like or CPU torch tensors without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _validate_pair(selected_logits, v) -> tuple[np.ndarray, np.ndarray]:
    logits = _array(selected_logits, dtype=np.float64)
    values = _array(v, dtype=np.float64)
    if logits.ndim != 2 or values.shape != logits.shape or logits.shape[1] < 1:
        raise ValueError("selected_logits and v must have equal nonempty [N,K] shape")
    if not np.isfinite(logits).all() or not np.isfinite(values).all():
        raise ValueError("selected_logits and v must be finite")
    return logits, values


def _validate_eta(eta: float) -> float:
    if (isinstance(eta, (bool, np.bool_)) or not np.isscalar(eta)
            or not math.isfinite(float(eta)) or float(eta) <= 0):
        raise ValueError("eta must be finite and strictly positive")
    return float(eta)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def routing_distributions(selected_logits, v, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``a=softmax(z)`` and ``q=softmax(z-eta*v)`` in float64."""
    logits, values = _validate_pair(selected_logits, v)
    eta = _validate_eta(eta)
    return _softmax(logits), _softmax(logits - eta * values)


def routing_gradients(a, q, v, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``g_BP=J_a v`` and detached-E-step ``g_GE=(a-q)/eta``."""
    eta = _validate_eta(eta)
    probabilities = _array(a, dtype=np.float64)
    responsibilities = _array(q, dtype=np.float64)
    values = _array(v, dtype=np.float64)
    if (probabilities.ndim != 2 or responsibilities.shape != probabilities.shape
            or values.shape != probabilities.shape):
        raise ValueError("a, q, and v must have equal [N,K] shape")
    mean_v = np.sum(probabilities * values, axis=-1, keepdims=True)
    return (probabilities * (values - mean_v),
            (probabilities - responsibilities) / eta)


def token_metrics(selected_logits, v, eta: float, *,
                  cbp_epsilon: float = 1e-12) -> dict[str, np.ndarray]:
    """Compute per-token fixed-support metrics.

    ``C_BP=v.T@g_BP=Var_a(v)`` and ``C_GE=v.T@g_GE`` define directional
    amplification ``R=C_GE/C_BP``. R is NaN when ``C_BP <= cbp_epsilon``
    rather than an unstable infinity. ``D=a.T@v-min(v)`` is absolute routing
    regret, ``sigma_v=sqrt(Var_a(v))``, and current-top v gap is
    ``v[argmax(a)]-min(v)``.
    """
    if cbp_epsilon < 0 or not math.isfinite(cbp_epsilon):
        raise ValueError("cbp_epsilon must be finite and nonnegative")
    logits, values = _validate_pair(selected_logits, v)
    eta = _validate_eta(eta)
    a, q = routing_distributions(logits, values, eta)
    g_bp, g_ge = routing_gradients(a, q, values, eta)
    mean_v = np.sum(a * values, axis=-1)
    centered_v = values - mean_v[:, None]
    c_bp = np.sum(a * centered_v**2, axis=-1)
    c_ge = np.sum(centered_v * g_ge, axis=-1)
    ratio = np.full_like(c_bp, np.nan)
    np.divide(c_ge, c_bp, out=ratio, where=c_bp > cbp_epsilon)
    minimum_index = np.argmin(values, axis=-1)
    top_index = np.argmax(a, axis=-1)
    rows = np.arange(values.shape[0])
    log_a = np.zeros_like(a)
    np.log(a, out=log_a, where=a > 0)
    entropy = -np.sum(a * log_a, axis=-1)
    entropy_norm = (entropy / math.log(values.shape[1])
                    if values.shape[1] > 1 else np.zeros_like(entropy))
    minimum_v = values[rows, minimum_index]
    return {
        "a": a,
        "q": q,
        "g_bp": g_bp,
        "g_ge": g_ge,
        "entropy_norm": entropy_norm,
        "selected_logit_std": np.std(logits, axis=-1),
        "delta_v": np.max(values, axis=-1) - minimum_v,
        "eta_delta_v": eta * (np.max(values, axis=-1) - minimum_v),
        "tv": 0.5 * np.sum(np.abs(a - q), axis=-1),
        "c_bp": c_bp,
        "c_ge": c_ge,
        "amplification_ratio": ratio,
        "d": mean_v - minimum_v,
        "sigma_v": np.sqrt(np.maximum(c_bp, 0.0)),
        "argmax_a_ne_argmin_v": top_index != minimum_index,
        "min_v_probability": a[rows, minimum_index],
        "current_top_v_gap": values[rows, top_index] - minimum_v,
    }


def load_cv(topk_experts, num_experts: int) -> tuple[float, int]:
    """Return population load CV and zero-load count over selected assignments."""
    experts = _array(topk_experts)
    if (experts.ndim != 2 or isinstance(num_experts, bool)
            or not isinstance(num_experts, int) or num_experts <= 0):
        raise ValueError("topk_experts must be [N,K] and num_experts must be positive")
    if experts.size and (experts.min() < 0 or experts.max() >= num_experts):
        raise ValueError("topk_experts contains an out-of-range expert index")
    counts = np.bincount(experts.astype(np.int64, copy=False).ravel(),
                         minlength=num_experts).astype(np.float64)
    mean = counts.mean()
    cv = float(counts.std() / mean) if mean > 0 else float("nan")
    return cv, int(np.count_nonzero(counts == 0))


def _quantile_label(quantile: float) -> str:
    return "p" + format(100 * quantile, ".12g").replace(".", "_")


def _finite_quantile(values: np.ndarray, quantile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, quantile)) if finite.size else float("nan")


def summarize_layer(selected_logits, v, topk_experts, *, eta: float,
                    num_experts: int, quantiles: Sequence[float] = (0.99, 0.999),
                    tv_threshold: float = 0.1, cbp_epsilon: float = 1e-12,
                    chunk_size: int = 65536) -> dict[str, float | int]:
    """Summarize one layer without retaining full [N,K] intermediates."""
    logits = _array(selected_logits)
    values = _array(v)
    experts = _array(topk_experts)
    if logits.ndim != 2 or values.shape != logits.shape or experts.shape != logits.shape:
        raise ValueError("selected_logits, v, and topk_experts must have equal [N,K] shape")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    quantiles = tuple(float(quantile) for quantile in quantiles)
    if any(not 0 <= quantile <= 1 for quantile in quantiles):
        raise ValueError("quantiles must lie in [0,1]")

    stored = {name: [] for name in ("eta_delta_v", "tv", "amplification_ratio")}
    totals = {name: 0.0 for name in (
        "entropy_norm", "selected_logit_std", "d", "sigma_v",
        "min_v_probability", "current_top_v_gap")}
    mismatch_count = tv_count = valid_ratio_count = 0
    tokens = logits.shape[0]
    for start in range(0, tokens, chunk_size):
        metrics = token_metrics(
            logits[start:start + chunk_size], values[start:start + chunk_size], eta,
            cbp_epsilon=cbp_epsilon)
        for name in totals:
            totals[name] += float(np.sum(metrics[name], dtype=np.float64))
        for name in stored:
            stored[name].append(metrics[name])
        mismatch_count += int(np.count_nonzero(metrics["argmax_a_ne_argmin_v"]))
        tv_count += int(np.count_nonzero(metrics["tv"] > tv_threshold))
        valid_ratio_count += int(np.count_nonzero(np.isfinite(metrics["amplification_ratio"])))

    concatenated = {
        name: (np.concatenate(parts) if parts else np.empty(0, dtype=np.float64))
        for name, parts in stored.items()
    }
    cv, zero_load = load_cv(experts, num_experts)
    denominator = tokens if tokens else 1
    summary: dict[str, float | int] = {
        "tokens": int(tokens),
        "support_size": int(logits.shape[1]),
        "eta": float(eta),
        "entropy_norm_mean": totals["entropy_norm"] / denominator,
        "selected_logit_std_mean": totals["selected_logit_std"] / denominator,
        "load_cv": cv,
        "load_zero": zero_load,
        "tv_gt_0_1_count": tv_count,
        "tv_gt_0_1_fraction": tv_count / denominator,
        "r_valid_count": valid_ratio_count,
        "d_mean": totals["d"] / denominator,
        "sigma_v_mean": totals["sigma_v"] / denominator,
        "argmax_a_ne_argmin_v_fraction": mismatch_count / denominator,
        "min_v_probability_mean": totals["min_v_probability"] / denominator,
        "current_top_v_gap_mean": totals["current_top_v_gap"] / denominator,
    }
    prefixes = {"eta_delta_v": "eta_dv", "tv": "tv", "amplification_ratio": "r"}
    for source, prefix in prefixes.items():
        for quantile in quantiles:
            summary[f"{prefix}_{_quantile_label(quantile)}"] = _finite_quantile(
                concatenated[source], quantile)
    return summary
