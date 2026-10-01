"""CPU contracts for optional router-weighted Grad-EM score normalization."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from modded_nanogpt_moe.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    _with_legacy_defaults,
    validate_checkpoint_config,
)
from modded_nanogpt_moe.config import (
    load_experiment_config,
    validate_experiment_config,
)
from modded_nanogpt_moe.grad_em import (
    GradEMCombine,
    grad_em_reference,
    normalize_grad_em_scores,
)
from modded_nanogpt_moe.model import GPT, MoE, combine_expert_outputs


def _responsibilities(base_probs, normalized_scores, beta=0.3):
    return torch.softmax(base_probs.log() - beta * normalized_scores, dim=-1)


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("spread", [1e-9, 0.2, 4.0])
def test_adaptive_temperature_contract_and_previous_q(k, spread):
    torch.manual_seed(910 + k)
    logits = torch.randn(7, k) * 2
    scores = torch.randn(7, k) * spread
    indices = torch.arange(k).expand(7, k)
    eta, eps = 0.3, 1e-6
    result = grad_em_reference(logits, indices, scores[..., None],
                               torch.ones(7, 1), eta, "std", eps)
    normalized, scale = normalize_grad_em_scores(scores, result.a, "std", eps,
                                                 return_scale=True)
    previous_q = (logits - eta * normalized).softmax(-1)
    torch.testing.assert_close(result.q, previous_q, rtol=0, atol=0)
    expected_q = (logits - (eta / scale) * scores).softmax(-1)
    torch.testing.assert_close(result.q, expected_q, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(result.grad_logits,
                               (scale / eta) * (result.a - result.q), rtol=0, atol=0)
    assert torch.isfinite(result.grad_logits).all()
    assert not scale.requires_grad and not result.q.requires_grad


@pytest.mark.parametrize("k", [2, 4, 8])
def test_affine_scores_preserve_q_and_rescale_router_signal(k):
    torch.manual_seed(221 + k)
    logits, scores = torch.randn(9, k), torch.randn(9, k)
    indices = torch.arange(k).expand(9, k)
    def run(values):
        return grad_em_reference(logits, indices, values[..., None],
                                 torch.ones(9, 1), 0.4, "std")
    original, transformed = run(scores), run(3.75 * scores - 8.5)
    torch.testing.assert_close(transformed.q, original.q, rtol=3e-6, atol=5e-7)
    torch.testing.assert_close(transformed.grad_logits, 3.75 * original.grad_logits,
                               rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("k", [2, 4, 8])
@pytest.mark.parametrize("spread", [0.2, 4.0])
def test_small_eta_recovers_raw_bp_scale(k, spread):
    torch.manual_seed(326 + k)
    logits, scores = torch.randn(11, k), spread * torch.randn(11, k)
    indices = torch.arange(k).expand(11, k)
    p = logits.softmax(-1)
    bp = p * (scores - (p * scores).sum(-1, keepdim=True))
    errors = []
    for eta in (0.1, 0.001):
        result = grad_em_reference(logits, indices, scores[..., None],
                                   torch.ones(11, 1), eta, "std")
        errors.append((result.grad_logits - bp).norm() / bp.norm())
    assert errors[1] < errors[0]
    # FP32 subtraction contributes O(machine-epsilon/eta), alongside O(eta)
    # truncation. This checks the raw BP scale, not the old 1/std amplification.
    assert errors[1] < 0.002


def test_softmin_potential_gradient_with_detached_scale():
    torch.manual_seed(12)
    logits = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    scores = torch.randn(5, 4, dtype=torch.float64)
    p = logits.softmax(-1)
    centered = scores - (p * scores).sum(-1, keepdim=True)
    scale = (p * centered.square()).sum(-1, keepdim=True).sqrt().clamp_min(1e-6).detach()
    eta = 0.3
    tilted = logits - eta / scale * scores
    potential = -(scale.squeeze(-1) / eta) * (
        tilted.logsumexp(-1) - logits.logsumexp(-1))
    actual, = torch.autograd.grad(potential.sum(), logits)
    expected = scale / eta * (p - tilted.softmax(-1))
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("eps", [1e-12, 1e-6, 10.0])
def test_none_is_exactly_the_legacy_responsibility_and_gradient_rule(eps):
    generator = torch.Generator().manual_seed(91)
    logits = torch.randn(5, 11, generator=generator)
    indices = logits.topk(4, dim=-1).indices
    outputs = torch.randn(5, 4, 7, generator=generator)
    incoming = torch.randn(5, 7, generator=generator)

    default = grad_em_reference(logits, indices, outputs, incoming, 0.2)
    explicit = grad_em_reference(
        logits, indices, outputs, incoming, 0.2, "none", eps)
    for actual, expected in zip(explicit, default):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_normalized_scores_have_weighted_moments_and_affine_invariance(k):
    generator = torch.Generator().manual_seed(100 + k)
    logits = 2 * torch.randn(6, k, generator=generator)
    base = logits.softmax(dim=-1)  # Deliberately uneven probabilities.
    scores = torch.randn(6, k, generator=generator)

    normalized = normalize_grad_em_scores(scores, base, "std")
    transformed = normalize_grad_em_scores(3.75 * scores - 8.5, base, "std")
    mean = (base * normalized).sum(dim=-1)
    variance = (base * normalized.square()).sum(dim=-1)

    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=2e-6, rtol=0)
    torch.testing.assert_close(variance, torch.ones_like(variance), atol=3e-6, rtol=0)
    torch.testing.assert_close(transformed, normalized, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(
        _responsibilities(base, transformed),
        _responsibilities(base, normalized), atol=5e-7, rtol=2e-6)


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_degenerate_scores_return_base_responsibilities(k):
    generator = torch.Generator().manual_seed(73)
    logits = 2 * torch.randn(31, k, generator=generator)
    base = logits.softmax(dim=-1)
    scores = torch.linspace(-1e4, 1e4, 31).unsqueeze(-1).expand_as(base)
    normalized = normalize_grad_em_scores(scores, base, "std")

    assert torch.equal(normalized, torch.zeros_like(normalized))
    torch.testing.assert_close(
        torch.softmax(logits - 0.3 * normalized, dim=-1), base,
        rtol=0, atol=0)

    indices = torch.arange(k).view(1, k).repeat(31, 1)
    outputs = torch.ones(31, k, 5)
    incoming = torch.ones(31, 5)
    result = grad_em_reference(
        logits, indices, outputs, incoming, 0.3, "std")
    torch.testing.assert_close(result.q, result.a, rtol=0, atol=0)
    assert torch.count_nonzero(result.grad_logits) == 0


@pytest.mark.parametrize("k", [2, 4, 8])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_small_nonzero_spread_uses_denominator_floor_not_zero_cutoff(k, eps):
    scores = torch.linspace(-1e-10, 1e-10, k).unsqueeze(0)
    base = torch.linspace(-2, 2, k).softmax(-1).unsqueeze(0)
    normalized = normalize_grad_em_scores(scores, base, "std", eps)
    centered = scores.double() - (base.double() * scores.double()).sum(-1, keepdim=True)
    torch.testing.assert_close(normalized, (centered / eps).float(), atol=1e-10, rtol=2e-6)
    assert torch.isfinite(normalized).all()
    assert torch.count_nonzero(normalized) > 0
    assert normalized.abs().max() <= (scores.max() - scores.min()) / eps
    # In the floor regime a positive scale changes z (intentionally).
    doubled = normalize_grad_em_scores(2 * scores, base, "std", eps)
    torch.testing.assert_close(doubled, 2 * normalized, atol=0, rtol=0)


def test_unnormalized_topk_forward_weights_remain_distinct_from_base_probs():
    torch.manual_seed(123)
    n, experts, k, width = 5, 9, 4, 6
    logits = torch.randn(n, experts, requires_grad=True)
    full_probs = logits.float().softmax(dim=-1)
    weights, indices = full_probs.topk(k, dim=-1)
    assert torch.all(weights.sum(dim=-1) < 1)
    weights.retain_grad()
    order = indices.flatten().argsort(stable=True)
    sorted_outputs = torch.randn(n * k, width, requires_grad=True)
    incoming = torch.randn(n, width)

    output = GradEMCombine.apply(
        sorted_outputs, logits, weights, indices, order, 0.3, None, 1.0,
        "std", 2.0)
    assert torch.equal(
        output, combine_expert_outputs(sorted_outputs, weights, order))
    output.backward(incoming)

    selected = torch.empty_like(sorted_outputs)
    selected[order] = sorted_outputs.detach()
    expected = grad_em_reference(
        logits.detach(), indices, selected.view(n, k, width), incoming,
        0.3, "std", 2.0)
    base = logits.detach().gather(1, indices).softmax(dim=-1)
    assert not torch.allclose(base, weights.detach())
    torch.testing.assert_close(
        sorted_outputs.grad,
        expected.grad_expert.flatten(0, 1)[order], rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, expected.grad_logits, rtol=0, atol=0)
    assert weights.grad is None


def _run(model, value, incoming):
    value = value.detach().clone().requires_grad_()
    output = model(value)
    gradients = torch.autograd.grad(
        output, (value, *model.parameters()), incoming, allow_unused=True)
    return output, gradients


@pytest.mark.parametrize("mode,backend", [("global", "grouped_gemm"),
                                            ("local_bp", "loop")])
def test_normalized_mode_runs_and_lambda_zero_is_exact_bp(monkeypatch, mode, backend):
    if backend == "grouped_gemm":
        def gmm(x, weights, counts, trans_b=False):
            assert trans_b is False
            return torch.cat([
                segment @ weight
                for segment, weight in zip(x.split(counts.tolist()), weights)
            ])
        monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "extension")
        monkeypatch.setitem(
            sys.modules, "grouped_gemm",
            SimpleNamespace(ops=SimpleNamespace(gmm=gmm)))
    torch.manual_seed(211)
    standard = MoE(8, 4, 2, hidden_dim=12, moe_backend=backend)
    normalized = copy.deepcopy(standard)
    normalized.moe_backward = "grad_em"
    normalized.grad_em_mode = mode
    normalized.grad_em_score_normalization = "std"
    normalized.grad_em_score_norm_eps = 0.5
    value = torch.randn(2, 5, 8)
    incoming = torch.randn_like(value)

    normalized_result = _run(normalized, value, incoming)
    assert all(torch.isfinite(tensor).all() for tensor in normalized_result[0:1])
    assert all(gradient is None or torch.isfinite(gradient).all()
               for gradient in normalized_result[1])

    bp_result = _run(standard, value, incoming)
    zero = copy.deepcopy(normalized)
    zero.grad_em_lambda = 0
    zero_result = _run(zero, value, incoming)
    torch.testing.assert_close(zero_result[0], bp_result[0], rtol=0, atol=0)
    for actual, expected in zip(zero_result[1], bp_result[1]):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_config_validation_and_legacy_checkpoint_default():
    config = load_experiment_config()
    assert config["model"]["grad_em_score_normalization"] == "none"
    assert config["model"]["grad_em_score_norm_eps"] == 1e-6
    config["model"]["grad_em_score_normalization"] = "std"
    validate_experiment_config(config)
    config["model"]["grad_em_score_normalization"] = "invalid"
    with pytest.raises(ValueError, match="grad_em_score_normalization"):
        validate_experiment_config(config)
    with pytest.raises(ValueError, match="grad_em_score_normalization"):
        MoE(8, 4, 2, grad_em_score_normalization="invalid")

    legacy = {"model": {"mlp_type": "moe"}, "training": {"total_steps": 20}}
    current = copy.deepcopy(legacy)
    current["model"]["grad_em_score_normalization"] = "none"
    current["model"]["grad_em_score_norm_eps"] = 1e-6
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "resolved_config": legacy,
    }
    validate_checkpoint_config(checkpoint, current)
    assert _with_legacy_defaults(legacy)["model"]["grad_em_score_norm_eps"] == 1e-6
    changed_eps = copy.deepcopy(current)
    changed_eps["model"]["grad_em_score_norm_eps"] = 1e-3
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_config(checkpoint, changed_eps)
    changed = copy.deepcopy(current)
    changed["model"]["grad_em_score_normalization"] = "std"
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_config(checkpoint, changed)
    assert "grad_em_score_normalization" not in legacy["model"]
    assert "grad_em_score_norm_eps" not in legacy["model"]


@pytest.mark.parametrize("eps", [0, -1, float("nan"), float("inf"), -float("inf"), True, "1e-6", None])
def test_invalid_epsilon_fails_config_and_constructor(eps):
    config = load_experiment_config()
    config["model"]["grad_em_score_norm_eps"] = eps
    with pytest.raises(ValueError, match="grad_em_score_norm_eps.*finite and positive"):
        validate_experiment_config(config)
    with pytest.raises(ValueError, match="grad_em_score_norm_eps"):
        MoE(8, 4, 2, grad_em_score_norm_eps=eps)


def test_epsilon_reaches_each_model_layer_without_changing_initialization():
    torch.manual_seed(981)
    baseline = GPT(32, 2, 128, mlp_type="moe", num_experts=4, top_k=2)
    torch.manual_seed(981)
    normalized = GPT(32, 2, 128, mlp_type="moe", num_experts=4, top_k=2,
                     moe_backward="grad_em", grad_em_mode="local_bp",
                     grad_em_score_normalization="std", grad_em_score_norm_eps=0.2)
    for block in normalized.blocks:
        assert block.mlp.grad_em_score_norm_eps == 0.2
    for name, value in normalized.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[name], atol=0, rtol=0)
