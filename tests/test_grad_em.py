"""CPU value tests for a replacement gradient, not forward gradcheck."""

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from modded_nanogpt_moe.checkpoint import CHECKPOINT_FORMAT_VERSION, validate_checkpoint_config
from modded_nanogpt_moe.config import load_experiment_config, validate_experiment_config
from modded_nanogpt_moe.grad_em import grad_em_reference


def inputs(k=2, dtype=torch.float32):
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(3, 12, generator=generator, dtype=dtype)
    idx = logits.topk(k, dim=-1).indices
    outputs = torch.randn(3, k, 7, generator=generator, dtype=dtype)
    g = torch.randn(3, 7, generator=generator, dtype=dtype)
    return logits, idx, outputs, g


@pytest.mark.parametrize("k", [1, 2, 8, 12])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_contract_and_support(k, dtype):
    z, idx, h, g = inputs(k, dtype)
    result = grad_em_reference(z, idx, h, g, eta=0.2)
    # Independent scalar dot products: products AND accumulation are FP32.
    v = torch.tensor([
        [sum(float(g[t, d]) * float(h[t, j, d]) for d in range(7))
         for j in range(k)] for t in range(3)])
    torch.testing.assert_close(result.v, v, rtol=2e-6, atol=1e-6)
    v_fp32 = (g[:, None, :].float() * h.float()).sum(-1)
    q = torch.softmax(z.float().gather(1, idx) - 0.2 * v_fp32, -1)
    a = torch.softmax(z.float().gather(1, idx), -1)
    expected = (v_fp32, q, a, q[..., None] * g[:, None, :].float(),
                torch.zeros_like(z, dtype=torch.float32).scatter_(1, idx, (a - q) / 0.2))
    for actual, wanted in zip(result, expected):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
    torch.testing.assert_close(result.q.sum(-1), torch.ones(3), rtol=0, atol=2e-7)
    torch.testing.assert_close(result.a.sum(-1), torch.ones(3), rtol=0, atol=2e-7)
    inactive = torch.ones_like(z, dtype=torch.bool).scatter_(1, idx, False)
    assert torch.count_nonzero(result.grad_logits[inactive]) == 0
    full_expert_grads = torch.zeros(3, 12, 7).scatter_(
        1, idx[..., None].expand(-1, -1, 7), result.grad_expert)
    assert torch.count_nonzero(full_expert_grads[inactive]) == 0
    if k == 1:
        assert torch.equal(result.q, torch.ones_like(result.q))
        assert torch.count_nonzero(result.grad_logits) == 0


def test_hand_computed_dot_products():
    result = grad_em_reference(
        torch.tensor([[2., 1., 0.]]), torch.tensor([[0, 1]]),
        torch.tensor([[[1., 2.], [-3., 4.]]]), torch.tensor([[2., -1.]]))
    assert torch.equal(result.v, torch.tensor([[0., -10.]]))
    assert torch.equal(result.q, torch.tensor([[0.5, 0.5]]))


def test_logit_shift_invariance():
    # Dyadic logits/shifts avoid input rounding differences in FP32.
    z, idx, h, g = inputs(8)
    z = z.round() / 4
    baseline = grad_em_reference(z, idx, h, g)
    shifted = grad_em_reference(z + torch.tensor([[4.], [-2.], [8.]]), idx, h, g)
    for actual, expected in zip(shifted[:-1], baseline[:-1]):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    # Compare the unscaled KL residual so the established softmax tolerance is
    # not artificially magnified by the required 1 / eta factor.
    torch.testing.assert_close(
        0.1 * shifted.grad_logits, 0.1 * baseline.grad_logits,
        rtol=1e-6, atol=1e-7)


def test_selected_logits_equivalent_to_log_probabilities_in_high_precision():
    z, idx, h, g = inputs(8, torch.float64)
    # Keep the frozen FP32 dot product; isolate the FP64 algebraic identity.
    v = (g[:, None, :].float() * h.float()).sum(-1).double()
    selected_q = (z.gather(1, idx) - 0.1 * v).softmax(-1)
    log_p_q = (z.softmax(-1).gather(1, idx).log() - 0.1 * v).softmax(-1)
    torch.testing.assert_close(selected_q, log_p_q, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("k", [1, 2, 8, 12])
def test_selected_kl_autograd_matches_reference_on_fixed_support(k):
    eta = 0.2
    z, idx, h, g = inputs(k)
    z.requires_grad_()
    result = grad_em_reference(z, idx, h, g, eta)
    loss = F.kl_div(F.log_softmax(z.gather(1, idx), dim=-1),
                    result.q.detach(), reduction="sum") / eta
    (actual,) = torch.autograd.grad(loss, z)
    # Independent log_softmax backward versus explicit FP32 softmax subtraction.
    torch.testing.assert_close(actual, result.grad_logits, rtol=2e-6, atol=5e-7)
    inactive = torch.ones_like(z, dtype=torch.bool).scatter_(1, idx, False)
    assert torch.count_nonzero(actual[inactive]) == 0


def test_selected_router_gradient_rejects_full_softmax_kl():
    eta = 0.25
    z = torch.tensor([[0.2, -0.1, 0., 0.1]], dtype=torch.float32)
    idx = torch.tensor([[0, 3]])
    h = torch.tensor([[[0.5, -0.2], [1.1, 0.7]]], dtype=torch.float64)
    g = torch.tensor([[0.4, -0.6]], dtype=torch.float64)
    result = grad_em_reference(z, idx, h, g, eta)
    p = torch.softmax(z.float(), dim=-1)
    assert p.gather(1, idx).sum() < 0.6
    expected = torch.zeros_like(z).scatter_(1, idx, (result.a - result.q) / eta)
    old_full_softmax = (p - torch.zeros_like(p).scatter_(1, idx, result.q)) / eta
    torch.testing.assert_close(result.grad_logits, expected, rtol=0, atol=0)
    assert not torch.allclose(result.grad_logits, old_full_softmax)
    inactive = torch.ones_like(z, dtype=torch.bool).scatter_(1, idx, False)
    assert torch.count_nonzero(result.grad_logits[inactive]) == 0


@pytest.mark.parametrize("k", [1, 2, 8, 12])
def test_small_eta_recovers_normalized_topk_backward_in_high_precision(k):
    z, idx, h, g = inputs(k, torch.float64)
    selected = z.gather(1, idx).detach().requires_grad_()
    v = (g[:, None, :] * h).sum(-1)
    a = selected.softmax(-1)
    # Ordinary normalized Top-K forward autograd, fixed support.
    output = (a[..., None] * h).sum(1)
    standard_z, = torch.autograd.grad(output, selected, g)
    analytic = a * (v - (a * v).sum(-1, keepdim=True))
    torch.testing.assert_close(standard_z, analytic, rtol=1e-13, atol=1e-14)
    eta = 1e-5  # FP64 cancellation ~1e-11, well below the O(eta) truncation.
    q = (selected.detach() - eta * v).softmax(-1)
    torch.testing.assert_close((a.detach() - q) / eta, standard_z, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(q[..., None] * g[:, None, :],
                               a.detach()[..., None] * g[:, None, :], rtol=2e-4, atol=2e-5)
    if k > 1:
        # Demonstrate convergence, not merely one within-tolerance sample.
        errors = []
        for step_eta in (1e-2, 1e-3, 1e-4):
            step_q = (selected.detach() - step_eta * v).softmax(-1)
            errors.append(((a.detach() - step_q) / step_eta - standard_z).norm())
        assert errors[1] < errors[0] / 5
        assert errors[2] < errors[1] / 5


def test_fp32_reference_small_eta_tracks_normalized_router_backward():
    z, idx, h, g = inputs(8)
    # A larger eta avoids dividing FP32 softmax roundoff by a tiny number.
    result = grad_em_reference(z, idx, h, g, eta=1e-3)
    standard = result.a * (result.v - (result.a * result.v).sum(-1, keepdim=True))
    torch.testing.assert_close(result.grad_logits.gather(1, idx), standard,
                               rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("k", [1, 2, 8, 12])
def test_zero_upstream_gives_zero_router_gradient(k):
    z, idx, h, g = inputs(k)
    result = grad_em_reference(z, idx, h, torch.zeros_like(g))
    assert torch.count_nonzero(result.v) == 0
    assert torch.count_nonzero(result.grad_expert) == 0
    assert torch.equal(result.q, z.gather(1, idx).softmax(-1))
    assert torch.count_nonzero(result.grad_logits) == 0


def test_extreme_logits_do_not_take_log_of_underflowed_probabilities():
    z = torch.tensor([[10000., 9999., -10000., -9999.]])
    # Fixed support deliberately includes probabilities that underflow to zero.
    idx = torch.tensor([[2, 3]])
    assert torch.count_nonzero(z.softmax(-1).gather(1, idx)) == 0
    result = grad_em_reference(z, idx, torch.ones(1, 2, 3), torch.ones(1, 3))
    assert torch.isfinite(result.q).all()
    torch.testing.assert_close(result.q.sum(-1), torch.ones(1), rtol=0, atol=1e-7)


def test_q_is_detached_without_higher_order_graph_or_input_mutation():
    tensors = inputs()
    saved = [tensor.clone() for tensor in tensors]
    for tensor in (tensors[0], tensors[2], tensors[3]):
        tensor.requires_grad_(True)
    result = grad_em_reference(*tensors)
    assert not result.q.requires_grad and result.q.grad_fn is None
    for value in result:
        assert not value.requires_grad and value.grad_fn is None
    for actual, expected in zip(tensors, saved):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("eta", [0, -0.1, float("nan"), float("inf"), -float("inf"), True, False, "0.1", None])
def test_invalid_eta_rejected_by_reference_and_config(eta):
    with pytest.raises(ValueError, match="finite and positive"):
        grad_em_reference(*inputs(), eta=eta)
    config = load_experiment_config()
    config["model"]["grad_em_eta"] = eta
    with pytest.raises(ValueError, match="finite and positive"):
        validate_experiment_config(config)


@pytest.mark.parametrize("path", sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.toml")))
def test_existing_configs_have_valid_backward_configuration(path):
    config = load_experiment_config(path)
    assert config["model"]["moe_backward"] in ("standard", "grad_em")
    assert config["model"]["grad_em_eta"] > 0


def test_grad_em_toml_reaches_cuda_training_setup(tmp_path, monkeypatch):
    from modded_nanogpt_moe.train import main
    path = tmp_path / "grad_em.toml"
    path.write_text('run_name = "reference"\n[model]\nmlp_type = "moe"\n'
                    'moe_backward = "grad_em"\ngrad_em_eta = 0.2\n'
                    'moe_backend = "grouped_gemm"\n')
    config = load_experiment_config(path)
    assert config["model"]["moe_backward"] == "grad_em"
    assert config["model"]["grad_em_eta"] == 0.2
    monkeypatch.delenv("MLP_TYPE_OVERRIDE", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    class SetupReached(Exception):
        pass
    def stop_at_setup(device):
        assert device.type == "cuda"
        raise SetupReached  # No actual CUDA/distributed setup in this CPU test.
    monkeypatch.setattr(torch.cuda, "set_device", stop_at_setup)
    with pytest.raises(SetupReached):
        main(["train", "--config", str(path)])


def test_backward_mode_validation():
    config = load_experiment_config()
    config["model"]["moe_backward"] = "unknown"
    with pytest.raises(ValueError, match="standard.*grad_em"):
        validate_experiment_config(config)
    config["model"]["moe_backward"] = "grad_em"
    with pytest.raises(ValueError, match="requires MoE"):
        validate_experiment_config(config)


def test_checkpoint_backward_metadata_legacy_defaults_and_incompatibility():
    legacy = {"model": {"mlp_type": "moe"}, "training": {"total_steps": 20}}
    current = copy.deepcopy(legacy)
    current["model"].update(moe_backward="standard", grad_em_eta=0.1)
    checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION, "resolved_config": legacy}
    validate_checkpoint_config(checkpoint, current)
    validate_checkpoint_config({**checkpoint, "resolved_config": current}, legacy)
    for key, value in [("moe_backward", "grad_em"), ("grad_em_eta", 0.2)]:
        changed = copy.deepcopy(current)
        changed["model"][key] = value
        with pytest.raises(ValueError, match="incompatible"):
            validate_checkpoint_config(checkpoint, changed)
        validate_checkpoint_config({**checkpoint, "resolved_config": changed}, changed)
    assert "moe_backward" not in legacy["model"]  # no checkpoint mutation
    assert "grad_em_eta" not in legacy["model"]


@pytest.mark.parametrize("idx", [torch.tensor([[0, 0]]), torch.tensor([[-1, 1]]), torch.tensor([[0, 3]])])
def test_invalid_support_rejected(idx):
    with pytest.raises(ValueError, match="unique in-range"):
        grad_em_reference(torch.zeros(1, 3), idx, torch.zeros(1, 2, 4), torch.zeros(1, 4))
