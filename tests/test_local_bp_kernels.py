"""CPU mixed-signal algebra and real CUDA kernel acceptance tests."""
import pytest
import torch

from modded_nanogpt_moe._local_bp import _activation_backward, _cpu_signals


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("e,k", [(1, 1), (8, 2), (64, 8)])
def test_rescaling_identity_uses_forward_weights(normalize, e, k):
    torch.manual_seed(91)
    logits = torch.randn(11, e, dtype=torch.float64)
    p, ids = logits.softmax(-1).topk(k)
    if normalize:
        p = p / p.sum(-1, keepdim=True)
    g = torch.randn(11, 7, dtype=torch.float64)
    outputs = torch.randn(11, k, 7, dtype=torch.float64)
    q = (logits.gather(1, ids) - 0.4 * (g[:, None] * outputs).sum(-1)).softmax(-1)
    weight = torch.randn(11, k, 7, 13, dtype=torch.float64)
    derivative = 2 * torch.randn(11, k, 13, dtype=torch.float64).relu()
    bp = torch.einsum("nkd,nkdh->nkh", p[..., None] * g[:, None], weight) * derivative
    ge = torch.einsum("nkd,nkdh->nkh", q[..., None] * g[:, None], weight) * derivative
    torch.testing.assert_close(bp * (q / p)[..., None], ge, atol=2e-14, rtol=2e-13)


DEVICES = [pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires real CUDA kernels"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("e,k", [(1, 1), (8, 2), (7, 3), (64, 8)])
def test_mixed_combine_signals(device, dtype, e, k):
    triton = pytest.importorskip("triton")
    from modded_nanogpt_moe._local_bp_cuda import _mixed_combine_backward

    torch.manual_seed(121)
    n, d = 7, 19
    logits = torch.randn(n, e * 2, device=device, dtype=dtype)[:, ::2]
    weights, ids = logits.float().softmax(-1).topk(k)
    weights = weights.to(dtype)
    weights[0, -1] = 0  # q can be nonzero even if a selected weight underflows.
    order = ids.flatten().argsort(stable=True)
    rows = torch.empty_like(order)
    rows[order] = torch.arange(n * k, device=device)
    out = torch.randn(n * k, d, device=device, dtype=dtype)
    grad = torch.randn(n, d * 2, device=device, dtype=dtype)[:, ::2]
    expected = _cpu_signals(out, logits, weights, ids, order, grad, 0.4)
    bp, ge = torch.empty_like(out), torch.empty_like(out)
    q = torch.empty(n, k, device=device)
    gw, v = torch.empty_like(weights), torch.empty_like(q)
    _mixed_combine_backward[(n,)](
        out, logits, weights, ids, rows, grad, bp, ge, q, gw, v,
        d, k, *out.stride(), *logits.stride(), *weights.stride(),
        *ids.stride(), *grad.stride(), 0.4, True, True, True,
        triton.next_power_of_2(k), triton.next_power_of_2(d),
        num_warps=4, enable_fp_fusion=False)
    tolerance = dict(atol=2e-5, rtol=8e-3) if dtype == torch.bfloat16 else dict(atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(bp, expected[0], atol=0, rtol=0)
    torch.testing.assert_close(ge, expected[1], **tolerance)
    torch.testing.assert_close(q, expected[3], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(gw, expected[4], **tolerance)
    torch.testing.assert_close(v, expected[5], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("need_ge", [True, False])
def test_mixed_activation_rounding_and_zero_weight(device, dtype, need_ge):
    triton = pytest.importorskip("triton")
    from modded_nanogpt_moe._local_bp_cuda import _mixed_activation_backward

    torch.manual_seed(781)
    weights = torch.tensor([[0., 0.25], [0.75, 1.]], dtype=dtype)
    q = torch.tensor([[0.3, 0.7], [0.99, 0.01]])
    order = torch.tensor([3, 0, 2, 1])
    pre = torch.randn(4, 13, dtype=dtype)
    hidden = torch.randn_like(pre)
    expected = _activation_backward(hidden, pre, weights, q, order, need_ge)
    pre, hidden, weights, q, order = (t.to(device) for t in (pre, hidden, weights, q, order))
    bp = torch.empty_like(pre)
    ge = torch.empty_like(pre) if need_ge else None
    _mixed_activation_backward[(triton.cdiv(pre.numel(), 32),)](
        hidden, pre, weights, q, order, bp, ge, pre.numel(), pre.shape[1], need_ge, 32,
        num_warps=4, enable_fp_fusion=False)
    torch.testing.assert_close(bp.cpu(), expected[0], atol=0, rtol=0)
    if need_ge:
        torch.testing.assert_close(ge.cpu(), expected[1], atol=0, rtol=0)


def test_rescaling_does_not_form_overflowing_ratio():
    # q/p would overflow FP32. Scaling the tiny BP signal first remains finite.
    p = torch.tensor([[1e-40]])
    q = torch.ones_like(p)
    assert torch.isinf(q / p).all()
    hidden = torch.full((1, 8), 1e-40)
    pre = torch.ones_like(hidden)
    bp, ge = _activation_backward(hidden, pre, p, q, torch.tensor([0]), True)
    assert torch.isfinite(ge).all()
    torch.testing.assert_close(bp, 2 * hidden, atol=0, rtol=0)
    torch.testing.assert_close(ge, torch.full_like(ge, 2), atol=0, rtol=0)
