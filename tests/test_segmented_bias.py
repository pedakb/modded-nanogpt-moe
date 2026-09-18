"""Bias-only optimization versus the former split/add/cat implementation."""
import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from modded_nanogpt_moe import model as model_module
from modded_nanogpt_moe.model import MoE, add_bias_by_expert_segments


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))]


def reference_bias(x, bias, counts, sorted_experts):
    # CPU transfer only in this test reference, never in the optimized helper.
    return torch.cat([segment + row for segment, row in zip(
        x.split(counts.cpu().tolist()), bias.unbind(0))])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("experts", [8, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("distribution", ["uneven", "imbalanced", "empty"])
def test_bias_parity_exact_representable_sums(device, experts, dtype, distribution):
    torch.manual_seed(51)
    counts_list = ([index % 5 for index in range(experts)] if distribution == "uneven"
                   else [4093, 0, 1, 2] + [0] * (experts - 4)
                   if distribution == "imbalanced" else [0] * experts)
    counts = torch.tensor(counts_list, device=device)
    ids = torch.repeat_interleave(torch.arange(experts, device=device), counts,
                                  output_size=sum(counts_list))
    values = torch.randn(sum(counts_list), 17, device=device, dtype=dtype)
    biases = torch.randn(experts, 17, device=device, dtype=dtype)
    # Binary fractions have exact FP32 sums here; no reduction-order tolerance.
    gradient = torch.randint(-8, 9, values.shape, device=device).to(dtype) / 8
    results = []
    for implementation in (reference_bias, add_bias_by_expert_segments):
        x, bias = values.clone().requires_grad_(), biases.clone().requires_grad_()
        out = implementation(x, bias, counts, ids)
        out.backward(gradient)
        results.append((out.detach(), x.grad, bias.grad))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(results[1][2][counts == 0]) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bias_random_long_reduction_against_float64(device, dtype):
    torch.manual_seed(53)
    counts_list = [8192, 0, 3, 1, 0, 7, 0, 0]
    counts = torch.tensor(counts_list, device=device)
    ids = torch.repeat_interleave(torch.arange(8, device=device), counts,
                                  output_size=sum(counts_list))
    values = torch.randn(sum(counts_list), 11, device=device, dtype=dtype)
    bias = torch.randn(8, 11, device=device, dtype=dtype)
    grad = torch.randn_like(values)
    outputs = []
    for implementation in (reference_bias, add_bias_by_expert_segments):
        x, b = values.clone().requires_grad_(), bias.clone().requires_grad_()
        out = implementation(x, b, counts, ids)
        out.backward(grad)
        torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)
        outputs.append((out.detach(), b.grad))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    # Bound FP32 reduction error by gamma_n * sum(abs(x)); this is an error
    # bound, not an empirically relaxed relative tolerance near cancellation.
    offset = 0
    for expert, length in enumerate(counts_list):
        segment = grad[offset:offset + length].double()
        exact = segment.sum(0)
        unit_roundoff = torch.finfo(torch.float32).eps / 2
        gamma = length * unit_roundoff / (1 - length * unit_roundoff)
        bound = gamma * segment.abs().sum(0)
        if dtype == torch.bfloat16:
            # Include one final BF16 rounding (both implementations return BF16).
            bound += torch.finfo(dtype).eps / 2 * (exact.abs() + bound)
        for _, actual in outputs:
            assert torch.all((actual[expert].double() - exact).abs() <= bound)
        offset += length


def test_bf16_reduction_does_not_accumulate_in_bf16():
    counts = torch.tensor([4096, 0])
    ids = torch.zeros(4096, dtype=torch.int64)
    x = torch.zeros(4096, 3, dtype=torch.bfloat16, requires_grad=True)
    bias = torch.zeros(2, 3, dtype=torch.bfloat16, requires_grad=True)
    add_bias_by_expert_segments(x, bias, counts, ids).sum().backward()
    torch.testing.assert_close(bias.grad[0], torch.full((3,), 4096, dtype=bias.dtype),
                               rtol=0, atol=0)
    assert torch.count_nonzero(bias.grad[1]) == 0


@pytest.mark.parametrize("experts", [8, 64])
def test_bias_has_one_segment_reduce_and_no_python_expert_operations(monkeypatch, experts):
    counts = torch.tensor([3] * experts)
    ids = torch.repeat_interleave(torch.arange(experts), counts)
    x = torch.randn(experts * 3, 11, requires_grad=True)
    bias = torch.randn(experts, 11, requires_grad=True)
    reduce = torch.segment_reduce
    calls = []
    def record(values, reduction, **kwargs):
        assert kwargs["lengths"] is counts and kwargs["unsafe"] is True
        calls.append(reduction)
        return reduce(values, reduction, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("bias helper used an expert loop/host-transfer operation")
    with monkeypatch.context() as patch:
        patch.setattr(torch, "segment_reduce", record)
        patch.setattr(torch, "cat", forbidden)
        for name in ("split", "unbind", "cpu", "item", "tolist"):
            patch.setattr(torch.Tensor, name, forbidden)
        add_bias_by_expert_segments(x, bias, counts, ids).sum().backward()
    assert calls == ["sum"]


def test_bias_gradcheck():
    counts = torch.tensor([2, 0, 1])
    ids = torch.tensor([0, 0, 2])
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: add_bias_by_expert_segments(a, b, counts, ids),
                                    (x, bias))


@pytest.mark.parametrize("experts", [8, 64])
def test_bias_operator_counts_do_not_scale_with_experts(experts):
    counts = torch.full((experts,), 3, dtype=torch.int64)
    ids = torch.repeat_interleave(torch.arange(experts), counts)
    x = torch.randn(experts * 3, 11, requires_grad=True)
    bias = torch.randn(experts, 11, requires_grad=True)
    grad = torch.randn_like(x)
    # CPU operator counts are a structural check, not a CUDA kernel-count claim.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        add_bias_by_expert_segments(x, bias, counts, ids).backward(grad)
    calls = {event.key: event.count for event in profile.key_averages()}
    assert calls["aten::index_select"] == calls["aten::add"] == calls["aten::segment_reduce"] == 1
    for forbidden in ("aten::split_with_sizes", "aten::unbind", "aten::cat",
                      "aten::index_add_", "aten::_index_put_impl_"):
        assert calls.get(forbidden, 0) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", ["modulelist", "packed"])
@pytest.mark.parametrize("experts,k", [(8, 2), (64, 8)])
@pytest.mark.parametrize("imbalanced", [False, True])
def test_moe_bias_change_all_gradients(monkeypatch, device, layout, experts, k, imbalanced):
    if device == "cuda":
        pytest.importorskip("grouped_gemm")
    else:
        def gmm(x, weights, counts, trans_b=False):
            return torch.cat([segment @ weight for segment, weight in
                              zip(x.split(counts.tolist()), weights)])
        monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(ops=SimpleNamespace(gmm=gmm)))
    torch.manual_seed(55)
    dim = 768 if device == "cuda" else 16
    hidden = dim * 2 if experts == 8 else dim // 2
    original = MoE(dim, experts, k, moe_backend="grouped_gemm", hidden_dim=hidden,
                   moe_parameter_layout=layout).to(device)
    if imbalanced:
        with torch.no_grad():
            original.router.weight.zero_()
            original.router.bias.copy_(-torch.arange(experts, device=device) / experts)
    updated = copy.deepcopy(original)
    # CPU BF16 also exercises the FP32-master/low-precision activation path.
    x = torch.randn(2, 33, dim, device=device, dtype=torch.bfloat16)
    gradient = torch.randn_like(x)
    results = []
    for model, implementation in ((original, reference_bias), (updated, add_bias_by_expert_segments)):
        monkeypatch.setattr(model_module, "add_bias_by_expert_segments", implementation)
        inputs = x.clone().requires_grad_()
        output = model(inputs)
        output.backward(gradient)
        results.append((output.detach(), inputs.grad, dict(model.named_parameters())))
    tolerance = dict(rtol=2e-2, atol=2e-2) if device == "cuda" else dict(rtol=0, atol=0)
    for actual, expected in zip(results[1][:2], results[0][:2]):
        torch.testing.assert_close(actual, expected, **tolerance)
    for name, reference in results[0][2].items():
        actual = results[1][2][name]
        assert actual.grad is not None and reference.grad is not None
        torch.testing.assert_close(actual.grad, reference.grad, **tolerance)
        if imbalanced:
            if layout == "packed" and name in ("fc_weight", "fc_bias", "proj_weight", "proj_bias"):
                assert torch.count_nonzero(actual.grad[k:]) == 0
            elif layout == "modulelist" and name.startswith("experts."):
                if int(name.split(".")[1]) >= k:
                    assert torch.count_nonzero(actual.grad) == 0
