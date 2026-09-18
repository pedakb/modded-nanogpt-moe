"""CUDA correctness/launch checks plus CPU-safe lazy dispatch plumbing.

CPU checks do not compile or execute Triton. Real kernel validation is explicit
and skipped without CUDA; no dependency installation is needed on macOS.
"""
import importlib.util
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from modded_nanogpt_moe.model import _ExpertSegmentBias


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")


def test_cuda_backward_dispatches_without_cast_or_segment_reduce(monkeypatch):
    counts = object()
    grad = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    expected = object()
    calls = []
    def reduction(values, lengths):
        assert values is grad and lengths is counts
        calls.append(True)
        return expected
    monkeypatch.setitem(sys.modules, "modded_nanogpt_moe._segmented_bias",
                        SimpleNamespace(segmented_bias_grad=reduction))
    ctx = SimpleNamespace(needs_input_grad=(True, True, False, False), saved_tensors=(counts,))
    with torch.no_grad():
        result = _ExpertSegmentBias.backward(ctx, grad)
    assert result == (grad, expected, None, None) and calls == [True]
    ctx.needs_input_grad = (True, False, False, False)
    with torch.no_grad():
        assert _ExpertSegmentBias.backward(ctx, grad) == (grad, None, None, None)
    assert calls == [True]


@pytest.mark.parametrize("experts,width", [(8, 384), (64, 768)])
def test_launcher_has_two_grids_and_bounded_scratch(monkeypatch, experts, width):
    # Mock only the unavailable CUDA/Triton boundary; execute the real wrapper.
    launches, allocations = [], []
    class Kernel:
        def __init__(self, function):
            self.name = function.__name__
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.name, grid))
            return launch
    triton = ModuleType("triton")
    language = ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language
    triton.jit = Kernel
    triton.cdiv = lambda x, y: (x + y - 1) // y
    triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    path = Path(__file__).resolve().parents[1] / "modded_nanogpt_moe/_segmented_bias.py"
    spec = importlib.util.spec_from_file_location("bias_launcher_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device("cuda:0")
    grad = SimpleNamespace(is_cuda=True, device=device, ndim=2, dtype=torch.bfloat16,
                           shape=(524288, width), stride=lambda axis: (width, 1)[axis])
    counts = SimpleNamespace(device=device, ndim=1, dtype=torch.int64,
                             numel=lambda: experts, stride=lambda axis: 1)
    def allocate(shape, **kwargs):
        allocations.append((shape, kwargs["dtype"]))
        return object()
    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    module.segmented_bias_grad(grad, counts)
    assert launches == [
        ("_expert_bias_partials", (experts, math.ceil(width / 32), 8)),
        ("_expert_bias_finish", (experts, math.ceil(width / 32))),
    ]
    assert allocations == [((experts, 8, width), torch.float32),
                            ((experts, width), torch.bfloat16)]


@CUDA
@pytest.mark.parametrize("experts", [8, 64])
@pytest.mark.parametrize("width", [384, 768])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("distribution", ["balanced", "imbalanced", "empty"])
def test_triton_reduction_against_fp64(experts, width, dtype, distribution):
    from modded_nanogpt_moe._segmented_bias import segmented_bias_grad
    torch.manual_seed(71)
    rows = 65557  # Multi-tile loops, uneven partitions, and masked tails.
    if distribution == "balanced":
        counts_cpu = torch.bincount(torch.randint(experts, (rows,)), minlength=experts)
    elif distribution == "imbalanced":
        counts_cpu = torch.zeros(experts, dtype=torch.int64)
        counts_cpu[0], counts_cpu[-1] = rows - 3, 3
    else:
        rows = 0
        counts_cpu = torch.zeros(experts, dtype=torch.int64)
    counts = counts_cpu.cuda()
    grad = torch.randn(rows, width, device="cuda", dtype=dtype)
    actual = segmented_bias_grad(grad, counts)
    assert actual.dtype == dtype and actual.shape == (experts, width)
    torch.testing.assert_close(segmented_bias_grad(grad, counts), actual, rtol=0, atol=0)
    offset = 0
    for expert, length in enumerate(counts_cpu.tolist()):
        segment = grad[offset:offset + length].double()
        reference = segment.sum(0)
        # Per-value FP32 rounding depth: striped loop accumulation, 128-row
        # tree (7 levels), then 8-partial tree (3). Includes final dtype rounding.
        depth = math.ceil(math.ceil(length / 8) / 128) + 7 + 3
        u = torch.finfo(torch.float32).eps / 2
        gamma = depth * u / (1 - depth * u)
        bound = gamma * segment.abs().sum(0)
        if dtype != torch.float32:
            bound += torch.finfo(dtype).eps / 2 * (reference.abs() + bound)
            # Half of the smallest subnormal bounds rounding near zero.
            bound += torch.finfo(dtype).tiny * torch.finfo(dtype).eps / 2
        assert torch.all((actual[expert].double() - reference).abs() <= bound), expert
        if length == 0:
            assert torch.count_nonzero(actual[expert]) == 0
        offset += length


@CUDA
@pytest.mark.parametrize("view", ["sliced", "transposed", "expanded"])
def test_triton_strides_and_exact_sums(view):
    from modded_nanogpt_moe._segmented_bias import segmented_bias_grad
    counts_cpu = [8193, 0, 7, 0, 1, 8, 0, 0]
    rows, width = sum(counts_cpu), 37  # Non-power-of-two feature tail.
    counts = torch.tensor(counts_cpu, device="cuda")
    if view == "sliced":
        grad = torch.ones(rows, width * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    elif view == "transposed":
        grad = torch.ones(width, rows, device="cuda", dtype=torch.bfloat16).T
    else:
        grad = torch.ones((), device="cuda", dtype=torch.bfloat16).expand(rows, width)
    actual = segmented_bias_grad(grad, counts)
    expected = counts.to(grad.dtype)[:, None].expand(-1, width)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@CUDA
def test_triton_bias_backward_two_kernels_no_generic_reduction():
    from modded_nanogpt_moe.model import add_bias_by_expert_segments
    counts = torch.full((64,), 129, device="cuda", dtype=torch.int64)
    ids = torch.repeat_interleave(torch.arange(64, device="cuda"), counts, output_size=64 * 129)
    x = torch.randn(64 * 129, 384, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(64, 384, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn_like(x)
    # Compile/initialize outside profiling; profile backward only.
    add_bias_by_expert_segments(x, bias, counts, ids).backward(grad)
    x.grad = bias.grad = None
    out = add_bias_by_expert_segments(x, bias, counts, ids)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        out.backward(grad)
        torch.cuda.synchronize()
    kernels = [event.name for event in profile.events()
               if event.device_type == torch.autograd.DeviceType.CUDA]
    assert sum("_expert_bias_partials" in name for name in kernels) == 1, kernels
    assert sum("_expert_bias_finish" in name for name in kernels) == 1, kernels
    assert not any("segment_reduce" in name for name in kernels), kernels
