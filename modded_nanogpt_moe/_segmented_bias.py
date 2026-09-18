"""CUDA-only contiguous-expert bias reduction; imported lazily by backward.

Triton comes from the Linux PyTorch environment. CPU/double/higher-order
autograd retain the PyTorch reference path in model.py. No autotuning runs or
host reads of routing metadata occur here.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _expert_bias_partials(
    Grad, Counts, Partials,
    WIDTH: tl.constexpr, EXPERTS: tl.constexpr,
    STRIDE_ROW: tl.constexpr, STRIDE_COL: tl.constexpr,
    COUNT_STRIDE: tl.constexpr, COUNT_BLOCK: tl.constexpr,
    SPLITS: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
):
    expert = tl.program_id(0)
    columns = tl.program_id(1) * COLS + tl.arange(0, COLS)
    split = tl.program_id(2)

    # Small cached metadata, reduced in-register. No offsets tensor/scan launch.
    experts = tl.arange(0, COUNT_BLOCK)
    preceding = tl.load(Counts + experts * COUNT_STRIDE,
                        mask=experts < expert, other=0)
    start = tl.sum(preceding, 0)
    length = tl.load(Counts + expert * COUNT_STRIDE)
    # Disjoint, exhaustive contiguous partitions; zero lengths need no branch.
    begin = start + length * split // SPLITS
    end = start + length * (split + 1) // SPLITS
    rows = tl.arange(0, ROWS)
    accumulator = tl.zeros((ROWS, COLS), tl.float32)
    for base in range(begin, end, ROWS):
        positions = base + rows
        values = tl.load(
            Grad + positions[:, None] * STRIDE_ROW + columns[None, :] * STRIDE_COL,
            mask=(positions[:, None] < end) & (columns[None, :] < WIDTH),
            other=0,
        ).to(tl.float32)
        accumulator += values
    partial = tl.sum(accumulator, axis=0)
    tl.store(Partials + (expert * SPLITS + split) * WIDTH + columns,
             partial, mask=columns < WIDTH)


@triton.jit
def _expert_bias_finish(Partials, Output, WIDTH: tl.constexpr,
                        SPLITS: tl.constexpr, COLS: tl.constexpr):
    expert = tl.program_id(0)
    columns = tl.program_id(1) * COLS + tl.arange(0, COLS)
    splits = tl.arange(0, SPLITS)
    partial = tl.load(
        Partials + (expert * SPLITS + splits[:, None]) * WIDTH + columns[None, :],
        mask=columns[None, :] < WIDTH, other=0,
    )
    # One rounding to the requested output dtype, after FP32 accumulation.
    tl.store(Output + expert * WIDTH + columns, tl.sum(partial, axis=0),
             mask=columns < WIDTH)


def segmented_bias_grad(grad_output, batch_sizes):
    """Sum trusted contiguous segments with two launches, including empty ones.

    Scratch is [E,8,H] FP32, independent of assignment count (1.5 MiB for
    E64/H768). Accepts strided/expanded gradients without a contiguous copy.
    Counts must be nonnegative and sum to grad_output.shape[0], as guaranteed
    by the caller's existing bincount. No GPU scalar is read by Python.
    """
    assert grad_output.is_cuda and batch_sizes.device == grad_output.device
    assert grad_output.ndim == 2 and batch_sizes.ndim == 1
    assert batch_sizes.dtype == torch.int64
    assert grad_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
    experts, width = batch_sizes.numel(), grad_output.shape[1]
    assert experts > 0 and width > 0
    splits, rows, cols = 8, 128, 32
    partials = torch.empty((experts, splits, width), device=grad_output.device,
                           dtype=torch.float32)
    output = torch.empty((experts, width), device=grad_output.device,
                         dtype=grad_output.dtype)
    # Respect the input device even if a caller has selected another CUDA device.
    with torch.cuda.device(grad_output.device):
        _expert_bias_partials[(experts, triton.cdiv(width, cols), splits)](
            grad_output, batch_sizes, partials, width, experts,
            grad_output.stride(0), grad_output.stride(1), batch_sizes.stride(0),
            triton.next_power_of_2(experts), splits, rows, cols, num_warps=4,
        )
        _expert_bias_finish[(experts, triton.cdiv(width, cols))](
            partials, output, width, splits, cols, num_warps=4,
        )
    return output
