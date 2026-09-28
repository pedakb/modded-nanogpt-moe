"""Synchronized single-MoE forward/backward and memory comparison.

Run each mode in a fresh process. This measures a full MoE layer with FP32
master parameters and fixed inputs/upstream gradients, not a training update.
No profiler, optimizer, or CPU substitute is included in the timed regions.
"""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

import torch

from modded_nanogpt_moe._grouped_gemm import expert_dgrad
from modded_nanogpt_moe.model import MoE


def timed(call):
    # Device-wide fences also include nv-grouped-gemm's auxiliary streams.
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, 1000 * (time.perf_counter() - start)


def statistics_ms(samples):
    return {"mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("standard", "global", "local_bp"), required=True)
    parser.add_argument("--implementation", choices=("extension", "torch"), default="torch")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--geometry", choices=("e8", "e64"), default="e8")
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("timing requires CUDA; no CPU timings are substituted")
    if min(args.tokens, args.warmup, args.iterations) < 1:
        parser.error("tokens, warmup and iterations must be positive")
    if args.implementation == "torch" and args.dtype != "bfloat16":
        parser.error("the production native grouped backend requires BF16")
    os.environ["MOE_GMM_IMPLEMENTATION"] = args.implementation
    torch.manual_seed(1234)
    dtype = getattr(torch, args.dtype)
    e, k, h = (8, 2, 1536) if args.geometry == "e8" else (64, 8, 384)
    model = MoE(768, e, k, hidden_dim=h, moe_backend="grouped_gemm",
                moe_parameter_layout="packed",
                moe_backward="standard" if args.mode == "standard" else "grad_em",
                grad_em_mode="local_bp" if args.mode == "local_bp" else "global",
                grad_em_eta=args.eta).cuda()
    x = torch.randn(1, args.tokens, 768, device="cuda", dtype=dtype, requires_grad=True)
    g = torch.randn_like(x) / 768**0.5

    def clear_gradients():
        model.zero_grad(set_to_none=True)
        x.grad = None

    for _ in range(args.warmup):
        clear_gradients()
        model(x).backward(g)
    torch.cuda.synchronize()
    clear_gradients()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    forward, backward = [], []
    for _ in range(args.iterations):
        clear_gradients()
        out, ms = timed(lambda: model(x))
        forward.append(ms)
        _, ms = timed(lambda: out.backward(g))
        backward.append(ms)
        del out
    report = {
        "mode": args.mode, "implementation": args.implementation,
        "dtype": args.dtype, "master_dtype": "float32", "eta": args.eta,
        "tokens": args.tokens, "dim": 768, "hidden_dim": h, "experts": e, "top_k": k,
        "warmup": args.warmup, "iterations": args.iterations,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "forward": statistics_ms(forward), "backward": statistics_ms(backward),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    if args.implementation == "extension":
        report["nv_grouped_gemm"] = importlib.metadata.version("nv-grouped-gemm")
    # Isolate the extra FC2 dgrad after recording layer peaks, so this scratch
    # cannot contaminate the forward/backward memory comparison.
    if args.mode == "local_bp":
        clear_gradients()
        with torch.no_grad():
            ids = model.router(x.flatten(0, 1)).float().softmax(-1).topk(k, -1).indices
            counts_device = torch.bincount(ids.flatten(), minlength=e)
            counts = counts_device.cpu()
            offsets = counts_device.cumsum(0, dtype=torch.int32)
            weight = model.proj_weight.to(dtype)
            grad = torch.randn(args.tokens * k, 768, device="cuda", dtype=dtype)
            call = lambda: expert_dgrad(grad, weight, counts, offsets, args.implementation, "fc2")
            for _ in range(args.warmup):
                result, _ = timed(call)
                del result
            samples = []
            for _ in range(args.iterations):
                result, ms = timed(call)
                samples.append(ms)
                del result
            report["isolated_extra_fc2_dgrad"] = statistics_ms(samples)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
