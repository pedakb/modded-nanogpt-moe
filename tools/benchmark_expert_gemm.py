"""Isolate real GEMM forward/dX/dW; full-update timing stays in benchmark.sh.

Run implementations sequentially in fresh processes. Device-wide synchronization
at timing boundaries includes the extension's four auxiliary streams. Optional
torch.profiler traces are collected separately, never included in latency stats.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import statistics
import time

import torch

from modded_nanogpt_moe._grouped_gemm import native_phase, validate_native_inputs


def routing_counts(experts, top_k, tokens, routing):
    assignments = tokens * top_k
    counts = torch.full((experts,), assignments // experts, dtype=torch.int64)
    counts[:assignments % experts] += 1
    if routing == "imbalanced":
        # All tokens select the same k experts: a valid worst-case top-k routing.
        counts.zero_()
        counts[:top_k] = tokens
    return counts


def report_environment(build_log=None):
    print(f"torch={torch.__version__} CUDA={torch.version.cuda}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}", flush=True)
    try:
        import grouped_gemm
        from grouped_gemm import backend, ops
    except ImportError:
        print("nv-grouped-gemm is not installed/importable", flush=True)
        return
    print(f"nv-grouped-gemm={importlib.metadata.version('nv-grouped-gemm')}", flush=True)
    for module in (grouped_gemm, ops, backend, backend.backend):
        path = Path(module.__file__).resolve()
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        print(f"{module.__name__}: {path} sha256={digest}", flush=True)
    print("Actual installed autograd source:\n" + inspect.getsource(ops.GroupedGemm), flush=True)
    from tools.validate_grouped_gemm import report_extension_binary_and_build_flags
    report_extension_binary_and_build_flags(build_log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("extension", "torch"), default="extension")
    parser.add_argument("--geometry", choices=("e8", "e64"), default="e64")
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--routing", choices=("balanced", "imbalanced"), default="balanced")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--build-log", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    report_environment(args.build_log)
    if args.inspect_only:
        return
    if not torch.cuda.is_available():
        parser.error("GEMM timing requires CUDA; no CPU timings are substituted")
    if args.tokens <= 0 or args.warmup < 1 or args.iterations < 1:
        parser.error("tokens, warmup, and iterations must be positive")
    e, k, h = (8, 2, 1536) if args.geometry == "e8" else (64, 8, 384)
    counts = routing_counts(e, k, args.tokens, args.routing)
    offsets = counts.cuda().cumsum(0, dtype=torch.int32)
    print(f"implementation={args.implementation} N={args.tokens} D=768 H={h} E={e} k={k} "
          f"dtype=BF16 routing={args.routing} counts={counts.tolist()}", flush=True)
    if args.implementation == "extension":
        from grouped_gemm import backend
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
    for fc, din, dout in (("fc1", 768, h), ("fc2", h, 768)):
        # Fixed seed per FC gives identical data across separate implementation runs.
        torch.manual_seed(1234)
        a = torch.randn(args.tokens * k, din, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(e, din, dout, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn(args.tokens * k, dout, device="cuda", dtype=torch.bfloat16)
        if args.implementation == "torch":
            validate_native_inputs(a, b)
        for phase in ("forward", "dx", "dw"):
            def call():
                if args.implementation == "torch":
                    return native_phase(a, b, offsets, phase, grad)
                if phase == "forward":
                    return backend.gmm(a, b, counts, False, False)
                if phase == "dx":
                    return backend.gmm(grad, b, counts, False, True)
                return backend.gmm(a, grad, counts, True, False)
            for _ in range(args.warmup):
                result = call()
                torch.cuda.synchronize()
                del result
            milliseconds = []
            for _ in range(args.iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = call()
                torch.cuda.synchronize()
                milliseconds.append((time.perf_counter() - start) * 1000)
                del result
            label = f"{args.implementation}-{args.geometry}-{args.routing}-{fc}-{phase}"
            print(json.dumps(dict(phase=label, mean_ms=statistics.mean(milliseconds),
                                  median_ms=statistics.median(milliseconds))), flush=True)
            if args.profile_dir:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    with torch.cuda.nvtx.range(f"grouped_gemm.{fc}.{phase}"):
                        result = call()
                    torch.cuda.synchronize()
                path = args.profile_dir / f"{label}.json"
                prof.export_chrome_trace(str(path))
                # Raw trace GPU kernels only; don't sum nested operator/device times.
                events = json.loads(path.read_text())["traceEvents"]
                kernels = [event for event in events if event.get("cat") == "kernel"]
                print(f"{label}: kernel_launches={len(kernels)} "
                      f"kernel_gpu_ms={sum(event['dur'] for event in kernels) / 1000:.3f} "
                      f"trace={path}", flush=True)
                for name in sorted({event["name"] for event in kernels}):
                    selected = [event for event in kernels if event["name"] == name]
                    print(f"  {len(selected)} launches {sum(event['dur'] for event in selected)/1000:.3f} ms {name}", flush=True)
                del result, prof, events, kernels
        del a, b, grad
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
