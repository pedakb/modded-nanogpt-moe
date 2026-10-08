"""Small, conservative comparison of this experiment's Nsight SQLite exports.

GPU attribution uses kernel -> runtime launch correlation -> same-thread NVTX
enclosure, never kernel-name guesses. All durations in exports are nanoseconds.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
from pathlib import Path
import sqlite3
import re


def validate_capture(path, log_path, status):
    """Accept intentional launcher termination only with independent evidence."""
    log = log_path.read_text(errors="replace")
    started = "Capture range started in the application."
    ended = "Capture range ended in the application."
    if log.count(started) != 1 or log.count(ended) != 1 or log.index(started) >= log.index(ended):
        raise ValueError("Nsight must confirm exactly one capture start followed by capture end")
    # A full SignalException traceback is expected; other exception endings and
    # explicit profiler errors are not. Never whitelist arbitrary exit failures.
    signal = re.compile(r"(?:[\w.]+\.)?SignalException:.*(?:signal:?\s*15|SIGTERM)", re.I)
    for line in log.splitlines():
        exception = re.search(r"(?:[\w.]*Error|[\w.]*Exception):", line)
        explicit_error = re.match(r"\s*(?:ERROR\b|Error:|FATAL\b)", line)
        if (exception or explicit_error) and not signal.search(line):
            raise ValueError(f"Unexpected error in execution log: {line}")
    if status != 0:
        evidence = signal.search(log)
        # Nsight buffers its start/end notices: merged stdout/stderr can put
        # both AFTER torchrun's termination traceback. Log order is not a clock.
        if status not in (1, 143) or evidence is None:
            raise ValueError(f"Profiler exit {status} is not a verified expected SIGTERM termination")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite export failed integrity check")
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        events = [(r["start"], r["end"], r["globalTid"],
                   r["text"] or strings.get(r["textId"], ""))
                  for r in db.execute("SELECT * FROM NVTX_EVENTS")]
        updates = sorted((r for r in events if r[3].startswith("optimizer_step.update_")),
                         key=lambda r: r[0])
        if [r[3] for r in updates] != ["optimizer_step.update_6", "optimizer_step.update_7"]:
            raise ValueError("NVTX must contain exactly updates 6 and 7")
        for lo, hi, tid, name in updates:
            if hi is None or hi <= lo:
                raise ValueError(f"Incomplete NVTX range: {name}")
            children = [r[3] for r in events if r[2] == tid and r[1] is not None
                        and lo <= r[0] < r[1] <= hi]
            expected = [f"{phase}.microbatch_{i}" for phase in ("forward", "backward") for i in range(8)]
            expected += ["data_preparation", "optimizer_update.index_0.AdamW", "optimizer_update.index_1.Muon"]
            if any(children.count(child) != 1 for child in expected):
                raise ValueError(f"{name}: missing/duplicate complete microbatch or optimizer NVTX ranges")
        # The trainer synchronizes outside the final CPU update range, before
        # profilerStop. Verify GPU work drained rather than trusting NVTX alone.
        kernel_end = db.execute("SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        syncs = [r for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME")
                 if "cudaDeviceSynchronize" in strings.get(r["nameId"], "")
                 and r["start"] >= updates[-1][1]
                 and r["end"] is not None and kernel_end is not None and r["end"] >= kernel_end]
        if not syncs:
            raise ValueError("Missing completed final CUDA synchronization after update 7")
    analyze(path)
    print(f"Validated updates 6 and 7, eight microbatches each, both optimizers and GPU drain; profiler exit={status}")


def union_duration(intervals):
    total = 0
    end = None
    for lo, hi in sorted(intervals):
        if end is None or lo > end:
            total += hi - lo
        else:
            total += max(0, hi - end)
        end = max(end if end is not None else hi, hi)
    return total


def category(name):
    if name.startswith("grouped_gemm."):
        return "grouped_gemm"
    if name.startswith("optimizer_update."):
        return "muon" if name.endswith(".Muon") else "adamw"
    if name in ("moe.router_topk", "moe.pack", "moe.combine", "moe.stack_params"):
        return "router_dispatch_forward"
    return None


def analyze(path, expected_updates=2):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME"}
    if not required <= tables:
        raise ValueError(f"{path}: missing required tables {required - tables}; inspect export/schema")
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    ranges = []
    for r in db.execute("SELECT * FROM NVTX_EVENTS WHERE end > start"):
        name = r["text"] or strings.get(r["textId"], "")
        ranges.append((r["start"], r["end"], r["globalTid"], name))
    updates = sorted(r for r in ranges if r[3].startswith("optimizer_step.update_"))
    if len(updates) != expected_updates:
        raise ValueError(f"{path}: expected {expected_updates} complete update ranges, found {len(updates)}")
    # Capture starts after a device fence. Final fence drains asynchronous work
    # after the last CPU update range. Include that tail in the envelope below.
    start = updates[0][0]
    kernels = list(db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    if not kernels:
        raise ValueError(f"{path}: no CUDA kernels")
    runtime = list(db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    launches = defaultdict(list)
    for r in runtime:
        launches[r["correlationId"]].append(r)
    indexed = defaultdict(list)
    for lo, hi, tid, name in ranges:
        label = category(name)
        if label:
            indexed[(tid, label)].append((lo, hi))
    for key, values in indexed.items():
        values.sort()
    starts = {key: [r[0] for r in values] for key, values in indexed.items()}
    def encloses(api, label):
        key = (api["globalTid"], label)
        i = bisect_right(starts.get(key, []), api["start"]) - 1
        return i >= 0 and indexed[key][i][1] >= api["end"]

    gpu = defaultdict(int)
    top = defaultdict(lambda: [0, 0])
    busy = []
    end = updates[-1][1]
    for k in kernels:
        if k["start"] < start:
            raise ValueError(f"{path}: kernel precedes first update; unexpected capture contents")
        duration = k["end"] - k["start"]
        busy.append((k["start"], k["end"]))
        end = max(end, k["end"])
        # Correlation IDs are process-local. Disambiguate using encoded PID.
        candidates = [a for a in launches[k["correlationId"]]
                      if a["globalTid"] >> 24 == k["globalPid"] >> 24
                      and a["start"] <= k["start"]]
        api = candidates[0] if len(candidates) == 1 else None
        label = "unattributed"
        if api is not None:
            for candidate in ("grouped_gemm", "muon", "adamw", "router_dispatch_forward"):
                if encloses(api, candidate):
                    label = candidate
                    break
        gpu[label] += duration
        name = strings.get(k["demangledName"], str(k["demangledName"]))
        top[name][0] += duration
        top[name][1] += 1
    # Include copies and memsets so 'idle' does not mistake transfers for idle.
    missing_activity = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            for lo, hi in db.execute(f'SELECT start,end FROM "{table}"'):
                if hi > start:
                    busy.append((max(start, lo), hi))
                    end = max(end, hi)
        else:
            missing_activity.append(table)
    api_total = sum(r["end"] - r["start"] for r in runtime)
    sync_total = sum(r["end"] - r["start"] for r in runtime
                     if "Synchronize" in strings.get(r["nameId"], ""))
    launch_total = sum(r["end"] - r["start"] for r in runtime
                       if "Launch" in strings.get(r["nameId"], ""))
    n = len(updates)
    metrics = {"updates": n, "capture_envelope_ms_per_update": (end-start)/1e6/n,
               "cpu_update_range_ms_per_update": sum(r[1]-r[0] for r in updates)/1e6/n,
               "gpu_kernel_sum_ms_per_update": sum(gpu.values())/1e6/n,
               "gpu_busy_union_ms_per_update": union_duration(busy)/1e6/n,
               "gpu_no_recorded_activity_ms_per_update": ((end-start)-union_duration(busy))/1e6/n,
               "cuda_runtime_api_sum_ms_per_update": api_total/1e6/n,
               "cuda_runtime_sync_sum_ms_per_update": sync_total/1e6/n,
               "cuda_runtime_launch_sum_ms_per_update": launch_total/1e6/n,
               "kernel_count_per_update": len(kernels)/n}
    for label in ("grouped_gemm", "muon", "adamw", "router_dispatch_forward", "unattributed"):
        metrics[f"{label}_kernel_sum_ms_per_update"] = gpu[label]/1e6/n
    db.close()
    return metrics, sorted(top.items(), key=lambda item: item[1][0], reverse=True)[:15], missing_activity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--validate-capture", type=Path)
    parser.add_argument("--execution-log", type=Path)
    parser.add_argument("--profile-status", type=int)
    args = parser.parse_args()
    if args.validate_capture:
        if args.execution_log is None or args.profile_status is None:
            parser.error("capture validation requires --execution-log and --profile-status")
        validate_capture(args.validate_capture, args.execution_log, args.profile_status)
        return
    names = ("moe_e64k8", "moe_e256k6", "moe_e256k6_shared")
    rows = []
    details = []
    for name in names:
        metrics, top, missing = analyze(args.results / name / "profile.sqlite")
        rows.append(dict(configuration=name, **metrics))
        details.append(f"\n### {name}: top kernels\n\n| ms/update | launches/update | kernel |\n|---:|---:|---|\n" +
                       "\n".join(f"| {ns/2e6:.3f} | {count/2:g} | {kernel.replace('|', '/')} |"
                                 for kernel, (ns, count) in top))
        if missing:
            details.append(f"\n{name}: absent activity tables: {', '.join(missing)}; no-recorded-activity is an upper bound on idle.\n")
    baseline = rows[0]
    delta_keys = [key for key in baseline if key.endswith("ms_per_update")]
    for row in rows:
        for key in delta_keys:
            row[f"delta_vs_e64_{key}"] = row[key] - baseline[key]
    with (args.results / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shown = [k for k in baseline if not k.startswith("delta") and k not in ("updates", "configuration")]
    lines = ["# Expert scaling: two profiled updates per configuration", "",
             "Unprofiled reference: E64 2428 ms, E128 2554 ms, E256 2807 ms, shared E256 2784 ms. E256−E64 = 379 ms; shared−E256 = −23 ms.", "",
             "| measurement (per update) | E64 | E256 | E256 shared | E256−E64 | shared−E256 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in shown:
        values = [r[key] for r in rows]
        lines.append(f"| {key} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {values[1]-values[0]:+.3f} | {values[2]-values[1]:+.3f} |")
    lines += ["", "## Interpretation and limits", "",
              "The capture envelope runs from the first CPU update range start through the last update range end or GPU activity end, whichever is later. It includes final GPU drain and inter-update gaps; it excludes profiler start/stop overhead outside that envelope. CPU update ranges are enqueue intervals, not synchronized step latency. Two updates give a short, instrumented estimate, not a replacement for unprofiled steady-state benchmarks.", "",
              "Kernel sums can overlap across streams. GPU busy union merges kernels, copies and memsets; no-recorded-activity is the envelope minus that union, not proof of a particular stall. CPU API, CPU NVTX, GPU sums and idle must not be added. Runtime API sums include concurrent CPU threads and synchronization waits for GPU work; synchronization is a subset of runtime API time.", "",
              "GPU component attribution requires a unique same-process runtime correlation and same-thread NVTX enclosure of the complete launch API. No attribution is inferred from kernel names. CUDA graph replay and driver-only launches may remain unattributed. Router/dispatch covers the explicit forward router_topk, pack, stack_params and combine ranges only. Backward hook ranges delimit autograd boundaries and can interleave branches: they are deliberately not used for precise router attribution.", "",
              "Muon includes all Muon-owned matrices, including attention, routers, shared weights and all routed expert weights. It is not an expert-only measurement. Fixed routed FFN FLOPs do not fix Muon work or expert parameter traffic. Use optimizer_update.*.Muon in the timeline to test this hypothesis; grouped_gemm.* measures explicitly annotated expert GEMM launches.", "",
              "Unattributed kernels include attention, head/loss, shared FFN, bias/activation/backward dispatch and any failed correlation. Inspect forward.microbatch_*, backward.microbatch_*, moe.*, grouped_gemm.* and optimizer_update.* in Nsight for detailed localization. Shared-expert and attention ranges are not individually annotated. Inspect CUDA API rows, OS runtime events and GPU gaps for CPU launch pressure, barriers and synchronization. CPU stack sampling/context switches are disabled to limit overhead.", "",
              "Compare deltas to the 379 ms unprofiled gap as evidence, not an additive explanation. An increase in Muon kernel time supports optimizer scaling; grouped GEMM duration/count changes support expert execution scaling. Idle/API growth needs manual timeline inspection. Repeat the experiment if first-capture compilation, profiler warnings, contention, or unstable clocks are visible in execution logs/timelines."]
    (args.results / "summary.md").write_text("\n".join(lines + details) + "\n")


if __name__ == "__main__":
    main()
