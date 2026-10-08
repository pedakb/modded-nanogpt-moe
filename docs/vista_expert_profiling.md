# Vista GH200 expert scaling profile

From the current repository root on an allocated GH200 node:

```bash
bash scripts/vista/profile_experts.sh
```

To reuse a completed E64 capture and profile only the two E256 configurations:

```bash
bash scripts/vista/profile_experts.sh --reuse-e64 "$STOCKYARD/profiles/modded-nanogpt-moe/vista/expert-scaling-20261008T215006Z-8ZGJsw"
```

Reuse requires identical saved package source, all three TOMLs, pyproject.toml
and uv.lock. It copies the raw E64 report, execution log and actual exit status
into a fresh results directory, records `reused-from.txt`, re-exports and
revalidates it, then generates E64 stats before starting E256. Existing results
are never overwritten. Use the same provisioned environment and GH200 node.

This current-node script never submits jobs. It profiles E64/K8, E256/K6,
and E256/K6 shared sequentially in separate processes, using the production
TOMLs unchanged. It freezes source and configs in persistent Stockyard storage
and reuses the checkout's provisioned uv environment (`--no-sync`). Queued jobs
and production trainer files are untouched. Scientific environment overrides
are rejected. The GH200 native torch grouped-GEMM backend matches train.sh.

The existing trainer synchronizes before cudaProfilerStart, captures complete
updates 6 and 7, synchronizes, then calls cudaProfilerStop. Nsight uses
`--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --kill=sigterm`.
The trainer stops capture after update 7 but does not exit its training loop;
Nsight shuts down the session and terminates the launched process group at that
capture boundary. STOP_AFTER_COMPLETED_UPDATES is explicitly unset because it
is incompatible with NSYS_PROFILE. The production schedule horizon remains
unchanged. Initial validation, initialization and compilation
are outside capture; termination avoids final validation. TensorBoard,
checkpointing and reproducibility snapshots are disabled. CUDA, NVTX and OS
runtime events are captured; CPU sampling and context switches are disabled.
Fresh per-config compiler caches are stored alongside reports. No dependencies
or extensions are installed or rebuilt.

The existing MoE prefix is eager; the head/loss is compiled. Muon's update is
compiled independently. Existing ranges include full update, data preparation,
microbatch forward/backward, individual optimizers, router/top-k, pack,
parameter preparation, activation, combine and explicit grouped-GEMM
forward/dX/dW. Backward tensor hooks describe autograd enqueue boundaries and
can interleave branches. They are not exclusive component measurements.
Shared FFN and attention are not independently ranged. No extra instrumentation
is necessary for the initial bottleneck comparison. The Muon optimizer range
covers attention/router/shared weights as well as all experts; an expert-only
optimizer split would require additional instrumentation in a separate snapshot.

Results are unique and never overwritten:

```text
$STOCKYARD/profiles/modded-nanogpt-moe/vista/expert-scaling-UTCSTAMP-RANDOM/
  environment.txt, source.diff, sha256.txt
  source/                    frozen package, pyproject.toml, uv.lock, trainer logs
  configs/                   the three exact TOMLs
  profile_experts.sh, compare_nsys_experts.py
  moe_e64k8/                 also moe_e256k6/ and moe_e256k6_shared/
    profile.nsys-rep, profile.sqlite
    execution.log
    cuda_gpu_kern_sum.csv, cuda_api_sum.csv, nvtx_sum.csv, nvtx_kern_sum.csv
    *.log                    statistics/export diagnostics
    inductor-cache/, triton-cache/
  summary.md, comparison.csv
```

The profile and tee exit statuses are recorded separately. A nonzero profiler
status (only 1 or 143) is accepted only with the expected SIGTERM
SignalException, Nsight start/end confirmations, a readable exported report,
exact closed NVTX update ranges 6 and 7 with all eight forward/backward
microbatches and both optimizers, and a completed final CUDA synchronization.
Other error/exception signatures, missing reports, failed exports, failed log
writes and failed statistics stop the suite. `profile.exit-status` and
`capture-validation.txt` record the decision. Stats run only after validation.
Statistics read the validated `.sqlite` directly and use `--quiet` for clean CSV.
Passing `.nsys-rep` to stats can trigger a redundant export/version check and
fail on an existing export; no force-overwrite workaround is needed here.
Nsight can buffer its capture notifications until after torchrun's traceback;
their position relative to SIGTERM messages is deliberately not used as timing
evidence. Capture start must still precede capture end, and all independent
SQLite/NVTX/synchronization checks remain mandatory.
SIGTERM is retained: a graceful in-process stop would need a wrapper or trainer
change; `--kill=none` would allow the full training run to continue.
Inspect execution/statistics logs for dropped events, compilation during
capture, contention and backend errors. The analyzer validates two full update
ranges and required SQLite tables; incompatible schemas fail clearly. Analysis
can be rerun with `uv run --no-sync python RESULTS/compare_nsys_experts.py RESULTS`.

Use the Markdown report's E256−E64 and shared−E256 columns. Kernel attribution
requires process-local CUDA runtime launch correlation and same-thread NVTX
enclosure; names alone never establish ownership. GPU kernel sums can overlap.
The GPU activity union includes copies/memsets. Its complement is time with no
recorded activity, an upper bound on idle when events are missing. The capture
envelope includes GPU drain and inter-update gaps, while CPU update ranges
measure enqueue intervals. CPU API durations overlap GPU execution and must not
be added to it. Launch and synchronization times are subsets of runtime API time.
Driver-only launches and graph replay may remain unattributed.

The approximately 379 ms unprofiled gap is a reference, not a delta this short
instrumented capture is guaranteed to reproduce. Fixed active FFN work leaves
all-expert Muon work, parameter traffic, router width and grouped-GEMM geometry
free to change. Muon kernel growth is evidence for optimizer scaling, not proof
that experts alone caused it. Router/dispatch GPU attribution covers explicit
forward ranges only. Unattributed GPU time includes attention/head/shared FFN
and backward operations. Inspect those ranges and GPU gaps manually in Nsight
before assigning residual latency. Two captured updates provide no robust
variance estimate; repeat the command if needed.

The repository previously had no dedicated Vista nsys launcher or cross-run
SQLite analyzer. `docs/grouped_gemm.md` contains direct full-trainer nsys command
examples; `tools/validate_grouped_gemm.py` and `tools/benchmark_expert_gemm.py`
provide isolated-layer/phase torch.profiler traces, which do not explain a
complete optimizer-update gap by themselves.

Nsight option semantics and export reference:
[NVIDIA User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
[NVIDIA Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html).
