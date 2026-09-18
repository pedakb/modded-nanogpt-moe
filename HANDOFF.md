# Current handoff

Updated: 2026-09-18

## Current goal and state

Current base on `cleanup-active-codebase`: `68325ff` (`Add native grouped GEMM
implementation`), initially clean. Current task: sampled TensorBoard diagnostics
for router/expert/optimizer dynamics. The prior GEMM candidate is committed;
its CUDA correctness/performance remain pending as recorded below. No commits,
remote jobs, dependency installs or extension rebuilds performed this task.

## TensorBoard diagnostics (current work)

- New `diagnostics.py` owns aggregation/snapshot/emission logic. TOML defaults:
  `[diagnostics] scalar_interval=10, histogram_interval=0, during_nsys=false`.
  Interval 0 disables diagnostics; histograms must be a scalar-interval multiple.
  Existing configs need no edits; settings print with resolved config and do
  not enter checkpoint compatibility state.
- Only rank-zero with an existing writer constructs the observer. Benchmark
  always bypasses it; Nsight bypasses it for the whole run unless explicitly
  enabled. Non-sampled updates do no diagnostic tensor work/copies/transfers.
- Two model observation sites (loop/grouped routing) supply detached existing
  full softmax/logits/actual top-k IDs. Capture is enabled only across accumulated
  training microbatches of a sampled update, cleared in finally. Model math,
  routing choices, grouped GEMM, bias/combine, optimizer code and compile
  boundaries are unchanged.
- Per-layer router norms (weight+bias), independent expert FC1/FC2 weight norms
  summarized min/median/mean/max/population-std; aggregate attention/embedding/
  head weight norms. Log pre-update parameter and accumulated all-reduced
  gradient norms/ratio, actual stored update norm/ratio, and router-versus-median-
  expert gradient and update-ratio comparisons. Denominator zero => NaN.
- Follow-up adds parameter/gradient/actual-update RMS using each expert's own
  element count before summarizing. Expert RMS tags use p10/median/p90; existing
  norm/ratio summaries remain and gain p10/p90. Median is the typical-expert
  statistic; percentiles use linear interpolation via kthvalue selection.
- Router `logit_rms` pools token-wise centered variance over all sampled
  microbatches, so common logit shifts do not change it. `dL_dlogits_rms` uses
  temporary router-forward/tensor-gradient hooks only inside sampled capture;
  hooks retain scalar sums/counts, not logits, full gradients or graphs, and
  are removed in finally. No normal-update hooks, extra backward, synchronization
  or device-to-host transfer. Gradient RMS reflects actual dtype and existing
  summed-loss scaling; omitted when no logit gradient is observed. Model,
  trainer, optimizer and kernels needed no additional edits for this follow-up.
- Exact actual update norm requires detached independent parameter snapshots,
  only at sampled updates. Includes AdamW/Muon momentum, weight decay and dtype
  rounding without touching optimizer internals. FP32 snapshots reused as
  subtraction scratch; BF16 promotes before subtracting. Roughly 2 GiB extra
  snapshots for standard E64 packed, plus routing temporaries; not memory-free.
- Routing covers full entropy/normalized entropy, top1/max probability, top1-
  top2 gap, k/k+1 logit margin mean/midpoint median/threshold fractions, and
  actual assignment utilization including empty experts. Extra topk(k+1), no
  expert sort; median via kthvalue selection. Counts/sums aggregate over the
  entire sampled update. Margins retained as detached per-token FP32 values
  (~24 MiB for 12 layers/524288 rank-local tokens). No graphs retained.
- `train/loss` is sampled rank-local cross entropy per token; `val/loss` aliases
  existing `eval/val_loss`. Existing scalar tags/cleanup/purge unchanged. Scalars
  use one batched host transfer per sample; optional histograms use a second.
  Routing/loss are rank-local; gradients follow the existing SUM all-reduce.
  No new distributed reductions or checkpoint payload state.
- Documentation: `docs/diagnostics.md` details all tags, definitions, gating,
  memory cost and caveats; linked from README. Diagnostics source is included
  in complete training source snapshots.
- Tests: deterministic entropy/margins/loads/zero experts, exact observer-on/off
  model/input/router/expert-gradient parity in both layouts, real AdamW/Muon
  update/state parity, actual BF16 rounding, config validation, callback cleanup,
  trainer optimizer boundaries, and factory/benchmark/Nsight/rank gating.
  Follow-up covers centered-shift invariance, unequal-microbatch pooling,
  per-expert RMS/percentiles, FP32/BF16 logit-gradient RMS, weak-reference graph
  release, and hook cleanup on exceptions. Targeted diagnostics/package/Nsight
  checks: 54 passed. Full local suite: 212 passed, 143 CUDA-dependent checks
  skipped. `git diff --check` passed.
- Pending: Vista/LS6 CUDA event-file inspection, peak-memory and runtime overhead
  with TB enabled. Lower frequency reduces average overhead, not snapshot peak
  memory; disable diagnostics for tight memory budgets. Do not infer GPU overhead
  from CPU tests. Existing end-to-end benchmark remains diagnostic-free.
- Dirty task files: diagnostics.py (new), config.py, model.py, train.py,
  tests/test_diagnostics.py (new), tests/test_package.py, docs/diagnostics.md
  (new), README.md, HANDOFF.md. Prior GEMM files and optimizers untouched.
  Follow-up edits are limited to diagnostics.py, tests/test_diagnostics.py,
  docs/diagnostics.md and HANDOFF.md; earlier uncommitted diagnostics work is
  preserved. Next: inspect these new tags and sampled overhead on Vista with
  TensorBoard enabled; CPU tests cannot establish CUDA correctness/performance.

## Native grouped GEMM candidate (committed; GPU validation pending)

- Source traced to `grouped_gemm.ops.gmm` / `GroupedGemm` autograd, pinned
  nv-grouped-gemm 1.1.4.post8 (15721c6). On SM90 all forward/dX/dW paths loop
  experts with cublasGemmEx on four streams. CUTLASS is SM80 forward-only with
  trans_b=False; variable-K dW is always cuBLAS. No exposed persistent mode,
  split-K override or workspace/algorithm selector. Not a per-tile launch loop.
  Counts of one nvjet family cannot conclusively identify dX/dW without traces.
- Lowest-risk option found in the existing torch 2.11 installation: native
  `F.grouped_mm` SM90 BF16 CUTLASS grouped kernel, also supporting variable-K dW.
  New `_grouped_gemm.py` implements opt-in `MOE_GMM_IMPLEMENTATION=torch`;
  default/unset/`extension` preserves the original extension path. Explicitly
  reject non-SM90, non-BF16 or unsupported widths rather than use torch's hidden
  slow fallback. Widths must be divisible by 8, E < 1024, rows fit int32.
- Forward consumes packed weights directly; dX uses W.mT and dW A.mT as views.
  These orientations support odd/empty/imbalanced segments with aligned row
  pointers, without padding. One device counts cumsum supplies offsets to both
  FCs and backward. Existing CPU counts transfer is deliberately retained for a
  controlled GEMM-only change. Both parameter layouts remain supported.
- New NVTX ranges `grouped_gemm.fc{1,2}.{forward,dx,dw}` use the existing capture
  flag. Extension profiling reproduces its raw backend calls; uncaptured
  extension execution uses original ops.gmm. No added phase synchronization.
- New `tools/benchmark_expert_gemm.py` fingerprints installed modules/binary,
  prints actual autograd source, and separately times/profiles forward/dX/dW
  at actual balanced geometry or a valid maximally imbalanced routing. Timings
  use device-wide boundaries including auxiliary streams. The existing training
  benchmark is still the end-to-end measure. Trainer prints implementation and
  includes the new source in snapshots; checkpoint structure is unchanged.
- Expected native launches per phase/FC for 192 layer calls: 192 grouped GEMMs
  + 192 descriptor kernels + any constant initialization work, versus 12,288
  per-expert cuBLAS calls. NOT YET MEASURED. Geometry/tuning/reduction-order
  differences may affect both performance and numerics. E8 regression testing
  is required. Do not enable the candidate by default before Vista validation.
- Local full suite: 181 passed, 143 skipped (CUDA unavailable); targeted CUDA
  checks skipped rather than executed. CPU tests exercise torch's real grouped-mm
  CPU fallback for the offset/transpose contract, and cover expressions/gradients,
  integration, unchanged state keys, phase ranges, frozen inputs and selection.
  CUDA tests cover actual D768 E8/H1536 and E64/H384, odd balanced/imbalanced/
  empty counts, FP32-reference outputs/dX/dW, full packed MoE/router/all expert
  gradients with FP32/BF16 masters, and exact binary-product sums. Existing
  BF16 tolerances/dtype checks retained. Mac has torch2.11 CPU and no extension.
  `git diff --check` passed. Installed Vista binary/build flags, CUDA parity,
  counts and speed are pending.
- Changed: `_grouped_gemm.py` (new), model.py, train.py, test_package.py,
  tests/test_native_grouped_gemm.py (new), tools/benchmark_expert_gemm.py (new),
  docs/grouped_gemm.md and this handoff. Bias/combine/optim/config/dependency
  files untouched. Keep the same implementation for resume comparisons; this
  runtime execution selector is not a new checkpoint compatibility field.

Next: follow the complete validation/benchmark/Nsight sequence in
`docs/grouped_gemm.md`. On allocated Vista GH200, after modules/compiler setup:

```bash
cd "$WORK/projects/modded-nanogpt-moe"
module load nvidia/25.3 cuda/12.9
export CC=/usr/bin/gcc CXX=/usr/bin/g++ TB_ROOT=
unset MOE_GMM_IMPLEMENTATION
uv run --no-sync python -m pytest -q -rs tests
# Continue only after correctness passes; unset stale overrides per docs.
MOE_GMM_IMPLEMENTATION=extension scripts/vista/benchmark.sh configs/moe_e64k8_r0.5_packed.toml
MOE_GMM_IMPLEMENTATION=torch scripts/vista/benchmark.sh configs/moe_e64k8_r0.5_packed.toml
MOE_GMM_IMPLEMENTATION=extension scripts/vista/benchmark.sh configs/moe_grouped.toml
MOE_GMM_IMPLEMENTATION=torch scripts/vista/benchmark.sh configs/moe_grouped.toml
```

The sections below retain earlier implementation notes; historical pending
measurements are not claims about the newly reported user baseline.

## Fused combine (committed baseline)

- User-measured post-bias baseline: E64/K8 packed median 4214.6 ms/update;
  combine forward ~307.9 ms/update, backward ~275.7 ms/update, total ~583.6.
  These are baseline measurements, not results from this patch.
- Existing `order` maps sorted rows to flattened token/slot assignments. There
  is no existing reverse lookup. `_combine_assignment_rows` constructs one
  compact inverse from order ONCE inside combine (no argsort or packing change),
  then forward/backward reuse it. This extra metadata is necessary for the
  atomics-free gather design: [N*k] int64, 4 MiB for N=65536/K8, rather than a
  [N*k,D] activation (768 MiB in BF16 at D768). No redundant inversion is done.
- `_combine_forward` gathers sorted outputs per token/feature tile, multiplies
  weights and sums k in one kernel. `_combine_backward` handles four assignments
  per program, writes each grad_out_sorted row once, and reduces D for each
  routing-weight gradient. No atomics or full unsorted activation/gradient
  temporary; no host reads/synchronization or Python token/expert loops.
- Preserve eager numerics: round each BF16/FP16 product to input dtype BEFORE
  FP32 reduction, both for output and grad_topk_weights. Disable FP fusion so
  FP32 multiplication/reduction do not become FMA. Final reductions cast once.
  FP32 reduction order can still differ; do not promise bitwise random-sum
  parity. CUDA tests use FP64 bounds on rounded products, exact binary sums,
  and a cancellation case that detects incorrectly unrounded dot products.
- Expected launches per combine call: two forward (index lookup + fused sum)
  and one backward (both gradients). Across 192 calls: 384 forward + 192
  backward, versus the previous separate unsort/multiply/sum and backward
  gather/multiplies/reduction. Metadata traffic is included, not hidden in pack.
  Counts and speed remain pending actual GH200 measurement.
- `combine_expert_outputs` dispatches CUDA BF16/FP16/FP32 to lazy Triton; CPU
  and float64 use the eager reference. Higher-order backward retains a
  differentiable PyTorch fallback. Both layouts and moe.combine/moe_bw.combine
  are preserved. Bias kernels, routing, GEMM, optimizers and state_dict unchanged.
- Full local suite: `163 passed, 126 skipped`; `git diff --check` passed.
  CPU reference diagnostic harness passed. CPU checks include exact fallback
  parity, full MoE/router/expert gradients, rounding regression, source logging,
  and mocked launch/allocation plumbing. New CUDA checks cover E8/K2, K4,
  E64/K8, K1/K3, balanced/imbalanced routing, BF16/FP16/FP32, tails, strides,
  empty inputs, frozen gradients, and actual kernel counts. Mac has no CUDA or
  Triton; the new GPU kernels have NOT been compiled/executed here. Existing
  model tolerances are unchanged. The new higher-order fallback test allows
  only 32 FP64 epsilons for addition-order roundoff (~4e-16 observed locally).

Run on an allocated Vista GH200; tests must pass before timing or profiling:

```bash
cd "$WORK/projects/modded-nanogpt-moe"
module load nvidia/25.3 cuda/12.9
export CC=/usr/bin/gcc CXX=/usr/bin/g++
uv run --no-sync python -m pytest -q -rs tests
# Continue only after tests pass. Use a clean benchmark environment.
unset TRAINING_BENCHMARK NSYS_PROFILE NSYS_WARMUP_STEPS NSYS_ACTIVE_STEPS
unset SEED_OVERRIDE MBS_OVERRIDE TRAIN_STEPS_OVERRIDE MLP_TYPE_OVERRIDE
unset MLP_RATIO_OVERRIDE NUM_EXPERTS_OVERRIDE TOP_K_OVERRIDE MOE_BACKEND_OVERRIDE
unset CHECKPOINT_DIR CHECKPOINT_INTERVAL RESUME_CHECKPOINT STOP_AFTER_COMPLETED_UPDATES
unset REPRO_DIAGNOSTICS_DIR BENCHMARK_WARMUP_UPDATES BENCHMARK_MEASURED_UPDATES
export TB_ROOT=
set -o pipefail
mkdir -p "$STOCKYARD/logs/modded-nanogpt-moe/vista" "$STOCKYARD/profiles/modded-nanogpt-moe/vista"
stamp="$(date +%Y%m%d-%H%M%S)-$$"
scripts/vista/benchmark.sh configs/moe_e64k8_r0.5_packed.toml \
  2>&1 | tee "$STOCKYARD/logs/modded-nanogpt-moe/vista/triton-combine-$stamp.log"
# Separate capture: complete optimizer updates 11 and 12, not benchmark timing.
report="$STOCKYARD/profiles/modded-nanogpt-moe/vista/triton-combine-$stamp"
TRAIN_STEPS_OVERRIDE=14 NSYS_PROFILE=1 NSYS_WARMUP_STEPS=10 NSYS_ACTIVE_STEPS=2 \
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop -o "$report" \
  uv run --no-sync torchrun --standalone --nproc_per_node=1 \
  --module modded_nanogpt_moe.train --config configs/moe_e64k8_r0.5_packed.toml \
  2>&1 | tee "$report.log"
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_kern_sum \
  --format csv "$report.nsys-rep" | tee "$report.stats.csv"
grep -E '_combine_|moe.combine|moe_bw.combine' "$report.stats.csv"
```

Inspect both kernel counts and summed GPU time in moe.combine/moe_bw.combine.
Expect 192 instances of each of the three combine kernels. Compare against
the ~583.6 ms/update baseline without adding overlapping operator/kernel times.
Gather locality, scatter-store efficiency, and register pressure remain possible
bottlenecks; tiles have not been GH200-tuned. No speedup is claimed yet.

## Packed-layout experiment

- `model.moe_parameter_layout` defaults to `"modulelist"`; `"packed"` requires
  grouped-GEMM MoE. Packed weights are `[E,D,H]` / `[E,H,D]`, biases `[E,H]` /
  `[E,D]`. Forward reads these directly (retaining FP32-master to activation-
  dtype casts); it does not stack or transpose/contiguous-reconstruct weights.
  Routing, segmented bias addition, ReLU², compilation and both sets of NVTX
  ranges are unchanged. The existing `moe.stack_params` range covers dtype
  preparation in the packed path for comparison.
- Constructor and training initialization preserve the original expert-wise
  random draws, scales, zero projections/biases, and RNG order. Trainer
  initialization is now shared in `initialize_model_parameters`.
- Packed biases stay in AdamW despite being 2D. Packed weights use Muon as E
  independent matrices, including independent momentum slices, unchanged
  hyperparameters and 12 NS iterations. Muon must use the original `[out,in]`
  orientation: applying its aspect-ratio scale to native packed axes is wrong.
  It uses contiguous transposed gradient/momentum workspaces per packed
  parameter and copies momentum back. This avoids cross-layer stacking but
  retains optimizer copies deliberately; initial strided-view tests showed a
  maximum parameter discrepancy of 4.8831e-5 in the first failing tensor.
  Canonical workspaces made the two-update CPU comparisons exact, without
  loosening tolerances. Benchmark the net effect; no speedup is established.
- Default model keys, optimizer ordering, and checkpoint format/config remain
  unchanged. Only packed checkpoint configs add the layout key, so cross-layout
  resume fails compatibility checks. Rebuild optimizers with `build_optimizers`
  to restore packed orientation metadata; there is no checkpoint converter.
- The packed E64/K8 config differs from `moe_e64k8_r0.5.toml` only in run name
  and layout. D=768, H=384, global batch=524288, microbatch=64, horizon=3500.

Pending Vista validation, from the repository root on an allocated GH200:

```bash
cd "$WORK/projects/modded-nanogpt-moe"
module load nvidia/25.3 cuda/12.9
export CC=/usr/bin/gcc CXX=/usr/bin/g++
uv run --no-sync python -m pytest -q -rs tests
# Only after correctness passes; run sequentially on the same allocation.
scripts/vista/benchmark.sh configs/moe_e64k8_r0.5.toml
scripts/vista/benchmark.sh configs/moe_e64k8_r0.5_packed.toml
```

The benchmark wrapper uses the real trainer, 10 warmup + 30 measured updates,
and disables TensorBoard/artifacts. Unset experiment/checkpoint/Nsight overrides
that the wrapper rejects. Profile separately to compare fc1/fc2 backward and
optimizer work; the reported 0.54 s vs 3.04 s FC backward figures came from the
user's prior E8/K2 versus E64/K8 ModuleList traces, not this patch.

The cleanup makes `modded_nanogpt_moe` the only active trainer implementation
and removes upstream trainers, historical records, old kernels/evaluation
scripts, and generated artifacts from the working tree. Their history remains
available in Git and on the earlier branches.

## Implemented state

- The package contains model/MoE, optimizers, data loading, checkpointing,
  TOML configuration, and training in separate focused modules.
- `configs/dense_baseline.toml` defines D=768, ratio=4 dense training;
  `configs/moe_grouped.toml` defines E=8, k=2, ratio=2 grouped MoE. Both keep
  the 524288-token global batch.
- `configs/moe_e64k8_r0.5.toml` adds a controlled OLMoE-style geometry:
  D=768, E=64, k=8, and a 384-wide expert FFN (`mlp_ratio=0.5`). It retains
  selected-probability normalization, grouped GEMM, and the checkpoint-confirmed
  E8/K2 reference settings, including its 3500-step schedule horizon. The older
  checked-in `configs/moe_grouped.toml` remains unchanged at 3250 steps.
- Dense uses whole-model compilation. MoE uses an eager transformer prefix and
  independently compiled head/loss to avoid the prior softcap OOM.
- Default grouped MoE preserves the expert `ModuleList`, dropless routing,
  `trans_b=False`, and segmented bias addition that avoids indexing backward.
- Checkpoint/resume includes model, both optimizers, loader cursor, schedule,
  counters, RNG, timing, identity, and metadata with atomic latest/previous
  rotation. The user confirmed an end-to-end Vista resume run worked.
- TensorBoard uses config `run_name`; new runs reject existing event
  directories. Optional Nsight and exact reproducibility diagnostics remain.
- The Vista launcher now derives the repository root from its script location
  and uses it as `DATA_ROOT` only when no explicit value is supplied. This
  matches Vista's repository-local `data/fineweb10B` symlink into SCRATCH and
  remains portable to other checkouts. The logical shard patterns are
  unchanged. Launchers default `TB_ROOT` from Stockyard when unset; explicit
  `TB_ROOT=` disables logging.
- Startup output now records resolved runtime paths, matched shards, output
  destinations, Git state, and environment metadata; checkpoints retain this
  as metadata without making absolute paths compatibility requirements.
- TensorBoard is now declared in `pyproject.toml`/`uv.lock`. The grouped-GEMM
  extension remains external and architecture-specific.
- `scripts/vista/benchmark.sh` opts the normal trainer into a single-GPU,
  single-trial benchmark. It uses the existing complete optimizer-update loop,
  defaults to 10 warmup plus 30 measured updates, synchronizes each measured
  update, resets peak memory after warmup, and excludes validation and durable
  run artifacts.
- Muon now batches the local rank's same-shaped matrices through one batched
  Newton--Schulz call per shape while keeping momentum state per parameter and
  retaining the existing distributed ownership and all-gather order.

## Verification and experiments

- Packed layout: local full suite `79 passed, 12 skipped`; CUDA is unavailable
  on this Mac. `git diff --check` passed. CPU checks cover E8/K2 and E64/K8:
  exact constructor/training initialization and RNG, outputs, input/router/all
  expert gradients (including BF16 activations and empty-expert zeros), two
  AdamW/Muon steps with slice-wise state checks, packed checkpoint restoration,
  old baseline checkpoints/grouping, and balanced capture on/off NVTX ranges.
  A CPU simulation covers two-rank ownership/gather ordering, not real NCCL.
  Skipped tests cover real CUDA grouped GEMM at D=768/H=1536 or H=384 with
  FP32/BF16 masters, CUDA initialization, and CUDA optimizer parity. Existing
  CUDA tolerance/dtype checks are unchanged; CPU and optimizer assertions are
  exact. No GH200 speedup or end-to-end packed resume claim is made.
- Backward instrumentation: full local suite `60 passed, 2 skipped` (CUDA
  unavailable); `git diff --check` passed. Tests verify exact outputs and all
  gradients with capture on/off, no inactive/no-grad hook installation, reverse
  boundary order, and balanced ranges on full, repeated, partial, and frozen-input
  traversals. Hooks bracket `out -> out_sorted -> h_act -> h_pre -> x_sorted`
  with `moe_bw.combine`, `moe_bw.fc2`, `moe_bw.activation`, `moe_bw.fc1`.
  A queued autograd completion callback closes a range if a partial grad()
  traversal omits the last hook. This uses the existing single-device backward
  worker; no added synchronization, transfers, or changed math. These are host
  enqueue intervals, potentially including interleaved router/parameter-gradient
  work, not exclusive GPU kernel durations. Inspect them in a fresh Vista trace.
- Trainer NVTX integration fix: `8 passed` focused; full suite `56 passed,
  2 skipped` (CUDA unavailable); `git diff --check` passed. Tests check updates
  11/12, disabled profiling, normal/early stops, and flag reset when synchronize
  or profiler.stop raises. Rerun the Vista capture and inspect all seven moe.*
  ranges after the complete fix is committed and deployed.
- Grouped-MoE NVTX ranges cover router/top-k, packing (including CPU counts),
  parameter stacking, FC1, activation, FC2, and combine, gated by the trainer's
  existing Nsight capture boundaries. No computations or synchronization were
  added. A CPU GEMM-stub test verifies exact output/gradient equality with
  markers on/off and no NVTX calls outside capture. Full suite: `48 passed,
  2 skipped` (CUDA unavailable). Actual Vista trace inspection remains pending.
- For shape-batched Muon: focused CPU correctness tests passed; the full local
  suite passed with `47 passed, 2 skipped`. Momentum matched the old update
  exactly; parameters matched within `2e-4`, approximately one BF16 update ULP
  at the tested learning rate. Vista correctness and speed remain pending.
- For the training benchmark: focused CPU-safe benchmark/launcher tests passed;
  the full local suite passed with `46 passed, 2 skipped`. The skipped checks
  require CUDA and `grouped_gemm`; Vista GH200 timing remains pending.
- For the new E=64, k=8 config: focused config/launcher tests `4 passed`; full
  local suite `44 passed, 2 skipped`. The skipped checks require CUDA and
  `grouped_gemm`. The Vista smoke and full experiment remain pending.
- After this fix: focused config/launcher tests `3 passed`; full local suite
  `43 passed, 2 skipped`. The skipped tests require CUDA and `grouped_gemm`.
- `bash -n scripts/ls6/train.sh scripts/vista/train.sh`: passed after the fix.
- Package Python compilation: passed.
- `git diff --check` passed before cleanup was committed.
- Vista GH200 inspection verified that `data/fineweb10B` points to the active
  FineWeb10B directory on SCRATCH and contains validation and training shards.
- CPU-safe launcher tests cover repository-root defaulting from an outside
  working directory and preservation of an explicit `DATA_ROOT`.
- The user confirmed the full post-fix pytest suite passed on a Vista GH200.
- The user confirmed both prescribed two-step Vista training smokes completed
  successfully with TensorBoard disabled: dense ratio=4 at microbatch 32 and
  grouped MoE E=8, k=2, ratio=2 at microbatch 64. Logs are under
  `$STOCKYARD/logs/modded-nanogpt-moe/vista/cleanup_{dense,grouped}_smoke.log`.
- The standalone grouped-GEMM BF16 harness previously passed all eight LS6
  A100 cases, including empty experts, with a CUTLASS GemmGrouped kernel.
- Integrated grouped E=1,k=1 training previously completed six steps on LS6
  with final validation loss 6.79716. Later project use established working
  dense and grouped-MoE training on LS6 and Vista.
- No remote job, active checkpoint path, or experiment output was supplied for
  the cleanup commit itself.

## Open issues and cautions

- The cleanup branch still needs a post-cleanup LS6 test/smoke validation. The
  Vista launcher fix does not change LS6's existing data-root convention.
- Two fresh Vista dense runs with seed 1234 previously matched exactly at
  initialization, the first batch, and after update 1, but differed in model
  and optimizer state by update 10 while RNG and loader state matched. Update-2
  snapshot hooks were added, but no concluding comparison result is recorded.
  This is separate from the confirmed checkpoint/resume workflow.
- The loop backend can leave `.grad is None` for empty experts, while grouped
  execution supplies exact zero gradients. Active multi-expert training uses
  grouped execution; do not erase this semantic distinction in tests.
- GH200 uses the extension's cuBLAS fallback with the pinned build. CUTLASS
  availability and relative speed remain architecture-specific.
- The GitHub default branch is still `moe-baseline`; do not change defaults or
  remove milestone branches until the cleanup branch is validated and reviewed.

## Next steps

1. Review the packed-layout diff, then run the Vista correctness/benchmark
   commands above. Verify actual multi-rank behavior separately if needed.
   Compare the two E64/K8 layouts before drawing conclusions about the original
   E8/K2 versus E64/K8 bottleneck. Preserve console output and profile separately.
2. Run a 10--20 update smoke test of `configs/moe_e64k8_r0.5.toml` on an
   existing Vista idev GH200 node using `TRAIN_STEPS_OVERRIDE`; this preserves
   the config's full-run schedule outside the smoke invocation.
3. If the smoke succeeds, submit the full experiment through
   `scripts/vista/submit.sh`.
4. Validate the cleanup branch on LS6, including its actual data path, full
   pytest suite, and short dense/grouped smokes.
5. Recheck checkpoint creation/resume and comparison from the cleaned paths
   when checkpoint validation is next in scope.
6. If exact fresh-run repeatability is required, compare the saved update-2
   diagnostic stages before extending instrumentation further.
7. After LS6 validation, decide whether to make `cleanup-active-codebase` the
   repository default branch; retain milestone branches for provenance.
