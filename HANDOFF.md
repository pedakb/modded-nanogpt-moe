# Current handoff

Updated: 2026-09-22

## Current goal and state

Current base on `cleanup-active-codebase`: `761c8cc` (Slurm submission options),
initially clean. Uncommitted task: **Grad-EM Stage 1 only**, mathematical reference,
config and CPU tests. No kernel/model/optimizer changes, dependency installation,
training, submission, commit or push.

## Grad-EM reference (current work)

- `grad_em.py::grad_em_reference` computes detached FP32 v, q, q_tilde,
  selected expert gradients q*g, and router gradients q_tilde-p. The sign is
  frozen. Uses selected logits directly, never log(top-k weights). g=0 still
  generally gives nonzero router gradients; eta=0 is not ordinary backward.
- Config defaults: `model.moe_backward="standard"`, `model.grad_em_eta=0.1`.
  Eta must be finite/nonnegative. Grad-EM requires MoE; the trainer explicitly
  rejects it before CUDA setup because production integration is not done.
  Production TOMLs unchanged. Resolved checkpoint metadata records both fields;
  missing legacy fields mean standard/0.1, while explicit differences reject
  resume. No state keys or format version change.
- Tests: `tests/test_grad_em.py`: **28 passed**; together with the existing
  checkpoint-resume suite: **37 passed**, all CPU. `tests/test_package.py`:
  **17 passed, 10 failed** (four existing checkpoint250-vs100 expectations and
  six launcher failures on Bash3.2). No skips in these runs. `git diff --check`
  passed. CUDA checks intentionally not run in Stage 1.
- Dirty files: config.py, checkpoint.py, train.py, new grad_em.py,
  tests/test_package.py, new tests/test_grad_em.py, README.md, HANDOFF.md,
  new docs/grad_em.md. See the new document for the contract and future boundary.
- Next: review Stage 1; only in a separately authorized stage integrate a
  custom combine backward with full logits/support, q*g to the existing expert
  graph, and q_tilde-p directly to logits. Do not change GEMM or standard combine.
  No Grad-EM CUDA correctness/performance claim or active run/checkpoint exists.

Follow-up: user confirmed checkpoint cadence **250 completed updates** for all
three production configs (already set in TOML). Updated stale package-test
expectations and README/handoff text; runtime defaults and overrides unchanged.
Follow-up validation: all four formerly failing config tests passed. Full
package suite: **21 passed, 6 failed**, all six due to Bash3.2 launcher support.
`git diff --check` passed. No training or CUDA checks run.
Committed Slurm options remain documented in README/AGENTS. Full launcher tests
require Bash4+ (this Mac has3.2); that unrelated limitation remains.

## Production config standardization (prior work)

- Dense ratio 4; E8/K2 ratio 2; E64/K8 ratio 0.5. Both MoE configs now use
  `grouped_gemm` + `packed` + selected-probability normalization. E8 is now
  `configs/moe_e8k2_r2.toml`, run name `moe-e8k2-r2`; repository references
  updated. Dense omits MoE-only fields and resolves their existing defaults.
- Shared D=768/L=12/vocab=50304, microbatch=64, global batch=524288, sequence=1024,
  horizon=3250, cooldown=0.7, seed=1234, one trial, identical optimizers/data
  patterns, diagnostics=25 (no histograms/Nsight), and checkpoint=250 plus final
  save. `[evaluation]` now owns 10485760 validation tokens and cadence 125,
  switching to 25 for the final 10%; step 0 and the final step are always due.
  This schedule is independent of diagnostics cadence.
- The earlier E64 horizon of 3500 intentionally matched an older checkpoint-confirmed
  E8 reference. This new suite explicitly supersedes that comparison with 3250;
  all three configs use checkpoint cadence 250. Preserve original configs for older
  resumes: ModuleList E8 and 3500-step checkpoints are not compatible with these
  production settings. Existing TensorBoard run directories are not migrated.
- Evaluation-focused tests cover parsing, validation, the exact 3250-update
  schedule, transition/final boundaries, restored-step behavior, rejection of
  the retired training field, and independence from diagnostics cadence: **14
  passed**.
  Requested package tests pass (22); packed-expert tests pass (29) with the
  documented Vista `CC=/usr/bin/gcc CXX=/usr/bin/g++` environment. The initial
  packed test invocation inherited `nvc++` and failed three Inductor CPU
  compilations before passing cleanly with GCC/G++.
  `git diff --check` passes.

Next: review the config diff and use the README three-config `--submit` command
on Vista only when ready. Check for existing run-name directories first.

## Compact production TensorBoard diagnostics (final design)

- The standard production run has exactly **51 scalar series**: 8 ordinary
  loss/performance/LR series, 4 global optimization series, and 13 heavy series
  for each of three representative layers. No legacy aliases or histogram
  series are emitted; `perf/train_s` was removed.
- `metric/loss/train` is logged after every completed optimizer update.
  `metric/loss/val`, `perf/step_ms`, `perf/tok_s`, the three AdamW LR groups,
  and the Muon LR retain their existing trainer write points and meanings.
- Heavy diagnostics follow completed-update `diagnostics.scalar_interval`
  cadence. Production uses `scalar_interval = 25`, `histogram_interval = 0`,
  and `during_nsys = false`.
- Global optimization health is
  `opt/global/{param_rms,grad_rms,update_rms,update_ratio}` over all optimized
  trainable parameters. RMS values use true parameter-count weighting, and
  update quantities use actual post-optimizer stored parameter changes.
- Heavy per-layer diagnostics are restricted to `l00`, `l05`, and `l11`.
  Router behavior is `entropy_norm`, `topk_margin_med`, `load/cv`, and
  `load/zero`. Router optimization is `param_rms`, `grad_rms`, `update_rms`,
  `update_ratio`, and `dlogit_rms`.
- Expert diagnostics combine each expert's FC1 and FC2 weight squared sums and
  element counts before summarizing across experts. The retained series are
  `param_rms_med`, `grad_rms_med`, `update_rms_med`, and `update_ratio_med`.
  Packed and ModuleList implementations remain supported by the observer.
- Removed work includes diagnostics for the other nine layers; separate FC1/FC2
  series; raw norms and grad ratios; min/max/mean/std/p10/p90; router-vs-expert
  comparisons; attention/embedding/head groups; discarded router behavior; and
  all histograms. Sampled updates still require one full-model parameter snapshot
  and full-model reductions to measure exact global update metrics. Representative
  groups reuse that snapshot rather than cloning parameters again.
- Benchmark mode still bypasses diagnostics completely. Nsight still bypasses
  diagnostics for the whole run unless `during_nsys=true` explicitly opts in.
  Rank, TensorBoard writer, checkpoint, resume, run-name, model, routing, and
  optimizer semantics are unchanged.

## Production E64/K8 configuration

- The production experiment is `configs/moe_e64k8_r0.5.toml`, with run name
  `moe-e64k8-r0.5`, D=768, E=64, K=8, `mlp_ratio=0.5`, 3250 updates, grouped
  GEMM execution, and packed expert parameters. Packed storage is an
  implementation detail and is no longer part of the scientific config/run name.
- Vista production training uses `MOE_GMM_IMPLEMENTATION=torch`, scoped to the
  actual torchrun process by the launchers. Submit the full run with
  `scripts/vista/train.sh --submit configs/moe_e64k8_r0.5.toml`.
  **Do not globally export `MOE_GMM_IMPLEMENTATION=torch` before pytest**: CPU
  unit tests that exercise the extension/fallback contract will fail under that
  global override.
- `configs/moe_e8k2_r2.toml` uses the same packed backend and shared production
  settings; only run name, ratio, expert count and top-k differ from E64/K8.

## Vista execution workflow

- `source scripts/vista/env.sh` establishes only common machine state: modules,
  system GCC/G++, user-local PATH, and Vista TensorBoard root/system.
- Run tests after sourcing with `uv run --no-sync python -m pytest -q`. The
  environment script deliberately does not select a grouped-GEMM implementation.
- `scripts/vista/train.sh` is the only Vista training entry point. Interactive
  commands are `train.sh --steps 30 CONFIG`, `train.sh --steps 100 CONFIG`, and
  `train.sh CONFIG`; `train.sh --submit CONFIG [CONFIG ...]` submits the same
  script into a non-recursive worker mode. It retains the established
  GH/one-node/one-task/six-hour default, validates the ordered config list,
  runs each config in a separate process, stops on failure, and writes one
  persistent Stockyard suite log. `--submit --time SLURM_TIME` overrides the
  time request. `--steps` is current-node only and disables TensorBoard and
  TOML checkpoint policy so smoke runs cannot overwrite production artifacts.
- The launcher defaults to the E64/K8 config when omitted, scopes the torch
  grouped-GEMM implementation to each torchrun process, and clears stale
  parent-shell run controls. The obsolete submission wrapper and separate
  Slurm worker were removed.
- Checkpoint cadence is configured by `[checkpoint].interval`; omission disables
  checkpointing, 0 saves only at completion, and positive values add periodic
  saves. All three production configs use 250. `--checkpoint-interval N` overrides every
  config in one invocation. Vista derives a separate Stockyard directory from
  each TOML `run_name`; `--resume PATH` resumes only the first supplied config.
  Checkpoint/resume still requires one trial, preventing file collisions.
  Atomic `latest.pt`/`previous.pt` rotation and restore semantics are unchanged.

## Final validation and pending work

- Full Vista test suite: **357 passed, 1 skipped**.
- A 30-update TensorBoard smoke test completed successfully with the compact
  diagnostics enabled.
- The unified Vista launcher/checkpoint refactor passes `bash -n` and
  `git diff --check`. Focused validation passed: `tests/test_package.py` has
  **20 passed**, and the config-focused packed-expert selection has **4 passed**.
- CUDA TensorBoard event-file inspection and the final diagnostics overhead
  benchmark remain pending. Do not infer final runtime overhead from tests or
  the smoke run.

## Prior sampled diagnostics implementation (historical; superseded)

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
  microbatches, so common logit shifts do not change it. `dlogit_rms` uses
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
- `metric/loss/train` is sampled rank-local cross entropy per token;
  `metric/loss/val` logs validation loss. Writer cleanup/purge unchanged. Scalars
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
- Next: inspect the canonical tags and sampled overhead on Vista with
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
MOE_GMM_IMPLEMENTATION=extension scripts/vista/benchmark.sh configs/moe_e8k2_r2.toml
MOE_GMM_IMPLEMENTATION=torch scripts/vista/benchmark.sh configs/moe_e8k2_r2.toml
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
unset CHECKPOINT_DIR CHECKPOINT_INTERVAL CHECKPOINT_ROOT RESUME_CHECKPOINT STOP_AFTER_COMPLETED_UPDATES
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
- Historical packed-layout experiment used D=768, H=384, global batch=524288,
  microbatch=64, horizon=3500. Its separate packed config has since been removed;
  the current production E64 config itself uses packed storage and horizon 3250.

Current geometry comparison, from the repository root on an allocated GH200:

```bash
cd "$WORK/projects/modded-nanogpt-moe"
module load nvidia/25.3 cuda/12.9
export CC=/usr/bin/gcc CXX=/usr/bin/g++
uv run --no-sync python -m pytest -q -rs tests
# Only after correctness passes; run sequentially on the same allocation.
scripts/vista/benchmark.sh configs/moe_e64k8_r0.5.toml
scripts/vista/benchmark.sh configs/moe_e8k2_r2.toml
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
  `configs/moe_e8k2_r2.toml` defines E=8, k=2, ratio=2 grouped MoE. Both keep
  the 524288-token global batch.
- `configs/moe_e64k8_r0.5.toml` adds a controlled OLMoE-style geometry:
  D=768, E=64, k=8, and a 384-wide expert FFN (`mlp_ratio=0.5`). It retains
  selected-probability normalization and grouped GEMM. The historical 3500-step
  checkpoint reference is superseded by the shared 3250-step production suite;
  both MoE configs now use packed storage.
- Dense uses whole-model compilation. MoE uses an eager transformer prefix and
  independently compiled head/loss to avoid the prior softcap OOM.
- Default grouped MoE preserves the expert `ModuleList`, dropless routing,
  `trans_b=False`, and segmented bias addition that avoids indexing backward.
- Checkpoint/resume includes model, both optimizers, loader cursor, schedule,
  counters, RNG, timing, identity, and metadata with atomic latest/previous
  rotation. The user confirmed an end-to-end Vista resume run worked.
- TensorBoard uses config `run_name`; new runs reject existing event
  directories. Optional Nsight and exact reproducibility diagnostics remain.
- The Vista launchers derive the repository root from their script locations
  and use it as `DATA_ROOT`. This matches Vista's repository-local
  `data/fineweb10B` symlink into SCRATCH and remains portable to other
  checkouts. The logical shard patterns are unchanged. `env.sh` selects the
  Stockyard TensorBoard root; step-limited and benchmark processes disable it.
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
2. Use `scripts/vista/train.sh --steps N configs/moe_e64k8_r0.5.toml` for any
   further short validation on an existing Vista idev GH200 node; do not queue
   an sbatch smoke.
3. Submit the full experiment with
   `scripts/vista/train.sh --submit configs/moe_e64k8_r0.5.toml`.
4. Validate the cleanup branch on LS6, including its actual data path, full
   pytest suite, and short dense/grouped smokes.
5. Recheck checkpoint creation/resume and comparison from the cleaned paths
   when checkpoint validation is next in scope.
6. If exact fresh-run repeatability is required, compare the saved update-2
   diagnostic stages before extending instrumentation further.
7. After LS6 validation, decide whether to make `cleanup-active-codebase` the
   repository default branch; retain milestone branches for provenance.
