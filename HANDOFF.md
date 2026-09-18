# Current handoff

Updated: 2026-09-18

## Current goal and state

The current base on `cleanup-active-codebase` is `02706ae` (`Instrument MoE
backward for Nsight`). Pending, uncommitted work adds opt-in packed expert
parameters; no training or remote jobs were launched. Modified: model, config,
optimizers, trainer, existing package/MoE tests, AGENTS.md, and this handoff.
New: `tests/test_packed_experts.py` and `configs/moe_e64k8_r0.5_packed.toml`.
Dependencies and existing experiment configs are unchanged. GPU correctness,
packed resume in a real training job, and performance remain pending.

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
