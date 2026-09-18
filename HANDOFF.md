# Current handoff

Updated: 2026-09-18

## Current goal and state

The repository cleanup and agent guides are committed and pushed on
`cleanup-active-codebase`; the current base is `ec435b1` (`Add coding agent
guidance and handoff`). The current task fixes the Vista launcher's incorrect
data-root default discovered during validation on a GH200 compute node. The
launcher, tests, README, `AGENTS.md`, and this file are modified but uncommitted.

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
- Dense uses whole-model compilation. MoE uses an eager transformer prefix and
  independently compiled head/loss to avoid the prior softcap OOM.
- Grouped MoE preserves the expert `ModuleList`, dropless routing,
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

## Verification and experiments

- After this fix: focused config/launcher tests `3 passed`; full local suite
  `43 passed, 2 skipped`. The skipped tests require CUDA and `grouped_gemm`.
- `bash -n scripts/ls6/train.sh scripts/vista/train.sh`: passed after the fix.
- Package Python compilation: passed.
- `git diff --check` passed before cleanup was committed.
- Vista GH200 inspection verified that `data/fineweb10B` points to the active
  FineWeb10B directory on SCRATCH and contains validation and training shards.
- CPU-safe launcher tests cover repository-root defaulting from an outside
  working directory and preservation of an explicit `DATA_ROOT`.
- The standalone grouped-GEMM BF16 harness previously passed all eight LS6
  A100 cases, including empty experts, with a CUTLASS GemmGrouped kernel.
- Integrated grouped E=1,k=1 training previously completed six steps on LS6
  with final validation loss 6.79716. Later project use established working
  dense and grouped-MoE training on LS6 and Vista.
- No remote job, active checkpoint path, or experiment output was supplied for
  the cleanup commit itself.

## Open issues and cautions

- The full post-fix test suite and two-step dense/grouped smokes remain to be
  run on Vista. Local macOS tests cannot verify CUDA, compilation, or grouped-
  GEMM behavior.
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

1. Review this uncommitted diff, then commit and push only after approval.
2. Pull `cleanup-active-codebase` on Vista and run the full pytest sequence.
3. Only after pytest succeeds, run the two-step
   dense and E=8,k=2 grouped smokes with `TB_ROOT=`.
4. Validate LS6 separately; its data-root convention was not changed here.
5. Recheck checkpoint creation/resume and comparison from the cleaned paths.
6. If exact fresh-run repeatability is required, compare the saved update-2
   diagnostic stages before extending instrumentation further.
