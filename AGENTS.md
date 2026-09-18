# Agent Guide

## Project purpose

This repository is the active research codebase for comparing the modded-
nanoGPT dense baseline with a sparse, dropless, top-k mixture-of-experts model.
The primary systems target is grouped-GEMM expert execution on LS6 A100 GPUs,
with Vista GH200 support. Experiments should remain directly comparable:
preserve the training objective, data stream, optimizer mathematics, schedule,
and validation semantics unless a task explicitly changes them.

Read `README.md` for user-facing commands and `HANDOFF.md` before continuing
unfinished work. Always inspect the current branch, `git status`, and the diff
before editing; the worktree may contain another agent's changes.

## Active layout

- `modded_nanogpt_moe/model.py`: attention, dense MLP, MoE, blocks, and GPT.
- `modded_nanogpt_moe/optim.py`: Muon and optimizer construction.
- `modded_nanogpt_moe/data.py`: FineWeb shard loading and serializable cursor.
- `modded_nanogpt_moe/checkpoint.py`: atomic checkpoints and restoration.
- `modded_nanogpt_moe/config.py`: built-in defaults, TOML loading, validation,
  and compatibility environment overrides.
- `modded_nanogpt_moe/train.py`: training, validation, compilation, logging,
  checkpointing, reproducibility diagnostics, and Nsight markers.
- `configs/`: portable experiment definitions. `run_name` identifies a run.
- `tests/`: package, model/MoE parity, and checkpoint tests.
- `tools/`: grouped-GEMM validation/profiling and checkpoint comparison.
- `scripts/`: data download and LS6/Vista launchers.
- `docs/grouped_gemm.md`: architecture-specific extension notes.

The canonical entry point is the package module; do not recreate a second
trainer implementation or a legacy wrapper unless compatibility requires it.

## Scientific and implementation invariants

- Dense and expert feed-forward layers use the same `MLP` implementation,
  including biases and ReLU-squared. The default expansion ratio is 4.
- Resolve `hidden_dim = model_dim * mlp_ratio` exactly. Reject invalid or
  fractional products; never silently truncate or divide width by `top_k`.
- Keep `mlp_ratio`, `num_experts`, and `top_k` independent.
- Preserve model parameter names, order, initialization, and checkpoint keys
  unless a migration is explicitly requested. Experts remain an
  `nn.ModuleList` for both MoE backends.
- Routing uses FP32 softmax followed by top-k selection and optional top-k
  renormalization. Execution is dropless: no capacity limit, padding, or token
  dropping. Preserve router, expert-weight, bias, input-gradient, and empty-
  expert behavior in parity tests.
- The production grouped backend uses `trans_b=False`, stacks the existing
  expert parameters differentiably, adds biases by contiguous expert segments,
  and combines every routed assignment. Do not add an E=1 shortcut.
- Dense training compiles the complete model. MoE keeps the transformer prefix
  eager and independently compiles the head/loss. This boundary prevents the
  softcap/head-loss OOM and must not be casually changed.
- Preserve the summed cross-entropy, logit softcap, gradient accumulation,
  global-token batch interpretation, validation token selection, and learning-
  rate schedule.
- Preserve optimizer membership and order: AdamW handles embedding, head, and
  parameters with fewer than two dimensions; Muon handles block parameters
  with at least two dimensions. Muon momentum is persistent checkpoint state.
- Attention defaults to `head_dim=128` and must reject configurations that
  produce zero heads before scaled-dot-product attention executes.
- Checkpoints are written only after completed optimizer updates with cleared
  gradients. They include unwrapped model state, both optimizers, schedule and
  counters, relocatable loader state, RNG state, timing, run identity, and
  environment metadata. Saves atomically rotate `latest.pt` to `previous.pt`.
- Checkpoint/resume, reproducibility snapshots, and Nsight capture currently
  support one GPU and one trial. Reject unsupported combinations clearly.
- TensorBoard is rank-zero only. A new run refuses to reuse an existing
  `<TB_ROOT>/modded-nanogpt-moe/<TB_SYSTEM>/<run_name>` directory; resume uses
  the saved run identity and purges stale steps.

Do not combine structural work with optimizer changes, routing experiments, or
new evaluation/checkpoint features. Make behavior changes separately and back
them with focused parity tests and measurements.

## Configuration and storage

Experiment TOMLs describe scientific settings. Machine-specific roots remain
environment settings so the same config moves between systems. Existing
environment overrides take precedence over TOML values; keep them compatible.

- Shard patterns are a logical repository-relative interface:
  `data/fineweb10B/fineweb_train_*.bin` and
  `data/fineweb10B/fineweb_val_*.bin`. `DATA_ROOT` is the base prepended to
  those patterns, not the physical dataset directory itself.
- In the trainer, an empty/unset `TB_ROOT` disables TensorBoard and `TB_SYSTEM`
  labels the system. The cluster launchers provide a Stockyard default when
  `TB_ROOT` is unset; pass `TB_ROOT=` explicitly to disable it through them.
- `CHECKPOINT_DIR`, `CHECKPOINT_INTERVAL`, `RESUME_CHECKPOINT`, and
  `STOP_AFTER_COMPLETED_UPDATES` control opt-in checkpoint workflows.
- `REPRO_DIAGNOSTICS_DIR` enables exact-state snapshots.
- `NSYS_PROFILE`, `NSYS_WARMUP_STEPS`, and `NSYS_ACTIVE_STEPS` control capture.
- Smoke-test overrides include `SEED_OVERRIDE`, `MBS_OVERRIDE`,
  `TRAIN_STEPS_OVERRIDE`, `MLP_TYPE_OVERRIDE`, `MLP_RATIO_OVERRIDE`,
  `NUM_EXPERTS_OVERRIDE`, `TOP_K_OVERRIDE`, and `MOE_BACKEND_OVERRIDE`.

Keep durable TensorBoard/checkpoint outputs on Stockyard and active data on
system scratch. On the current Vista setup, repository-local
`data/fineweb10B` is a symlink to the physical FineWeb10B directory on SCRATCH,
and the Vista launcher defaults `DATA_ROOT` to its robustly derived repository
root. Other checkouts may provide their own directory/symlink or explicitly set
`DATA_ROOT`. Application code and committed configs must remain agnostic to the
physical target. Never commit user-specific absolute paths, datasets,
checkpoints, logs, or profiles.

## Environments and commands

Python is 3.11 and dependencies are managed with `uv`. Use `uv run --no-sync`
on provisioned cluster environments. Do not install CUDA packages on macOS.
The external `nv-grouped-gemm` package is deliberately excluded from the lock
file and must be built per architecture; follow `docs/grouped_gemm.md` rather
than rebuilding it speculatively.

Local tests from the repository root:

```bash
uv run --no-sync python -m pytest -q -rs tests
uv run --no-sync python -m tools.validate_grouped_gemm --help
```

Canonical single-GPU training:

```bash
uv run --no-sync torchrun --standalone --nproc_per_node=1 \
  --module modded_nanogpt_moe.train \
  --config configs/dense_baseline.toml
```

Cluster launchers load the established modules and set reusable defaults:

```bash
scripts/ls6/train.sh configs/moe_grouped.toml
scripts/vista/train.sh configs/moe_grouped.toml
```

- LS6: A100, `gcc/11.2.0`, CUDA 12.8; the validated grouped-GEMM build used
  `nv-grouped-gemm==1.1.4.post8` and device capability 80.
- Vista: GH200, `nvidia/25.3`, CUDA 12.9, `CC=/usr/bin/gcc`,
  `CXX=/usr/bin/g++`. The pinned extension has used its cuBLAS fallback here;
  do not infer CUTLASS portability or performance without profiling. Ordinary
  tests use `uv run --no-sync python -m pytest -q -rs tests`; do not re-sync an
  already provisioned environment during routine validation.

Run cluster commands from the repository root because data patterns are
relative. Use `grep`, not `rg`, in TACC-facing commands. When piping a run to
`tee`, preserve the training command's exit status with `set -o pipefail`.

## Verification and handoff discipline

- Use `apply_patch` for edits and preserve unrelated changes in a dirty tree.
- Prefer focused tests first, then the full local suite. Report passed, skipped,
  and pending CUDA checks separately; macOS cannot establish CUDA correctness.
- Do not loosen dtype checks or numerical tolerances to hide discrepancies.
- Do not claim performance improvements without synchronized measurements;
  keep profiler collection separate from benchmark timing.
- Do not commit, push, install dependencies, rebuild extensions, or launch
  remote jobs unless the user explicitly requests it.
- Preserve `LICENSE` and upstream attribution. Historical upstream content is
  available through Git history and the `master`/`moe-baseline` milestones.
- Before handing substantial unfinished work to another agent, update
  `HANDOFF.md` with the branch/commit, dirty files, tests, active runs or
  checkpoints, blockers, and concrete next steps. Keep it concise and remove
  stale details rather than accumulating a project diary.
