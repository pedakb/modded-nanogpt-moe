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
- `modded_nanogpt_moe/train.py`: training, validation, compilation,
  actual-pipeline benchmarking, logging, checkpointing, reproducibility
  diagnostics, and Nsight markers.
- `configs/`: portable experiment definitions. `run_name` identifies a run.
- `tests/`: package, model/MoE parity, and checkpoint tests.
- `tools/`: grouped-GEMM validation/profiling and checkpoint comparison.
- `scripts/`: data download and LS6/Vista launchers.
- `docs/grouped_gemm.md`: architecture-specific extension notes.

The canonical entry point is the package module; do not recreate a second
trainer implementation or a legacy wrapper unless compatibility requires it.

## Scientific and implementation invariants

- Dense and default expert feed-forward layers use the same `MLP` implementation,
  including biases and ReLU-squared. Packed experts preserve this computation
  and initialization with different storage. The default expansion ratio is 4.
- Resolve `hidden_dim = model_dim * mlp_ratio` exactly. Reject invalid or
  fractional products; never silently truncate or divide width by `top_k`.
- Keep `mlp_ratio`, `num_experts`, and `top_k` independent.
- Preserve model parameter names, order, initialization, and checkpoint keys
  unless a migration is explicitly requested. Experts default to an
  `nn.ModuleList` for both MoE backends. Experimental
  `model.moe_parameter_layout = "packed"` requires grouped GEMM, stores native
  `[E,D,H]`/`[E,H,D]` weights, and has explicitly incompatible checkpoint keys.
- Routing uses FP32 softmax followed by top-k selection and optional top-k
  renormalization. Execution is dropless: no capacity limit, padding, or token
  dropping. Preserve router, expert-weight, bias, input-gradient, and empty-
  expert behavior in parity tests.
- The grouped backend uses `trans_b=False`, stacks ModuleList parameters
  differentiably (or reads packed parameters), adds biases by contiguous expert
  segments, and combines every routed assignment. Preserve activation-dtype
  casts with FP32 master parameters. Do not add an E=1 shortcut.
- Dense training compiles the complete model. MoE keeps the transformer prefix
  eager and independently compiles the head/loss. This boundary prevents the
  softcap/head-loss OOM and must not be casually changed.
- Preserve the summed cross-entropy, logit softcap, gradient accumulation,
  global-token batch interpretation, validation token selection, and learning-
  rate schedule.
- Preserve optimizer membership and order: AdamW handles embedding, head, and
  parameters with fewer than two dimensions; Muon handles block parameters
  with at least two dimensions. Exception: packed 2D biases stay in AdamW;
  packed 3D weights are E independent matrices under Muon, in the reference
  `[out,in]` orientation, including its aspect-ratio scale and 12 Newton--Schulz
  iterations. Muon momentum is persistent checkpoint state. Reconstruct packed
  orientation metadata through `build_optimizers` when loading a checkpoint.
  `optimizers.router_optimizer` defaults to `"muon"`; the `"adamw"` ablation
  moves only MoE router weights into existing AdamW group 2. Router biases
  remain there in both modes. Ownership is exclusive; legacy checkpoint
  metadata defaults to Muon, and changing ownership on resume is rejected.
- Attention defaults to `head_dim=128` and must reject configurations that
  produce zero heads before scaled-dot-product attention executes.
- Checkpoints are written only after completed optimizer updates with cleared
  gradients. They include unwrapped model state, both optimizers, schedule and
  counters, relocatable loader state, RNG state, timing, run identity, and
  environment metadata. Each save retains `step_NNNNNN.pt`; `latest.pt` and
  `previous.pt` are atomically updated hard-link aliases, avoiding duplicate
  multi-GB payloads while keeping the existing resume paths.
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
  labels the system. Vista `env.sh` sets the shared Stockyard root; its
  step-limited and benchmark launch paths scope `TB_ROOT=` to the trainer.
- `[evaluation]` owns validation token count and regular/final-phase cadence.
  It is independent of `[diagnostics].scalar_interval`, which controls sampled
  training diagnostics only. Evaluation always includes step 0 and the final
  completed update.
- `[checkpoint].interval` is the portable checkpoint policy; omission disables
  it, 0 saves only at completion, and a positive value adds periodic saves.
  Vista derives the directory from its Stockyard root and TOML `run_name`.
  `CHECKPOINT_INTERVAL` is a runtime cadence override; `CHECKPOINT_DIR`,
  `RESUME_CHECKPOINT`, and `STOP_AFTER_COMPLETED_UPDATES` remain compatibility
  and recovery controls.
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
scripts/ls6/train.sh configs/moe_e8k2_r2.toml
source scripts/vista/env.sh
scripts/vista/train.sh --steps 30 configs/moe_e64k8_r0.5.toml
scripts/vista/train.sh configs/moe_e64k8_r0.5.toml
```

Vista `env.sh` establishes machine state only. Do not globally export
`MOE_GMM_IMPLEMENTATION=torch`; `train.sh` scopes it to the GH200 trainer
process and clears stale experiment/profile/checkpoint state. It defaults to
`configs/moe_e64k8_r0.5.toml`; multiple configs run sequentially. The `--steps`
validation path is current-node only, preserves the configured schedule horizon,
and stops through `STOP_AFTER_COMPLETED_UPDATES`. It disables TensorBoard and
periodic TOML checkpoint policy, saving only to a unique `interactive-smoke`
checkpoint directory outside the production run directory.

Vista training-pipeline benchmarks reuse the same trainer and update loop,
defaulting to 10 warmup plus 30 measured optimizer updates. They disable
TensorBoard and reject experiment/checkpoint/profiling overrides:

```bash
scripts/vista/benchmark.sh configs/moe_e8k2_r2.toml
```

Use `BENCHMARK_WARMUP_UPDATES` and `BENCHMARK_MEASURED_UPDATES` only when a
different benchmark window is intentionally required.

Submit one or more Vista configs from a login node with:

```bash
scripts/vista/train.sh --submit configs/moe_e64k8_r0.5.toml
```

Use `scripts/vista/train.sh --submit CONFIG [CONFIG ...]`. The script submits
itself in a non-recursive worker mode, validates all configs before starting,
runs them sequentially in one allocation, clears inherited run controls, and
writes one persistent combined log under
`$STOCKYARD/logs/modded-nanogpt-moe/vista/slurm/`. It uses the established
six-hour default; `--submit --time SLURM_TIME` overrides it. Short validation
runs belong on an existing interactive allocation. Checkpoint cadence normally
comes from each TOML; the optional override and first-config resume controls are
documented in `README.md`.

Submission accepts `--job-name NAME`, `--account ACCOUNT`, and repeatable
`--sbatch-arg=--option=value` (or `--sbatch-arg --flag`), all requiring
`--submit`. Scheduler options are quoted array elements, not shell commands;
`--wrap` is rejected to preserve the worker. These do not change TOML run names,
checkpoint/TensorBoard paths, or enable distributed training. See README.

Set `SLURM_MAIL_USER` to a nonempty email address before `--submit` to request
`--mail-user` and `--mail-type=ALL`. Unset/empty adds no email options.
Notifications cover the Slurm job, not individual configs within the suite.

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
