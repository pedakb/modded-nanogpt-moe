# Experiment configs

Scientific settings live in these TOMLs; machine roots remain environment
settings. The canonical standard-BP references are `dense_baseline.toml`,
`moe_e8k2_r2.toml`, and `moe_e64k8_r0.5.toml`.

## Main Grad-EM comparison

All entries below use grouped GEMM with packed experts, the Muon router, no
router AdamW LR, and the full 3250-update horizon. Within each geometry, only
the run identity and backward mode/eta differ; optimizer, model geometry,
data, batch, evaluation, diagnostics and checkpoint settings match the
canonical baseline. The E64 local eta=0.01 condition also matches the existing
Muon-router `moe_e64k8_r0.5_gradem_eta0.1.toml` except run identity, mode and eta.

These are values resolved through `load_experiment_config`. In standard BP,
`grad_em_mode=global` and `eta=0.1` are inactive defaults. The router's
`muon` default is intentionally retained by omitting optimizer overrides.

filename | run_name | E | K | backend | layout | moe_backward | grad_em_mode | eta | router_optimizer | router_adamw_lr | total_steps
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
moe_e8k2_r2.toml | moe-e8k2-r2 | 8 | 2 | grouped_gemm | packed | standard | global | 0.1 | muon | None | 3250
moe_e8k2_r2_gradem_global_eta0.1.toml | moe-e8k2-r2-gradem-global-eta0.1 | 8 | 2 | grouped_gemm | packed | grad_em | global | 0.1 | muon | None | 3250
moe_e8k2_r2_gradem_global_eta0.01.toml | moe-e8k2-r2-gradem-global-eta0.01 | 8 | 2 | grouped_gemm | packed | grad_em | global | 0.01 | muon | None | 3250
moe_e8k2_r2_gradem_local_bp_packed.toml | moe-e8k2-r2-gradem-local-bp-packed-eta0.1 | 8 | 2 | grouped_gemm | packed | grad_em | local_bp | 0.1 | muon | None | 3250
moe_e8k2_r2_gradem_local_bp_eta0.01.toml | moe-e8k2-r2-gradem-local-bp-eta0.01 | 8 | 2 | grouped_gemm | packed | grad_em | local_bp | 0.01 | muon | None | 3250
moe_e64k8_r0.5.toml | moe-e64k8-r0.5 | 64 | 8 | grouped_gemm | packed | standard | global | 0.1 | muon | None | 3250
moe_e64k8_r0.5_gradem_global_eta0.01.toml | moe-e64k8-r0.5-gradem-global-eta0.01 | 64 | 8 | grouped_gemm | packed | grad_em | global | 0.01 | muon | None | 3250
moe_e64k8_r0.5_gradem_local_bp_eta0.01.toml | moe-e64k8-r0.5-gradem-local-bp-eta0.01 | 64 | 8 | grouped_gemm | packed | grad_em | local_bp | 0.01 | muon | None | 3250

For the requested local-BP runs, use:

- E8/K2, eta=0.1: [moe_e8k2_r2_gradem_local_bp_packed.toml](moe_e8k2_r2_gradem_local_bp_packed.toml).
- E8/K2, eta=0.01: [moe_e8k2_r2_gradem_local_bp_eta0.01.toml](moe_e8k2_r2_gradem_local_bp_eta0.01.toml).
- E64/K8, eta=0.01, Muon: [moe_e64k8_r0.5_gradem_local_bp_eta0.01.toml](moe_e64k8_r0.5_gradem_local_bp_eta0.01.toml).

The E8 eta=0.1 packed run keeps
`moe-e8k2-r2-gradem-local-bp-packed-eta0.1` because the shorter
`moe-e8k2-r2-gradem-local-bp-eta0.1` already identifies the loop reference.
This avoids TensorBoard/checkpoint directory collisions while preserving the
older reference identity. No other new run name includes a backend detail.

## Existing references and historical experiments

- Permanent references: the three canonical baselines above, the E8 loop BP
  reference `moe_e8k2_r2_bp_loop.toml`, the E8 loop local-BP reference
  `moe_e8k2_r2_gradem_local_bp.toml`, and the packed local eta=0.1 config.
  The loop and packed references are useful execution comparisons, not duplicates.
- Existing E64 global Grad-EM experiments:
  `moe_e64k8_r0.5_gradem_eta0.001.toml`,
  `moe_e64k8_r0.5_gradem_eta0.1.toml`,
  `moe_e64k8_r0.5_gradem_eta0.3.toml`, and
  `moe_e64k8_r0.5_gradem_eta1.0.toml`. Their omitted mode resolves to global;
  all retain the Muon router and their original names/settings.
- The eta=0.001 historical config has a run name ending in `-500`, but still
  has `total_steps=3250`. It also evaluates every 50 updates, unlike the
  125-update cadence of the main comparison. Those existing settings are
  preserved; the suffix does not stop training after 500 updates.
- Both E64 eta=0.01 router-AdamW ablations remain separate and unchanged:
  `moe_e64k8_r0.5_gradem_eta0.01_router_adamw.toml` and
  `moe_e64k8_r0.5_gradem_eta0.01_router_adamw_lr0.003.toml`.
  They retain checkpoint interval 50 and their existing run names.
- No `*_tmp.toml`, ignored temporary TOML, or filename containing escaped
  punctuation/backslashes was present during the 2026-09-28 inspection.
  No permanent config was deleted or renamed.

The untracked `moe_e8k2_r2_gradem_global_eta0.1.toml` initially requested
unsupported `loop + modulelist + global Grad-EM`. Its original bytes were
preserved locally in the ignored
`logs/config-archive/2026-09-28/moe_e8k2_r2_gradem_global_eta0.1_loop.toml`,
outside the config glob. The production file at its original path now uses
`grouped_gemm + packed`; its scientific settings and run name are retained.
All other pre-existing config files were verified byte-for-byte unchanged.

## Vista run length and TensorBoard

From the repository root, on an existing Vista allocation:

```bash
source scripts/vista/env.sh
scripts/vista/train.sh --steps 2 configs/moe_e8k2_r2_gradem_local_bp_packed.toml
```

The launcher's `--steps` path is a smoke/recovery facility: it clears
`TB_ROOT`, disables the periodic TOML checkpoint policy, writes completion
state to a separate interactive-smoke directory, and preserves the configured
training schedule horizon. It does not create a TensorBoard production run.

For a shortened run that should retain TensorBoard and the full schedule,
use the existing package entry point with a per-command
`STOP_AFTER_COMPLETED_UPDATES` and an explicit durable `CHECKPOINT_DIR`,
leaving the `TB_ROOT` established by `scripts/vista/env.sh` intact. Do not
use `TRAIN_STEPS_OVERRIDE` to cap such a run: that changes the schedule horizon.
The launcher's normal path clears inherited stop controls, so those controls
must be passed to the package entry point directly. See the root
[README](../README.md) for the command and checkpoint conventions.

No launcher change or permanent `_500.toml` config was introduced.

## Validation

The cleanup loaded all 17 TOMLs through `load_experiment_config`, verified
unique run names, constructed a small MoE with each config's actual backend,
layout and backward settings, and compared complete resolved configs against
their baseline/reference. All requested local-BP configs resolve to a Muon
router with no AdamW LR. The existing config-globbing, package and optimizer
checks plus the packed-config reference check passed: **129 passed**.

```bash
uv run --no-sync python -m pytest -q -rs \
  tests/test_grad_em.py tests/test_package.py tests/test_router_optimizer.py \
  tests/test_grad_em_grouped_local_bp.py::test_packed_config_preserves_loop_experiment
git diff --check
```
