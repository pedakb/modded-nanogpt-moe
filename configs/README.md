# Experiment configs

Scientific settings live in these TOMLs; machine roots remain environment
settings. The canonical standard-BP references are `dense_baseline.toml`,
`moe_e8k2_r2.toml`, and `moe_e64k8_r0.5.toml`.

## Shared-expert comparison

`moe_e256k6_r0.5_shared_r0.5.toml` and `moe_e256k6_r0.5.toml` are the
standard-BP E256/K6 comparison. Both use packed routed experts of width 384;
the first additionally enables one dense shared expert of width 384. The
shared expert runs for every token and is not part of routing or Grad-EM
responsibilities. Their distinct run names give them separate TensorBoard and
checkpoint directories under the normal Vista roots.

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
moe_e64k8_r0.5_gradem_local_bp_eta0.001.toml | moe-e64k8-r0.5-gradem-local-bp-eta0.001 | 64 | 8 | grouped_gemm | packed | grad_em | local_bp | 0.001 | muon | None | 3250
moe_e64k8_r0.5_gradem_local_bp_eta0.003.toml | moe-e64k8-r0.5-gradem-local-bp-eta0.003 | 64 | 8 | grouped_gemm | packed | grad_em | local_bp | 0.003 | muon | None | 3250
moe_e64k8_r0.5_gradem_local_bp_eta0.03.toml | moe-e64k8-r0.5-gradem-local-bp-eta0.03 | 64 | 8 | grouped_gemm | packed | grad_em | local_bp | 0.03 | muon | None | 3250
moe_e64k8_r0.5_gradem_local_bp_eta0.1.toml | moe-e64k8-r0.5-gradem-local-bp-eta0.1 | 64 | 8 | grouped_gemm | packed | grad_em | local_bp | 0.1 | muon | None | 3250

The four additional E64/K8 local-BP eta sweep configs differ from the eta=0.01
reference only in `run_name` and `model.grad_em_eta`. Use `--steps 500` for
500 completed updates with the original 3250-update LR schedule and normal
TensorBoard/checkpoint behavior.

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
scripts/vista/train.sh --smoke --steps 2 configs/moe_e8k2_r2_gradem_local_bp_packed.toml
```

The launcher's `--steps N` sets only `STOP_AFTER_COMPLETED_UPDATES=N`.
TensorBoard logging, the normal run checkpoint directory and TOML checkpoint
policy are retained, while the configured training/LR schedule horizon remains
unchanged. For example, `--steps 500` on any main comparison config stops after
update 500 with a 3250-update schedule and normal logging/checkpointing.
Do not use `TRAIN_STEPS_OVERRIDE` to cap such a run: it changes the schedule
horizon. See the root [README](../README.md) for launcher option restrictions
and checkpoint conventions. No permanent `_500.toml` config is needed.

Add `--smoke` when the short run should produce no TensorBoard events or
checkpoints. It requires `--steps`, keeps the full LR schedule, clears
`TB_ROOT`, and sets `CHECKPOINT_POLICY_DISABLED=1`. No checkpoint directory
is created, and neither periodic nor final/early-stop checkpoints are saved.

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
