# Experiment configs

Scientific settings live in explicit TOMLs; machine roots remain environment
settings. No inheritance layer is used.

## Directory policy

- `baselines/` contains the established dense and E64/K8 standard-BP
  baselines.
- `moe_architectures/` contains the standard-BP expert-count, routing-sparsity,
  and shared-expert studies. These active configs have unique run identities
  and therefore distinct TensorBoard and checkpoint directories.
- `archive/legacy/` contains five historical standard-BP baselines and scaling
  references.
- `archive/grad_em/` contains 29 historical Grad-EM comparisons and sweeps.
- `archive/router_ablations/` contains two router-optimizer ablations.
- `archive/prototypes/` contains two superseded shared-expert prototypes.

The `configs/` root intentionally contains only this catalog and organized
subdirectories. Archived TOMLs keep their original bytes and run names and
must not be reused for new experiments.

The loader accepts any explicit TOML path, including nested paths. The Vista
launcher derives checkpoint and TensorBoard destinations from `run_name`, not
from the config filename. Checkpoint compatibility compares the saved resolved
configuration, so moving a byte-identical TOML does not change model or resume
semantics.

## Active standard-BP matrix

Every matrix entry uses D=768, 12 layers, packed grouped-GEMM routed experts,
standard BP, seed 1234, 1,703,936,000 scheduled training tokens, a 524,288-token
effective batch, the production optimizer/LR schedule, and checkpoint interval
250. No load-balancing loss, adaptive routing bias, or router z-loss exists in
the implementation/config schema.

Widths are multiples of 16 for grouped-GEMM efficiency. When 4x cannot be met
exactly, the closest practical width is used; shared and routed widths are
identical within each shared-expert config. Parameter counts include routed
and shared MLP weights and biases across all 12 layers, but exclude routers and
the transformer backbone.

config | run_name | E | K | routed H | shared H | active expansion | expert params
--- | --- | ---: | ---: | ---: | ---: | ---: | ---:
`dense.toml` | `bp-matrix-dense-d768-h3072` | — | — | 3072 | — | 4.0000 | 56.7M
`moe_e64k8.toml` | `bp-matrix-moe-e64k8-h384` | 64 | 8 | 384 | — | 4.0000 | 453.9M
`moe_e64k8_shared.toml` | `bp-matrix-moe-e64k8-h336-shared-h336` | 64 | 8 | 336 | 336 | 3.9375 | 403.4M
`moe_e128k8.toml` | `bp-matrix-moe-e128k8-h384` | 128 | 8 | 384 | — | 4.0000 | 907.7M
`moe_e128k8_shared.toml` | `bp-matrix-moe-e128k8-h336-shared-h336` | 128 | 8 | 336 | 336 | 3.9375 | 800.6M
`moe_e256k6.toml` | `bp-matrix-moe-e256k6-h512` | 256 | 6 | 512 | — | 4.0000 | 2.420B
`moe_e256k6_shared.toml` | `bp-matrix-moe-e256k6-h432-shared-h432` | 256 | 6 | 432 | 432 | 3.9375 | 2.050B
`moe_e512k10.toml` | `bp-matrix-moe-e512k10-h304` | 512 | 10 | 304 | — | 3.9583 | 2.875B
`moe_e512k10_shared.toml` | `bp-matrix-moe-e512k10-h288-shared-h288` | 512 | 10 | 288 | 288 | 4.1250 | 2.730B

`dense.toml` and `moe_e64k8.toml` are under `configs/baselines/`; the remaining
seven entries are under `configs/moe_architectures/`.

## Archive inventory

Every former root-level TOML is retained under the following purpose-based
directory. Moving a TOML does not affect its `run_name`, checkpoint directory,
TensorBoard identity, or resolved-configuration compatibility.

- `archive/legacy/`:
  - `dense_baseline.toml`, `moe_e8k2_r2.toml`, and
    `moe_e64k8_r0.5.toml` are the original production comparison baselines.
  - `moe_e64k2_r2.0.toml` and `moe_e64k4_r1.0.toml` are standard-BP routing
    sparsity/width scaling references.
- `archive/grad_em/`:
  - E8/K2: `moe_e8k2_r2_gradem_global_eta0.01.toml`,
    `moe_e8k2_r2_gradem_global_eta0.1.toml`,
    `moe_e8k2_r2_gradem_local_bp.toml`,
    `moe_e8k2_r2_gradem_local_bp_eta0.01.toml`, and
    `moe_e8k2_r2_gradem_local_bp_packed.toml` compare global/local Grad-EM and
    loop/packed execution.
  - E64/K2 and E64/K4: each family contains one global eta=0.01 reference and
    local-BP eta=0.003/0.01/0.03 sweeps: `moe_e64k2_r2.0_gradem_*.toml` and
    `moe_e64k4_r1.0_gradem_*.toml`.
  - E64/K8 global eta studies: `moe_e64k8_r0.5_gradem_eta0.001.toml`,
    `moe_e64k8_r0.5_gradem_eta0.1.toml`,
    `moe_e64k8_r0.5_gradem_eta0.3.toml`,
    `moe_e64k8_r0.5_gradem_eta1.0.toml`, and
    `moe_e64k8_r0.5_gradem_global_eta0.01.toml`.
  - E64/K8 global alpha and lambda studies: the three
    `moe_e64k8_r0.5_gradem_global_eta0.01_alpha*.toml` and three
    `moe_e64k8_r0.5_gradem_global_eta0.01_lambda*.toml` files.
  - E64/K8 local-BP eta studies: the five
    `moe_e64k8_r0.5_gradem_local_bp_eta*.toml` files covering eta 0.001,
    0.003, 0.01, 0.03, and 0.1.
- `archive/router_ablations/`: the default-LR and dedicated-LR AdamW-router
  experiments, `moe_e64k8_r0.5_gradem_eta0.01_router_adamw.toml` and
  `moe_e64k8_r0.5_gradem_eta0.01_router_adamw_lr0.003.toml`.
- `archive/prototypes/`: `moe_e256k6_r0.5.toml` and
  `moe_e256k6_r0.5_shared_r0.5.toml`, the initial E256/K6 H384 unshared/shared
  pair superseded before launch by the approximately-4x matrix.

Historical commands and tests now use these explicit archive paths. New
production work should use `baselines/` or `moe_architectures/` according to
its purpose.

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

The standard-BP rows are under `archive/legacy/`; the Grad-EM rows are under
`archive/grad_em/`.

The four additional E64/K8 local-BP eta sweep configs differ from the eta=0.01
reference only in `run_name` and `model.grad_em_eta`. Use `--steps 500` for
500 completed updates with the original 3250-update LR schedule and normal
TensorBoard/checkpoint behavior.

For the requested local-BP runs, use:

- E8/K2, eta=0.1: [moe_e8k2_r2_gradem_local_bp_packed.toml](archive/grad_em/moe_e8k2_r2_gradem_local_bp_packed.toml).
- E8/K2, eta=0.01: [moe_e8k2_r2_gradem_local_bp_eta0.01.toml](archive/grad_em/moe_e8k2_r2_gradem_local_bp_eta0.01.toml).
- E64/K8, eta=0.01, Muon: [moe_e64k8_r0.5_gradem_local_bp_eta0.01.toml](archive/grad_em/moe_e64k8_r0.5_gradem_local_bp_eta0.01.toml).

The E8 eta=0.1 packed run keeps
`moe-e8k2-r2-gradem-local-bp-packed-eta0.1` because the shorter
`moe-e8k2-r2-gradem-local-bp-eta0.1` already identifies the loop reference.
This avoids TensorBoard/checkpoint directory collisions while preserving the
older reference identity. No other new run name includes a backend detail.

## Existing references and historical experiments

- Historical references: the three legacy baselines above, the E8 loop BP
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
- Both E64 eta=0.01 router-AdamW ablations remain separate and unchanged under
  `archive/router_ablations/`:
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
scripts/vista/train.sh --steps 2 configs/archive/grad_em/moe_e8k2_r2_gradem_local_bp_packed.toml
scripts/vista/train.sh --smoke --steps 2 configs/archive/grad_em/moe_e8k2_r2_gradem_local_bp_packed.toml
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

Recursive config tests load every active and archived TOML; enforce
globally unique run names; check the nine-job BP geometry, widths, active
expansion, parameter counts, schedule, optimizer policy, and shared-expert
contract; and exercise nested paths through the real Vista submission
preflight with a fake `sbatch`.

```bash
uv run --no-sync python -m pytest -q -rs \
  tests/test_experiment_configs.py tests/test_grad_em.py \
  tests/test_shared_expert.py \
  tests/test_package.py::test_production_configs_share_training_policy
git diff --check
```
