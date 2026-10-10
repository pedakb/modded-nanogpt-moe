# E256/K6 shared BP auxiliary load-balancing sweep

Run from the original repository root. The persistent isolated worktree is
`.worktrees/router-regularizers` on branch `router-regularizers` (base
`7095ce6`). Its local data link uses the existing FineWeb data, and
`UV_PROJECT_ENVIRONMENT` reuses the provisioned environment. No active BP or
Grad-EM configurations are edited. The existing unregularized BP run
`bp-matrix-moe-e256k6-h432-shared-h432` is the control; do not relaunch it.

All five new configs are under `configs/router_regularizers/` in the isolated
worktree:

| File | Auxiliary coefficient | Z coefficient | Job |
| --- | ---: | ---: | --- |
| `moe_e256k6_shared_bp_aux0.001_z0.toml` | 0.001 | 0 | `bp256-aux0.001` |
| `moe_e256k6_shared_bp_aux0.003_z0.toml` | 0.003 | 0 | `bp256-aux0.003` |
| `moe_e256k6_shared_bp_aux0.01_z0.toml` | 0.01 | 0 | `bp256-aux0.01` |
| `moe_e256k6_shared_bp_aux0.03_z0.toml` | 0.03 | 0 | `bp256-aux0.03` |
| `moe_e256k6_shared_bp_aux0.1_z0.toml` | 0.1 | 0 | `bp256-aux0.1` |

Run names are `bp-aux-moe-e256k6-h432-shared-h432-auxCOEF-z0`.
All retain the control architecture (E256/K6, routed/shared H432, D768/L12),
seed 1234, optimizer mathematics and learning-rate schedule, 524288-token
global batch, 64-sequence microbatch, eight accumulation microbatches on one
GPU, 3250 updates, evaluation/diagnostic cadence, and checkpoint interval 250.
Only auxiliary coefficient, run identity, and enabled divergence guard differ
from the resolved BP baseline. Z-loss is explicitly zero; backward is
explicitly standard BP. Guard thresholds retain their existing defaults.

The auxiliary loss is
`E * sum_e (count_e / (K*T)) * mean_t softmax(logits)[t,e]`.
Counts are detached and local to each microbatch; only the E routed experts
appear. The loss is averaged over MoE layers, then multiplied by microbatch
tokens before addition to summed CE. Balanced routing gives one, without an
extra E, K, or layer multiplier. BP captures this loss in the original forward
graph and calls backward once. Grad-EM's separate replay remains unchanged.
Zero coefficients bypass regularizer hooks and loss computations.

Each Slurm job has a 03:05:00 walltime, one Vista gh/GH200 node and one task.
TIME_LIMIT_90 is supported by the installed Slurm 23.11.11 manual.
This submits five independent jobs, with all collision checks before the first
submission. It aborts if squeue cannot be queried. The scheduler is not queried
through this document, and the command has not been submitted.

```bash
set -euo pipefail
cd .worktrees/router-regularizers
export UV_PROJECT_ENVIRONMENT="$(cd ../.. && pwd)/.venv"
: "${STOCKYARD:?}" "${SCRATCH:?}"
checkpoint_root="${VISTA_CHECKPOINT_ROOT:-$SCRATCH/checkpoints/modded-nanogpt-moe}"
log_dir="$STOCKYARD/logs/modded-nanogpt-moe/vista/slurm"
queued_names=$(squeue --noheader --user "$USER" --format='%j')
configs=(); jobs=()

for coef in 0.001 0.003 0.01 0.03 0.1; do
  config="configs/router_regularizers/moe_e256k6_shared_bp_aux${coef}_z0.toml"
  job="bp256-aux${coef}"
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
  run=$(sed -nE 's/^run_name[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$config")
  [[ -n "$run" ]] || exit 1
  for path in "$checkpoint_root/$run" "$STOCKYARD/tensorboard/modded-nanogpt-moe/vista/$run" "logs/$run.txt"; do
    [[ ! -e "$path" && ! -L "$path" ]] || {
      echo "Existing experiment output: $path; stopping" >&2; exit 1;
    }
  done
  if printf '%s\n' "$queued_names" | grep -Fxq -- "$job"; then
    echo "Existing queued/running job: $job; stopping" >&2; exit 1
  fi
  configs+=("$config"); jobs+=("$job")
done

mkdir -p "$log_dir"
for i in "${!configs[@]}"; do
  SLURM_MAIL_USER= VISTA_CHECKPOINT_ROOT="$checkpoint_root" \
    scripts/vista/train.sh --submit \
    --account ASC26009 --time 03:05:00 --job-name "${jobs[$i]}" \
    --sbatch-arg=--partition=gh --sbatch-arg=--nodes=1 \
    --sbatch-arg=--ntasks-per-node=1 \
    --sbatch-arg=--mail-user=pe.akbarian@gmail.com \
    --sbatch-arg=--mail-type=BEGIN,END,FAIL,TIME_LIMIT_90 \
    --sbatch-arg="--output=$log_dir/%x-%j.out" \
    "${configs[$i]}"
done
```

The launcher clears inherited step/smoke/benchmark/checkpoint controls. No
`--steps` or `--smoke` is passed. Checkpoints are separate by run name under
`VISTA_CHECKPOINT_ROOT` or the default Scratch root; TensorBoard is under the
Stockyard root. Slurm output filenames use `%x-%j`, and the existing worker
also writes `train-configs-JOBID.log`.

A guard stop records `[DIVERGENCE_STOP]`, saves the completed stopping update
and guard state, and returns success: normally COMPLETED/END, not FAILED/FAIL.
Previous numbered checkpoints remain available; the stopping checkpoint is not
a best-model checkpoint. Validation and guard decisions continue to use CE
alone. No CUDA device is accessible in the preparation session; a GH200 smoke
remains pending. No throughput or walltime-completion claim is made.

Validation: focused router/shared/guard/config tests pass 59 with 3 CUDA skips;
config/inventory checks pass 31. Full suite: 865 passed, 339 skipped, 7 existing
launcher-fixture failures, all reproduced on an untouched archive of 7095ce6.
The exact shell block passes bash syntax and five fake-sbatch launcher
preflights, including queued-job/checkpoint/TensorBoard collision rejection
before the first submission. The control TOML is byte-identical to the original
checkout. Python compilation and git diff --check pass.
