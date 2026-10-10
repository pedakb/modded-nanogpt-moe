# E256/K6 shared-expert Grad-EM sweep

These configs derive from
`configs/moe_architectures/moe_e256k6_shared.toml`. They retain the original
3250-update LR schedule, seed, optimizer, batch, evaluation, diagnostics, and
checkpoint policy. Only the run identity and routed Grad-EM backward mode/eta
change. The shared expert remains outside the routed Grad-EM boundary and uses
ordinary BP.

config | mode | eta | run name | schedule horizon
--- | --- | ---: | --- | ---:
`moe_e256k6_shared_gradem_global_eta0.001.toml` | global | 0.001 | `gradem-moe-e256k6-h432-shared-h432-global-eta0.001` | 3250
`moe_e256k6_shared_gradem_global_eta0.003.toml` | global | 0.003 | `gradem-moe-e256k6-h432-shared-h432-global-eta0.003` | 3250
`moe_e256k6_shared_gradem_global_eta0.007.toml` | global | 0.007 | `gradem-moe-e256k6-h432-shared-h432-global-eta0.007` | 3250
`moe_e256k6_shared_gradem_global_eta0.01.toml` | global | 0.01 | `gradem-moe-e256k6-h432-shared-h432-global-eta0.01` | 3250
`moe_e256k6_shared_gradem_local_bp_eta0.001.toml` | local_bp | 0.001 | `gradem-moe-e256k6-h432-shared-h432-local-bp-eta0.001` | 3250
`moe_e256k6_shared_gradem_local_bp_eta0.003.toml` | local_bp | 0.003 | `gradem-moe-e256k6-h432-shared-h432-local-bp-eta0.003` | 3250
`moe_e256k6_shared_gradem_local_bp_eta0.007.toml` | local_bp | 0.007 | `gradem-moe-e256k6-h432-shared-h432-local-bp-eta0.007` | 3250
`moe_e256k6_shared_gradem_local_bp_eta0.01.toml` | local_bp | 0.01 | `gradem-moe-e256k6-h432-shared-h432-local-bp-eta0.01` | 3250

All paths in the table are relative to `configs/grad_em/`.

## Screening semantics

Do not set `training.total_steps=200` or use `TRAIN_STEPS_OVERRIDE=200`: either
would compress the cooldown into a 200-update horizon. The first 200 updates of
the intended full schedule stay at the initial LR because the 3250-update
schedule does not begin cooldown until update 975.

Vista deliberately rejects `--submit --steps`. To queue an exact 200-update
prefix, submit the existing launcher itself as the Slurm batch script and pass
`--steps 200` to its current-node path. This sets only
`STOP_AFTER_COMPLETED_UPDATES=200`; update 200 writes a checkpoint even though
the normal periodic interval is 250. Resume that checkpoint with the same TOML
for Stage 2.

## Memory smoke

E256 full-model memory has not been measured on GH200. Before screening, run
both backward modes for two updates. Smoke mode uses the production microbatch
but disables TensorBoard and all checkpoint writes.

```bash
log_dir="$STOCKYARD/logs/modded-nanogpt-moe/vista/slurm"
mkdir -p "$log_dir"

sbatch --account=ASC26009 --partition=gh --nodes=1 --ntasks-per-node=1 \
  --time=00:30:00 --job-name=ge256-global-smoke \
  --output="$log_dir/%x-%j.out" \
  scripts/vista/train.sh --smoke --steps 2 \
  configs/grad_em/moe_e256k6_shared_gradem_global_eta0.007.toml

sbatch --account=ASC26009 --partition=gh --nodes=1 --ntasks-per-node=1 \
  --time=00:30:00 --job-name=ge256-local-smoke \
  --output="$log_dir/%x-%j.out" \
  scripts/vista/train.sh --smoke --steps 2 \
  configs/grad_em/moe_e256k6_shared_gradem_local_bp_eta0.007.toml
```

## Stage 1: six independent 200-update jobs

Run from the repository root after both smokes pass:

```bash
set -euo pipefail
log_dir="$STOCKYARD/logs/modded-nanogpt-moe/vista/slurm"
mkdir -p "$log_dir"

while read -r job_name config; do
  job_id=$(sbatch --parsable \
    --account=ASC26009 --partition=gh --nodes=1 --ntasks-per-node=1 \
    --time=02:00:00 --job-name="$job_name" \
    --output="$log_dir/%x-%j.out" \
    scripts/vista/train.sh --steps 200 "$config")
  printf '%s %s\n' "$job_id" "$config"
done <<'EOF'
ge256-g-003 configs/grad_em/moe_e256k6_shared_gradem_global_eta0.003.toml
ge256-g-007 configs/grad_em/moe_e256k6_shared_gradem_global_eta0.007.toml
ge256-g-010 configs/grad_em/moe_e256k6_shared_gradem_global_eta0.01.toml
ge256-l-003 configs/grad_em/moe_e256k6_shared_gradem_local_bp_eta0.003.toml
ge256-l-007 configs/grad_em/moe_e256k6_shared_gradem_local_bp_eta0.007.toml
ge256-l-010 configs/grad_em/moe_e256k6_shared_gradem_local_bp_eta0.01.toml
EOF
```

The repository convention for Vista `gh` is one node, one task, and one
single-GPU `torchrun` process. The six loop iterations therefore create six
independent one-GH200 jobs, rather than one sequential allocation.

## Stage 2: resume selected screening runs

Edit `selected` to the screened configs chosen for continuation. Each job uses
the update-200 checkpoint and keeps its TensorBoard/run identity. Do not launch
the same config fresh after screening: the trainer intentionally refuses to
reuse its TensorBoard directory.

```bash
set -euo pipefail
selected=(
  configs/grad_em/moe_e256k6_shared_gradem_global_eta0.007.toml
  configs/grad_em/moe_e256k6_shared_gradem_local_bp_eta0.007.toml
)

for config in "${selected[@]}"; do
  run_name=$(sed -nE 's/^run_name[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$config")
  checkpoint="$SCRATCH/checkpoints/modded-nanogpt-moe/$run_name/latest.pt"
  [[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
  if [[ "$config" == *local_bp* ]]; then
    walltime=12:00:00
  else
    walltime=10:00:00
  fi
  scripts/vista/train.sh --submit \
    --account ASC26009 --time "$walltime" --job-name "${run_name:0:100}" \
    --sbatch-arg=--partition=gh --sbatch-arg=--nodes=1 \
    --sbatch-arg=--ntasks-per-node=1 \
    --resume "$checkpoint" "$config"
done
```

The eta=0.001 configs were not screened. Submit either one fresh by omitting
`--resume`, for example:

```bash
scripts/vista/train.sh --submit \
  --account ASC26009 --time 10:00:00 --job-name ge256-global-eta0001 \
  --sbatch-arg=--partition=gh --sbatch-arg=--nodes=1 \
  --sbatch-arg=--ntasks-per-node=1 \
  configs/grad_em/moe_e256k6_shared_gradem_global_eta0.001.toml
```

The launcher's normal six-hour default is retained for ordinary experiments.
The 10/12-hour recommendations here are conservative because E256 optimizer
state and validation/checkpoint overhead have not been timed end to end. Use
the 200-update logs to tighten them before Stage 2, and confirm the current
partition maximum with `sinfo -p gh -o '%P %l'`.

## Monitoring

```bash
squeue -u "$USER" -o '%.18i %.36j %.2t %.10M %.10l %R'
sacct -X -S today -u "$USER" --format=JobID,JobName%36,State,Elapsed,Timelimit,ExitCode
tail -F "$STOCKYARD/logs/modded-nanogpt-moe/vista/slurm/"ge256-*.out

run_name=gradem-moe-e256k6-h432-shared-h432-global-eta0.007
ls -lh "$SCRATCH/checkpoints/modded-nanogpt-moe/$run_name/"
find "$STOCKYARD/tensorboard/modded-nanogpt-moe/vista/$run_name" -maxdepth 2 -type f

tensorboard --logdir "$STOCKYARD/tensorboard/modded-nanogpt-moe/vista" --port 6006
```

Stage 1 checkpoints contain the resolved Grad-EM mode, eta, alpha/lambda,
shared-expert architecture, both optimizers, loader cursor, RNG state, and run
identity. Resume rejects a different mode, eta, architecture, or run name.
