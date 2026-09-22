# modded-nanogpt-moe

Research code for dense and dropless grouped-GEMM mixture-of-experts training
on LS6 A100 and Vista GH200 systems. The implementation is derived from
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) and retains
its license and attribution.

## Layout

- `modded_nanogpt_moe/`: model, optimizer, data, checkpoint, configuration,
  and training implementation.
- `configs/`: reproducible TOML experiment configurations.
- `tests/`: local and CUDA parity tests.
- `tools/`: checkpoint comparison, grouped-GEMM validation, benchmarking, and
  profiling utilities.
- `scripts/`: data download and LS6/Vista launchers.
- `docs/`: system-specific implementation notes.

## Environment

Create the environment appropriate for the current platform with `uv`. Cluster
runs use the already-synchronized environment and therefore pass `--no-sync`.
The grouped-GEMM extension is installed separately on CUDA systems; see
[`docs/grouped_gemm.md`](docs/grouped_gemm.md).

## Data

The trainer resolves shard patterns relative to the repository root unless
`DATA_ROOT` is set. Download the cached FineWeb10B GPT-2 shards with:

```bash
uv run python scripts/download_fineweb10b.py 20
```

On LS6 and Vista, keep the shared source dataset on Stockyard and place or
link the active copy at `<DATA_ROOT>/data/fineweb10B` on the system's scratch
filesystem.

On the current Vista checkout, `data/fineweb10B` is a repository-local symlink
to the active FineWeb10B copy on SCRATCH. The config paths remain repository-
relative; the application does not depend on the symlink's physical target.

## Training

Run from the repository root. A single-GPU dense baseline is:

```bash
uv run --no-sync torchrun \
  --standalone \
  --nproc_per_node=1 \
  --module modded_nanogpt_moe.train \
  --config configs/dense_baseline.toml
```

The production comparison uses dense ratio 4, E8/K2 ratio 2, and E64/K8 ratio
0.5. Both MoE configs use grouped GEMM with packed experts. All three share
D=768, 12 layers, 3250 updates, seed 1234, a 524288-token global batch,
microbatch 64, 10485760 validation tokens per evaluation, diagnostics every 25
updates, and checkpoints every 100 updates. Evaluation runs every 125 updates,
switching to every 25 updates for the final 10%.

The E8/K2 configuration is:

```bash
uv run --no-sync torchrun \
  --standalone \
  --nproc_per_node=1 \
  --module modded_nanogpt_moe.train \
  --config configs/moe_e8k2_r2.toml
```

Vista machine setup is reusable without selecting any experiment state:

```bash
source scripts/vista/env.sh
uv run --no-sync python -m pytest -q
```

Do **not** globally export `MOE_GMM_IMPLEMENTATION=torch`; it is specific to
GH200 training and breaks CPU unit tests. The Vista training launchers scope it
only to the trainer process.

The Vista launcher is the single entry point for both current-node runs and
Slurm submissions. It defaults to the production E64/K8 config when no path is
supplied. Short runs stay on an already allocated node:

```bash
scripts/vista/train.sh --steps 30 configs/moe_e64k8_r0.5.toml
scripts/vista/train.sh --steps 100 configs/moe_e64k8_r0.5.toml
scripts/vista/train.sh configs/moe_e64k8_r0.5.toml
scripts/vista/train.sh configs/moe_e8k2_r2.toml
```

It resolves the repository root from its location, uses that root as
`DATA_ROOT`, clears inherited run state, and runs one GPU. `--steps` requires
one config, cannot be combined with `--submit` or checkpoint controls, and
disables TensorBoard and TOML checkpoint policy so a smoke run cannot claim or
overwrite production artifacts. It does not modify the TOML or parent shell.

Submit one config by adding `--submit`:

```bash
scripts/vista/train.sh --submit configs/moe_e64k8_r0.5.toml
```

Multiple configs run sequentially in the supplied order inside one allocation:

```bash
scripts/vista/train.sh --submit \
  configs/dense_baseline.toml \
  configs/moe_e8k2_r2.toml \
  configs/moe_e64k8_r0.5.toml
```

`train.sh --submit` submits the same script in a non-recursive worker mode. It
preserves the established `gh`, one-node, one-task, six-hour default. Override
Slurm settings with `--time`, `--job-name`, and `--account`, for example:

```bash
scripts/vista/train.sh --submit \
  --job-name dense-moe-comparison --account YOUR_ALLOCATION --time 06:00:00 \
  --sbatch-arg=--partition=gh \
  configs/dense_baseline.toml configs/moe_e8k2_r2.toml configs/moe_e64k8_r0.5.toml
```

Use repeatable `--sbatch-arg=--option=value` (or `--sbatch-arg --flag`) for other
scheduler options. Each occurrence forwards one argument, without shell
evaluation; use `=` inside options taking a value. These options require
`--submit` and are not passed to the training worker. `--wrap` is disallowed
because it would replace the worker script. Additional options follow the
launcher time/email settings in the sbatch argument list; avoid conflicting
settings. Resource overrides do not enable multi-node or multi-GPU training.
The job name labels the allocation only: TOML `run_name`, TensorBoard paths,
checkpoint paths, and the persistent suite log name are unchanged.

All config paths and
TOML run names are validated before submission and again before execution.
Configs run as separate processes in the supplied order, stopping at the first
failure. Duplicate run names in one suite are rejected. One combined suite log
is kept at
`$STOCKYARD/logs/modded-nanogpt-moe/vista/slurm/train-configs-JOBID.log`.

For optional Slurm email notifications, set `SLURM_MAIL_USER` for submission:

```bash
SLURM_MAIL_USER="you@example.com" scripts/vista/train.sh --submit configs/moe_e8k2_r2.toml
```

A nonempty value adds `--mail-user` and `--mail-type=ALL`; unset/empty adds no
email options. Notifications apply to the whole job, not each config in a suite.

Checkpoint cadence normally comes from the experiment TOML:

```toml
[checkpoint]
interval = 100
```

An omitted section disables checkpointing. An explicit interval of zero writes
only the final checkpoint; a positive interval also writes at that completed-
update cadence. All three production configs use 100. Each run writes under
`$STOCKYARD/checkpoints/modded-nanogpt-moe/RUN_NAME`. Override the TOML cadence
for every config in one invocation when needed:

```bash
scripts/vista/train.sh --submit \
  --checkpoint-interval 250 \
  configs/moe_e64k8_r0.5.toml
```

After a failure, resubmit only the unfinished configs. To resume the first one,
pass its checkpoint explicitly; `--resume` never applies to later configs:

```bash
checkpoint_dir="$STOCKYARD/checkpoints/modded-nanogpt-moe/moe-e8k2-r2"
scripts/vista/train.sh --submit \
  --resume "$checkpoint_dir/latest.pt" \
  configs/moe_e8k2_r2.toml \
  configs/moe_e64k8_r0.5.toml
```

`latest.pt` and `previous.pt` rotate atomically. Resume restores the saved run
identity, model, both optimizers, loader cursor, RNG, and timing. Resume is
enabled only by `--resume`; stale checkpoint variables are not inherited from
the parent shell. Effective cadence precedence is
`--checkpoint-interval`, then TOML `[checkpoint].interval`, then disabled.
Checkpoint/resume remains restricted to `num_trials = 1`; multi-trial
checkpointing is rejected before any files can collide.

These configs define fresh production runs: older ModuleList E8 checkpoints
and 3500-step schedules are not compatible with the new packed/3250-step
settings. Preserve the original config when resuming an older run. Existing
TensorBoard run directories are not migrated or overwritten.

The LS6 launcher remains available as before:

```bash
scripts/ls6/train.sh configs/moe_e8k2_r2.toml
```

TensorBoard events are written under:

```text
<TB_ROOT>/modded-nanogpt-moe/<TB_SYSTEM>/<run_name>/trial_<n>
```

A fresh run refuses to reuse an existing TensorBoard run directory.

Validation policy is independent of training diagnostics. `[evaluation]`
controls validation tokens and the regular/final-phase cadence; validation is
always run at step 0 and the final step. `[diagnostics].scalar_interval` controls
only sampled training diagnostics. Validation loss remains `metric/loss/val`.

TensorBoard training diagnostics default in code to every 10 optimizer updates;
all three production configs explicitly use 25, with histograms off.
See [diagnostic metrics and overhead](docs/diagnostics.md) for
the optional `[diagnostics]` TOML settings. Benchmarks always bypass diagnostics;
Nsight bypasses them unless explicitly enabled.

## Validation

Run local tests with:

```bash
uv run --no-sync python -m pytest -q -rs tests
```

Inspect the available grouped-GEMM validation and benchmark modes with:

```bash
uv run --no-sync python -m tools.validate_grouped_gemm --help
```

Checkpoint and reproducibility comparisons are available through
`tools.compare_checkpoints` and `tools.compare_repro_diagnostics`.

## Checkpoint support

Checkpointing is opt-in and currently supports single-GPU, single-trial runs.
Checkpoints contain model and optimizer state, loader position, RNG state,
resolved model/training configuration, schedule horizon, run identity, and
training-time accounting. Dataset roots may move between systems when shard
identities and ordering remain unchanged.

## Attribution

This project builds on the architecture, training methodology, and historical
work in the upstream modded-nanogpt project and NanoGPT/llm.c lineage. See the
repository history and `LICENSE` for the preserved upstream record and terms.
