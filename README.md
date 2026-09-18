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

## Training

Run from the repository root. A single-GPU dense baseline is:

```bash
uv run --no-sync torchrun \
  --standalone \
  --nproc_per_node=1 \
  --module modded_nanogpt_moe.train \
  --config configs/dense_baseline.toml
```

The grouped MoE configuration is:

```bash
uv run --no-sync torchrun \
  --standalone \
  --nproc_per_node=1 \
  --module modded_nanogpt_moe.train \
  --config configs/moe_grouped.toml
```

Cluster launchers accept a config path and default to the dense baseline:

```bash
scripts/ls6/train.sh configs/moe_grouped.toml
scripts/vista/train.sh configs/moe_grouped.toml
```

Each launcher supplies reusable machine defaults when the corresponding TACC
variables are available:

```text
DATA_ROOT=$SCRATCH/modded-nanogpt-moe
TB_ROOT=$STOCKYARD/tensorboard
TB_SYSTEM=ls6 or vista
```

An already-set value takes precedence. Set `TB_ROOT=` explicitly to disable
TensorBoard for an individual run. The trainer prints the resolved runtime
paths, matched data shards, Git/environment metadata, and output destinations
at startup; checkpoints retain that information as non-compatibility metadata.

Existing operational environment controls remain available, including
`DATA_ROOT`, `TB_ROOT`, `TB_SYSTEM`, checkpoint/resume variables, Nsight
profiling variables, and the historical smoke-test overrides. Environment
overrides take precedence over TOML values.

TensorBoard events are written under:

```text
<TB_ROOT>/modded-nanogpt-moe/<TB_SYSTEM>/<run_name>/trial_<n>
```

A fresh run refuses to reuse an existing TensorBoard run directory.

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
