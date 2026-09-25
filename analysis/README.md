# Grad-EM analysis

`grad_em_dynamics.ipynb` combines existing Vista TensorBoard scalars with
fixed-batch outputs from `scripts/extract_grad_em_diagnostics.py`. It is a
CPU-only consumer: it does not import the model, run extraction, or mutate
experiment data.

The notebook reads these roots by default:

```text
$STOCKYARD/tensorboard/modded-nanogpt-moe/vista
$STOCKYARD/analysis/modded-nanogpt-moe
```

Override them with `GRAD_EM_TB_ROOT` and `GRAD_EM_DIAG_ROOT`. Add or change a
run only in the `RUNS` registry near the top of the notebook. Missing runs,
tags, and diagnostic directories are reported and skipped.

Open the notebook from this repository in VS Code and select an existing
Python 3.11 kernel that can import the project's NumPy, PyTorch, TensorBoard,
and optionally pandas packages. Matplotlib is not required; plots are inline
SVG. This repository does not add Jupyter or kernel dependencies.

## Generic checkpoint routing extraction

`scripts/extract_checkpoint_routing.py` reuses the fixed-batch checkpoint
replay to save method-agnostic router, task-sensitivity, and selected-expert
geometry inputs for later CPU analysis. For example:

```bash
MOE_GMM_IMPLEMENTATION=torch \
uv run --no-sync python scripts/extract_checkpoint_routing.py \
  --checkpoint-dir "$STOCKYARD/checkpoints/modded-nanogpt-moe/RUN_NAME" \
  --steps 50,100,150 \
  --layers 0,5,11 \
  --output-dir "$STOCKYARD/analysis/modded-nanogpt-moe/RUN_NAME/routing"
```

Use `--checkpoint PATH` instead of `--checkpoint-dir`/`--steps` for one
checkpoint. `--eval-batch PATH` lets multiple extraction directories share
one fixed `eval_batch.pt`; its companion `eval_tokens.pt` records flattened
input/target IDs, sequence indices, and token positions. Artifacts with
different batch hashes are rejected. On memory-constrained replays,
`--analysis-microbatch-sequences N` processes contiguous sequence chunks
without changing token order or the saved sufficient statistics.

Each checkpoint artifact stores full logits, actual selected IDs and weights,
summed-task-loss sensitivities, incoming-gradient norms, `H H^T / d`, router
input norms and first/second moments, router parameters, and unreduced token
loss. It intentionally does not store entropy, load, KL/TV, eigenspectra, or
other quantities that are inexpensive to derive on CPU. Grad-EM checkpoints
use their unchanged forward but the standard task-loss replay backward, so
saved sensitivities contain neither `q` nor `eta` and have the same meaning as
backprop checkpoints.
