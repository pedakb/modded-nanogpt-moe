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
