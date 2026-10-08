# Optional router regularization

Add these settings under `[model]` in a new experiment TOML:

```toml
router_aux_loss_coef = 0.01
router_z_loss_coef = 0.001
```

Both default to zero. Values must be finite and nonnegative; enabled losses
require MoE. Existing production configs are unchanged. Resume requires matching
coefficients; legacy checkpoints missing them mean zero. Model keys and optimizer
state are unchanged.

The auxiliary objective is `sum_e f_e P_e`, with detached assignment counts
`f_e = E * count_e / (K*T)` and full pre-Top-K FP32 probabilities
`P_e = mean_t p[t,e]`. Balanced assignment gives a loss of one regardless of E/K.
Z-loss is the token mean of squared FP32 logsumexp of pre-softmax router logits.
Shared experts are absent from these calculations. Each objective is averaged
over MoE layers. Counts are local to each microbatch, not pooled over the update;
equal-sized microbatches receive equal weight across gradient accumulation.
The trainer multiplies the weighted mean by microbatch tokens to match summed
CE, then uses the unchanged gradient summation across microbatches and ranks.

Regularization uses a separate ordinary-BP transformer replay without the LM
head. Temporarily selecting standard backward avoids both the current layer's
local-BP custom router boundary and upstream Grad-EM boundaries. This is necessary
even in global mode: a regularizer added directly to later router logits would
otherwise feed into earlier Grad-EM loss-to-go signals. The original LM graph
is backpropagated first, unchanged. Regularizer backward is separate and additive;
eta, lambda, and alpha never scale it. Shared experts use ordinary BP, including
upstream effects on later router losses. Replay restores modes and diagnostic
hooks on success or failure. No Grad-EM functions or optimizer math were edited.

Enabled regularization costs an extra eager transformer forward/backward and
activation memory. Both BP and Grad-EM use this same replay definition. With zero
coefficients no replay occurs. Validation, divergence detection and LM loss logs
remain CE-only. Rank-zero TensorBoard logs microbatch averages under
`metric/router/aux_loss`, `metric/router/z_loss`, and
`metric/router/regularization`; individual losses are logged when their coefficient
is positive. As with existing train loss, logs describe rank-zero local data.

For E256/K6, start with auxiliary coefficient 0.01 and z coefficient 0.001.
These are initial experimental choices, not measured optima.
These weights also match the recipe reported by
[OLMoE](https://arxiv.org/html/2409.02060v1#S4.SS1.SSS6).
There is no extra E, K or layer multiplier to apply. Near uniform logits the z-loss is about
`log(256)^2 = 30.75`, so its weighted contribution is about 0.031 per token;
balanced auxiliary contribution is 0.01. A small sweep could use auxiliary
0.001/0.01 and z 0.0001/0.001. Use fresh run names and keep old runs unchanged.

On an existing Vista allocation, from this isolated checkout:

```bash
source scripts/vista/env.sh
export UV_PROJECT_ENVIRONMENT=/path/to/original-checkout/.venv
uv run --no-sync python -m pytest -q -rs tests/test_router_regularization.py
```

Create fresh smoke TOMLs without editing the production file:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
source = Path("configs/moe_architectures/moe_e256k6.toml").read_text()
for mode in ("standard", "global", "local_bp"):
    text = source.replace('run_name = "bp-matrix-moe-e256k6-h512"',
                          f'run_name = "router-reg-smoke-{mode}"')
    backward = "standard" if mode == "standard" else "grad_em"
    em_mode = "global" if mode == "standard" else mode
    fields = (f'moe_backward = "{backward}"\ngrad_em_mode = "{em_mode}"\n'
              'router_aux_loss_coef = 0.01\nrouter_z_loss_coef = 0.001\n')
    text = text.replace('[model]\n', '[model]\n' + fields)
    Path(f"/tmp/router-reg-smoke-{mode}.toml").write_text(text)
PY
scripts/vista/train.sh --smoke --steps 2 /tmp/router-reg-smoke-standard.toml
scripts/vista/train.sh --smoke --steps 2 /tmp/router-reg-smoke-global.toml
scripts/vista/train.sh --smoke --steps 2 /tmp/router-reg-smoke-local_bp.toml
```

`UV_PROJECT_ENVIRONMENT` points to the already provisioned environment; adapt
it to your checkout. This worktree has an ignored data/fineweb10B symlink to the
original checkout's data. If moved, recreate that link; do not resynchronize
CUDA dependencies. Full E256 validation and memory feasibility remain GPU checks.
