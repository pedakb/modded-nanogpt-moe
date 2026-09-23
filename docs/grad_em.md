# Grad-EM: selected-support KL backward

`modded_nanogpt_moe/grad_em.py::grad_em_reference` is a small detached PyTorch
value oracle, not an autograd implementation or an optimized training path.
It takes full logits `z [T,E]`, fixed selected indices `[T,K]`, selected expert
outputs `h [T,K,D]`, and incoming output gradient `g [T,D]`. In FP32:

```text
v[t,i] = sum_d g[t,d] * h[t,i,d]
a = softmax(z_selected)                      # selected K only
q = softmax(z_selected - eta * v)              # selected K only; detached
grad_expert[t,i,d] = q[t,i] * g[t,d]
grad_z_selected = (a - q) / eta
grad_z_unselected = 0
```

The router rule is the exact logits gradient of
`(1 / eta) * KL(q || a)` with q detached and selected support fixed.
There is **no full-E softmax in Grad-EM backward**. Forward routing still uses
its existing full-E probabilities/top-k/normalization, unchanged.

Products are FP32 before summation, even for BF16 inputs. Results are FP32;
the Stage-2A boundary casts gradients to their original activation dtypes.
No inactive expert outputs are evaluated. No gradient graph is retained.
This replacement backward is not the derivative of the unchanged forward:
numerical forward gradcheck is not its correctness oracle.

Eta must be finite, numeric (not bool), and strictly positive. At `g=0`, v=0
and q=a: both expert and router gradients are exactly zero. K=1 also gives
zero router gradient; K=E uses the same selected-support formula.
As eta approaches zero, `(a-q)/eta` approaches
`a * (v - (a*v).sum(-1, keepdim=True))`, the ordinary **normalized Top-K**
router gradient with fixed support, and q*g approaches a*g. Eta=0 itself is
rejected. The regression checks this against ordinary forward autograd in
FP64 at eta=1e-5, keeping cancellation error below the O(eta) truncation;
a separate FP32 oracle test uses eta=1e-3. Very small eta in FP32 can amplify
softmax/subtraction roundoff; no stability approximation or new limit branch
is introduced. The recovery statement assumes normalized Top-K forward.
Using selected logits avoids underflow from taking `log(topk_weights)`;
in exact arithmetic `log(p_selected)` differs only by a common shift.

## Configuration and compatibility

`[model].moe_backward` defaults to `"standard"`; `"grad_em"` is opt-in and
requires MoE. `grad_em_eta` defaults to `0.1`, must be finite and positive,
and is fixed (no schedule). Existing TOMLs remain unchanged. Stage 2A supports
the grouped-GEMM combine boundary on CPU (tests supply a differentiable CPU
GEMM double; the external CUDA extension itself has no CPU fallback).
The loop backend explicitly rejects Grad-EM. CPU/CUDA devices are supported;
the prior CUDA implementation was enabled after user-reported GH200 validation.
This selected-support correction needs fresh GH200 validation.

Resolved experiment/checkpoint configs record both fields. Compatibility
checks interpret missing legacy fields as `"standard"` / `0.1`, without
mutating the checkpoint. Explicit mode/eta mismatches are rejected. No model
state keys or checkpoint format version change.

Do not treat old Grad-EM checkpoints/results as continuations of this corrected
algorithm: their router rule differed. Config mode/eta alone do not distinguish
these semantics, so retain the code revision with every result and start fresh
comparisons. This patch does not add checkpoint migrations or version fields.

## Custom combine boundary

The boundary is `MoE._forward_grouped_gemm`'s call to
`combine_expert_outputs(out_sorted, topk_weights, order)` in `model.py`, currently
backed by `_combine.py::_FusedCombine` on CUDA for standard mode. The opt-in
`GradEMCombine` Function reuses that same helper's forward (on CPU in Stage 2A).
It saves only `out_sorted`, original `router_logits`, `topk_experts`, and `order`,
plus fixed eta as a scalar. It does not save the mixing weights or any new
inverse permutation. The forward uses exactly the existing mixing weights.

Backward unsorts the selected expert outputs into `[T,K,D]`, invokes the
Stage-1 FP32 oracle, and gathers q*g back to expert-sorted order. The existing
FC2/activation/FC1 graph receives that gradient unchanged except for casting to
the original output dtype. It scatters `(a-q)/eta` **directly to selected logits** (cast to
the logits dtype) and **None for mixing weights**: ordinary top-k, normalization
and softmax gradients cannot be double-counted. The router linear remains
connected to x, so expert and router input gradients both accumulate normally.
The backward is explicitly once-differentiable; q has no higher-order graph.

Standard mode keeps its existing combine call, with no new autograd boundary,
saved tensors or tensor operations. GEMM, routing, bias, optimizer, kernels and
compilation boundaries are unchanged. No checkpoint keys change.

## CUDA implementation (correction needs GPU validation)

`_grad_em_cuda.py` adds two backward kernels. First, one program per token loads
K sorted rows via the compact inverse permutation and reduces FP32 products
across D (unlike standard backward, no activation-dtype intermediate rounding).
It computes v once, softmaxes selected logits minus eta*v, emits FP32 `[T,K]` q,
and writes q*g directly into unique expert-sorted gradient rows. Second, one
program per token softmaxes only K selected logits to obtain a, matches selected
IDs to `(a-q)/eta` in registers, and writes each full logits-gradient entry
exactly once, with exact zeros outside the support. No atomics, global dense intermediate, global
`[T,K,D]`, gathered incoming-gradient copy, or second expert execution.

CUDA forward launches the **unchanged** `_combine_assignment_rows` and
`_combine_forward` kernels with the same geometry/flags as standard mode. The
inverse permutation is created once and saved instead of order. Other saved
tensors are out_sorted, original logits and selected IDs; eta is a scalar.
Mixing weights are neither saved nor differentiated. CPU uses the same corrected rule.
Expected launches with both gradients: two forward + two backward. Tests can
optionally emit a compact FP32 v buffer; normal backward does not allocate it.
Normal scratch is inverse rows (8*T*K bytes, saved) and q (4*T*K bytes,
backward only), besides required outputs/gradients. At T=65536,K=8 these are
4 MiB and 2 MiB, respectively; measured peak memory is still unknown.

Candidate supports FP32/BF16/FP16, strided inputs, expanded incoming gradients,
and empty token batches. Current explicit tile bounds: K<=32, D<=4096, E<=1024,
rounded K*D<=32768. Higher-order backward remains unsupported. No performance
claim: register pressure and the extra router kernel must be measured on GH200.

Tests use the public CUDA path without a guard bypass. Re-run acceptance for
this mathematical correction before training; old GPU results do not validate
the new rule. This Mac cannot compile/execute Triton CUDA or establish correctness.

On an existing Vista GH200 allocation, from the repository root:

```bash
cd "$WORK/projects/modded-nanogpt-moe"
module load nvidia/25.3 cuda/12.9
export CC=/usr/bin/gcc CXX=/usr/bin/g++
MOE_GMM_IMPLEMENTATION=torch uv run --no-sync python -m pytest -q -rs \
  tests/test_grad_em_cuda.py
uv run --no-sync python -m pytest -q -rs \
  tests/test_grad_em.py tests/test_grad_em_integration.py \
  tests/test_combine.py tests/test_packed_experts.py tests/test_package.py
```

Inspect any skip reasons, especially missing grouped GEMM. Record device,
PyTorch/Triton versions and results. No GPU tests/smokes/benchmarks have been run
for this selected-support correction in the current local session.

### Smokes and benchmark (only AFTER validation of the current rule)

Create a temporary config differing from E64/K8 only in run identity/mode/eta:

```bash
source scripts/vista/env.sh
grad_em_config="$(mktemp /tmp/moe-grad-em-XXXXXX.toml)"
uv run --no-sync python - "$grad_em_config" <<'PY'
import sys
from pathlib import Path
text = Path("configs/moe_e64k8_r0.5.toml").read_text()
assert 'moe_backward' not in text and 'grad_em_eta' not in text
text = text.replace('run_name = "moe-e64k8-r0.5"', 'run_name = "moe-e64k8-r0.5-grad-em"')
text = text.replace('[model]\n', '[model]\nmoe_backward = "grad_em"\ngrad_em_eta = 0.1\n')
Path(sys.argv[1]).write_text(text)
PY
log_dir="$STOCKYARD/logs/modded-nanogpt-moe/vista/grad-em-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$log_dir"
set -o pipefail
scripts/vista/train.sh --steps 2 "$grad_em_config" 2>&1 | tee "$log_dir/smoke-2.log"
# Run the next command only if the two-step run succeeds with finite loss.
scripts/vista/train.sh --steps 5 "$grad_em_config" 2>&1 | tee "$log_dir/smoke-5.log"
grep -E 'val_loss|step:|mem_alloc_peak|mem_reserved_peak|Traceback|Error' "$log_dir"/smoke-*.log

BENCHMARK_WARMUP_UPDATES=10 BENCHMARK_MEASURED_UPDATES=30 \
  scripts/vista/benchmark.sh configs/moe_e64k8_r0.5.toml 2>&1 | tee "$log_dir/standard.log"
BENCHMARK_WARMUP_UPDATES=10 BENCHMARK_MEASURED_UPDATES=30 \
  scripts/vista/benchmark.sh "$grad_em_config" 2>&1 | tee "$log_dir/grad-em.log"
grep -E 'model_dim=|gmm_implementation=|warmup_updates=|ms/update=|tokens/sec=|peak_.*GiB=' \
  "$log_dir/standard.log" "$log_dir/grad-em.log"
```

Keep config microbatch=64 and accumulation=8 identical. Unset stale experiment,
checkpoint and Nsight overrides if the benchmark launcher rejects them. Report
both mean/median ms, tokens/s and peak allocated/reserved GiB. Runtime overhead
is `100*(grad_em/standard - 1)` for matching timing statistics; memory overhead
is the difference of each corresponding peak. Two/five-step timings include
startup and are not a steady-state benchmark. No numbers are inferred here.

CPU verification:

```bash
uv run --no-sync python -m pytest -q tests/test_grad_em.py
uv run --no-sync python -m pytest -q tests/test_grad_em_integration.py
uv run --no-sync python -m pytest -q -rs tests/test_grad_em_cuda.py
uv run --no-sync python -m pytest -q tests/test_package.py
git diff --check
```
