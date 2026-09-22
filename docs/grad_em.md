# Grad-EM: reference, CPU integration and guarded CUDA candidate

`modded_nanogpt_moe/grad_em.py::grad_em_reference` is a small detached PyTorch
value oracle, not an autograd implementation or an optimized training path.
It takes full logits `z [T,E]`, fixed selected indices `[T,K]`, selected expert
outputs `h [T,K,D]`, and incoming output gradient `g [T,D]`. In FP32:

```text
v[t,i] = sum_d g[t,d] * h[t,i,d]
q = softmax(z_selected - eta * v)              # selected K only; detached
p = softmax(z)                               # all E
q_tilde[t, selected_idx[t,i]] = q[t,i]         # zero elsewhere
grad_expert[t,i,d] = q[t,i] * g[t,d]
grad_logits = q_tilde - p                     # this sign is intentional
```

Products are FP32 before summation, even for BF16 inputs. Results are FP32;
the Stage-2A boundary casts gradients to their original activation dtypes.
No inactive expert outputs are evaluated. No gradient graph is retained.
This replacement backward is not the derivative of the unchanged forward:
numerical forward gradcheck is not its correctness oracle.

At `eta=0`, q is the selected-logit softmax, but the router replacement is
still `q_tilde-p`, not ordinary routing backward. At `g=0`, expert gradients
are zero while router gradients generally remain nonzero. This is intentional.
Using selected logits avoids underflow from taking `log(topk_weights)`;
in exact arithmetic `log(p_selected)` differs only by a common shift.

## Configuration and compatibility

`[model].moe_backward` defaults to `"standard"`; `"grad_em"` is opt-in and
requires MoE. `grad_em_eta` defaults to `0.1`, must be finite and nonnegative,
and is fixed (no schedule). Existing TOMLs remain unchanged. Stage 2A supports
the grouped-GEMM combine boundary on CPU (tests supply a differentiable CPU
GEMM double; the external CUDA extension itself has no CPU fallback).
The loop backend explicitly rejects Grad-EM. **CUDA training rejects it before
device/distributed setup** until the Stage-2B candidate passes GPU validation.

Resolved experiment/checkpoint configs record both fields. Compatibility
checks interpret missing legacy fields as `"standard"` / `0.1`, without
mutating the checkpoint. Explicit mode/eta mismatches are rejected. No model
state keys or checkpoint format version change.

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
the original output dtype. It returns q_tilde-p **directly to logits** (cast to
the logits dtype) and **None for mixing weights**: ordinary top-k, normalization
and softmax gradients cannot be double-counted. The router linear remains
connected to x, so expert and router input gradients both accumulate normally.
The backward is explicitly once-differentiable; q has no higher-order graph.

Standard mode keeps its existing combine call, with no new autograd boundary,
saved tensors or tensor operations. GEMM, routing, bias, optimizer, kernels and
compilation boundaries are unchanged. No checkpoint keys change.

## Stage 2B candidate (GPU validation pending)

`_grad_em_cuda.py` adds two backward kernels. First, one program per token loads
K sorted rows via the compact inverse permutation and reduces FP32 products
across D (unlike standard backward, no activation-dtype intermediate rounding).
It computes v once, softmaxes selected logits minus eta*v, emits FP32 `[T,K]` q,
and writes q*g directly into unique expert-sorted gradient rows. Second, one
program per token softmaxes all E logits, matches selected IDs to q in
registers, and writes q_tilde-p directly. No atomics, global q_tilde, global
`[T,K,D]`, gathered incoming-gradient copy, or second expert execution.

CUDA forward launches the **unchanged** `_combine_assignment_rows` and
`_combine_forward` kernels with the same geometry/flags as standard mode. The
inverse permutation is created once and saved instead of order. Other saved
tensors are out_sorted, original logits and selected IDs; eta is a scalar.
Mixing weights are neither saved nor differentiated. CPU behavior is unchanged.
Expected launches with both gradients: two forward + two backward. Tests can
optionally emit a compact FP32 v buffer; normal backward does not allocate it.
Normal scratch is inverse rows (8*T*K bytes, saved) and q (4*T*K bytes,
backward only), besides required outputs/gradients. At T=65536,K=8 these are
4 MiB and 2 MiB, respectively; measured peak memory is still unknown.

Candidate supports FP32/BF16/FP16, strided inputs, expanded incoming gradients,
and empty token batches. Current explicit tile bounds: K<=32, D<=4096, E<=1024,
rounded K*D<=32768. Higher-order backward remains unsupported. No performance
claim: register pressure and the extra router kernel must be measured on GH200.

**Release gate:** all three public guard calls remain. The CUDA path is wired
but tests alone monkeypatch the guard. There is no training environment bypass.
Do not remove the gate or start smokes until the GPU tests pass without relevant
skips. This Mac cannot compile/execute Triton CUDA or establish correctness.

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
PyTorch/Triton versions and results. Only then replace the CPU-only guard with
device validation admitting CUDA and update guard-specific tests in a reviewed
follow-up. No GPU tests/smokes/benchmarks have been run for this candidate.

### Smokes and benchmark (only AFTER validation and guard enablement)

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
