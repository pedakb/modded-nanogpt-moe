# Grouped local-BP Grad-EM

This page records the initial implementation. The current optimized backward
and new correctness/timing results are in [Mixed local-BP backward](mixed_local_bp.md).

Implemented on `cleanup-active-codebase`, based on `18d5d5a`. Measurements below
were collected on 2026-09-28 on an existing Vista GH200 allocation. No optimizer,
objective, schedule, data, checkpoint keys, initialization, or global Grad-EM
rule changed. The existing loop/ModuleList local-BP implementation is unchanged.

## Inspected backward and implementation choice

The installed `nv-grouped-gemm` autograd function uses separate calls:

- Forward: `backend.gmm(x, W, counts, trans_a=False, trans_b=False)`.
- Input gradient: `backend.gmm(g, W, counts, trans_a=False, trans_b=True)`.
- Weight gradient: `backend.gmm(x, g, counts, trans_a=True, trans_b=False)`.

Vista's native backend independently calls `F.grouped_mm(g, W.mT, offs=offsets)`
for dgrad and `F.grouped_mm(x.mT, g, offs=offsets)` for wgrad. Both already honor
`ctx.needs_input_grad`. Bias gradients are contiguous expert-segment reductions
in `_ExpertSegmentBias`, outside either GEMM API.

Global Grad-EM passes `q*g` into the existing FC2 -> ReLU-squared -> FC1 graph,
so that signal supplies both dgrad and wgrad/bias gradients. Its selected-logit
signal `(a-q)/eta` flows through the router linear into both router parameters
and the MoE input. There is no mixing-weight gradient from `GradEMCombine`.

For local BP, use the algebraically simplified form of the proposed correction:

```text
J_x^T g_GE + J_x^T(g_BP - g_GE) = J_x^T g_BP
```

The implementation directly substitutes the right-hand side. It does not
subtract/add rounded BF16 gradients and does not run a second full backward.

1. Execute the existing grouped expert and router forward on `x.detach()`.
   The unchanged `GradEMCombine` supplies all trainable parameter gradients:
   `q*g` for experts and `(a-q)/eta` for the router.
2. `RouterInputOnly` reuses the computed logits and activation-dtype router
   weight. The existing FP32 softmax, top-k, optional normalization, cast and
   ordinary combine graph produce `r_BP`. Its backward returns only
   `r_BP @ router_weight`; it has no parameter-gradient edge.
3. `ExpertInputOnly` reuses the computed expert outputs, FC1 preactivation,
   activation-dtype weights, counts, offsets and routing permutation. Ordinary
   combine supplies `g_BP`. Two raw dgrad-only calls, separated by the exact
   activation-dtype square/ReLU derivative, compute the expert input VJP.
   Scatter uses the original gather's `index_put_(accumulate=True)` semantics.
   This branch has no wgrad or bias reduction.
4. `LocalBPOutput` returns the original Grad-EM forward value and sends the
   same upstream `g` into both branches. Their only accumulated input gradient
   is the standard-BP expert plus router VJP. Later layers cannot propagate
   a Grad-EM replacement signal across this boundary.

With all parameters/input trainable, expert GEMM counts per MoE are:

| Mode | Forward | Backward dgrad | Backward wgrad |
|---|---:|---:|---:|
| Standard / global | 2 | 2 | 2 |
| Local BP | 2 | 3 | 2 |

Local mode needs both GE and BP FC2 dgrad, but only BP FC1 dgrad. Thus it adds
**one net FC2 dgrad GEMM**, plus an extra activation VJP and ordinary combine
backward. It also computes an additional ordinary combine forward, retains
the BP routing graph and FC1 preactivation, and uses compact routing metadata
for both combines. No router/expert forward is repeated. If input gradients
are unnecessary, the input-only branch is omitted. Global/standard paths do
not enter these new autograd functions. Higher-order local-BP backward is
unsupported, as in the loop reference.

## Correctness

`tests/test_grad_em_grouped_local_bp.py`: **87 passed, 3 skipped** on GH200:
73 portable cases and 14 real CUDA cases. The three skips probe the installed
extension's explicit FP32 rejection. The native production backend also
requires BF16. FP32 checks run first using CPU backend API doubles; CUDA
combine/kernel FP32 coverage remains in the existing Grad-EM suite.

Coverage includes FP32/BF16 activations with FP32 master parameters, identical
parameters/support/upstream gradients, E=1, K=1, K=E, E8/K2 and E64/K8,
normalization on/off, empty experts, frozen parameters, absent input gradients,
zero upstream gradients, forward-only/no-grad execution, two successive MoEs,
both grouped APIs, and a GEMM-call count check excluding duplicate wgrad.

Maximum absolute differences from the **loop local-BP reference** in the
fixed numerical-report cases (14 tokens, eta=0.4):

| Case | Output | Input gradient | Expert gradients | Router gradients |
|---|---:|---:|---:|---:|
| CPU FP32, D16/H32, E8/K2 | 0 | 0 | 0 | 0 |
| CPU BF16, D16/H32, E8/K2 | 0.00390625 | 0.001953125 | 0.01171875 | 0.00390625 |
| GH200 BF16, D768/H1536, E8/K2 | 0.005859375 | 0.000244140625 | 0.005859375 | 0.0078125 |
| GH200 BF16, D768/H384, E64/K8 | 0.00390625 | 0.000244140625 | 0.00244140625 | 0.00390625 |

Both backend APIs produced the listed results. The tests retain dtype checking
and the established BF16 full-MoE tolerance (`atol=rtol=0.02`); FP32 uses
`atol=2e-6, rtol=2e-5`. Separately, **local input gradients match standard
grouped BP bit-for-bit**, and **local parameter gradients match global
grouped Grad-EM bit-for-bit**, in these direct same-upstream tests. The two-MoE
test also checks the intermediate/input BP signal and both layers' parameter
gradients against the loop reference; global mode changes that intermediate
signal as expected.

Existing regressions on GH200:

- `tests/test_grad_em_cuda.py` with `MOE_GMM_IMPLEMENTATION=torch`:
  **63 passed**, including explicit global gradient injection.
- `test_native_grouped_gemm.py`, `test_packed_experts.py`, `test_moe.py`,
  `test_combine.py`: **199 passed, 1 skipped** (the missing-extension test).
- Full CPU run: **497 passed, 216 skipped, 5 failed**, before adding the final
  config case and eight CUDA edge cases. The five launcher-parser failures
  (`BASH_SOURCE[0]: unbound variable` in `test_vista_submission.py`) reproduce
  unchanged in files extracted from base commit `18d5d5a`.
- Final full suite with GPU access: **716 passed, 5 skipped, 6 failed**. This
  includes those five launcher failures and
  `test_triton_bias_backward_two_kernels_no_generic_reduction`, whose profiler
  returned no CUDA events in the full run. That unchanged profiler test passes
  when rerun alone (**1 passed**). None of the new local-BP tests failed; the
  complete suite is not claimed green. Five skips are the three unsupported
  FP32 extension cases, Apple MPS, and the missing-extension error case.

## Timing and memory

Single **complete MoE layer**, packed E8/K2, D768/H1536, 65,536 tokens, BF16
activations/FP32 master parameters, eta=0.1, fixed seed 1234. GH200, PyTorch
2.11.0+cu129, native `MOE_GMM_IMPLEMENTATION=torch`. Each mode ran in a fresh
process with 10 warmups and 30 measured forwards/backwards, device-wide
synchronization around each region, no profiler or optimizer. Gradients are
cleared between iterations. Allocator peaks reset after warmup and cache clear;
peaks include model/input/upstream/gradient tensors and all measured iterations.

| Mode | Forward median (mean), ms | Backward median (mean), ms | Peak allocated, GiB | Peak reserved, GiB |
|---|---:|---:|---:|---:|
| Standard BP | 3.167 (3.302) | 4.136 (4.156) | 2.518 | 3.004 |
| Global Grad-EM | 3.244 (3.417) | 4.122 (4.132) | 2.518 | 3.004 |
| Local-BP Grad-EM | 3.683 (3.814) | 6.025 (6.079) | 3.270 | 3.756 |

Local backward adds **1.902 ms (46.1%)** over global in this measurement.
An isolated extra FC2 dgrad using the same geometry/routing counts measured
**0.517 ms median**, about 27% of that difference. The remaining overhead
includes the extra combine/activation VJP, memory traffic, and graph/launch
work; the standalone GEMM measurement is an estimate, not an additive profiler
attribution. Peak allocated/reserved differences are **0.751/0.752 GiB**.
These are layer measurements, not training-update throughput or total model
memory. No optimization or training-speed claim is inferred from them.

Reproduce from the repository root on an existing GH200 allocation:

```bash
source scripts/vista/env.sh
uv run --no-sync python -m pytest -q -rs -s tests/test_grad_em_grouped_local_bp.py
for mode in standard global local_bp; do
  uv run --no-sync python -m tools.benchmark_grad_em \
    --implementation torch --geometry e8 --mode "$mode" \
    --output "/tmp/grouped-local-bp-${mode}.json" || break
done
```

The benchmark JSON includes individual timing samples and both memory peaks.
Original session results are `/tmp/moe-local-bp-benchmark-{standard,global,local_bp}.json`.

## Config and Vista smoke command

`configs/moe_e8k2_r2_gradem_local_bp_packed.toml` matches the existing loop
local-BP experiment except for backend, packed layout and unique run identity.
It sets `moe_backward="grad_em"`, `grad_em_mode="local_bp"`, eta=0.1,
E8/K2/ratio2, and preserves the 3250-update horizon and checkpoint interval 250.

From the repository root on an existing Vista GPU allocation:

```bash
source scripts/vista/env.sh
scripts/vista/train.sh --smoke --steps 2 configs/moe_e8k2_r2_gradem_local_bp_packed.toml
```

The launcher scopes native GEMM to the trainer and stops after two completed
updates while preserving the full schedule horizon. `--smoke` disables
TensorBoard and all checkpoint writes without creating a checkpoint directory;
omit it to retain normal logging and checkpointing. The validation session
validated the layer/backward and ran benchmarks; it did not launch a dataset
training smoke or a production training run.

## Changed files

- `modded_nanogpt_moe/model.py`: grouped local-BP graph selection.
- `modded_nanogpt_moe/_local_bp.py`: parameter-free input VJPs/output boundary.
- `modded_nanogpt_moe/_grouped_gemm.py`: raw dgrad-only backend dispatch.
- `modded_nanogpt_moe/config.py`: remove the obsolete local-BP backend restriction.
- `tests/test_grad_em_grouped_local_bp.py`: parity, boundary and call-count tests.
- `tests/test_grad_em.py`, `tests/test_grad_em_local_bp.py`: configuration acceptance.
- `tools/benchmark_grad_em.py`: reproducible layer latency/memory measurement.
- `configs/moe_e8k2_r2_gradem_local_bp_packed.toml`: portable E8/K2 configuration.
- `README.md`, `docs/grad_em.md`, `docs/grouped_local_bp.md`: usage and results.

The two pre-existing untracked BP/global E8 configuration files were untouched.
