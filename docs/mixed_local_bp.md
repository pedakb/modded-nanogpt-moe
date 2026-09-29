# Mixed local-BP backward

Validated on 2026-09-29 against `1928123` on an idle Vista GH200, using
PyTorch 2.11.0+cu129 / CUDA 12.9. This replaces the grouped local-BP backward
described in [the earlier report](grouped_local_bp.md). The loop reference,
standard BP, global Grad-EM, forward values, configuration/checkpoint format,
optimizer definitions and launcher/backend defaults are unchanged.

## Routing convention and numerical contract

The actual forward weight `p` is full-expert FP32 softmax followed by Top-K,
optional selected-weight normalization, then the activation-dtype cast.
Grad-EM instead uses `a = softmax(selected_logits)` and
`q = softmax(selected_logits - eta * dot(g, expert_output))`, both in FP32.
Its router signal is `(a-q)/eta` on selected logits and zero elsewhere.

For each selected token/expert pair, linearity gives
`W2.T @ (q*g) = (q/p) * (W2.T @ (p*g))` when `p > 0`. The same scalar commutes
through the elementwise ReLU-squared derivative. This identity also holds for
`normalize_topk=False`: dividing by `a` instead of the actual forward `p`
would incorrectly omit the selected probability mass.

The identity is exact in real arithmetic. Moving scaling across a rounded
BF16 GEMM changes FC1 parameter-gradient rounding; it does not preserve
bitwise GE FC1 gradients. The input VJP retains ordinary BP's operation and
rounding order, and FC2/router parameter signals retain the existing GE rule.
Established tolerances stay unchanged: FP32 `atol=2e-6, rtol=2e-5`, full-MoE
BF16 `atol=rtol=0.02`. FC2/router and the BP boundary retain exact comparisons.

GE FC2 wgrad receives `q*g` directly, avoiding a second rounding through
`(q/p)*(p*g)`. GE hidden signals use FP32 `(v_BP/p)*q`, avoiding overflow from
forming `q/p` first. For a zero selected forward weight, the single dgrad uses
an unweighted working signal for that row; its BP hidden signal is zeroed and
its GE signal is scaled by q. No division by zero or unselected weights occurs.

## Backward structure

- Reuse the original packed inputs, preactivations, activations, weights,
  counts, offsets and permutation. Combine once and retain one inverse lookup.
- Form BP and GE expert-output signals and both inexpensive router signals.
- FC2 wgrad/bias use GE; one FC2 dgrad uses BP.
- Compute activation backward once; derive GE hidden signals by rescaling.
- FC1 wgrad/bias use GE; one FC1 dgrad uses BP, followed by the original gather VJP.
- The parameter-only router linear receives GE. Its input is detached, so it
  never computes GE router dgrad. The original softmax/Top-K/normalization
  graph supplies BP router dgrad through `RouterInputOnly`.

Actual backend instrumentation, outside timing, confirms these per-layer counts:

| Mode | FC1 forward/dgrad/wgrad | FC2 forward/dgrad/wgrad |
|---|---|---|
| Standard BP / global GE | 1 / 1 / 1 | 1 / 1 / 1 |
| Previous local BP | 1 / 1 / 1 | 1 / 2 / 1 |
| Mixed local BP | 1 / 1 / 1 | 1 / 1 / 1 |

The extra GE FC2 dgrad, second activation VJP, and redundant ordinary combine
forward are eliminated. Router tests count one wgrad and one BP dgrad.
If x does not require a gradient, the existing parameter-only GE path remains.

## Correctness results

`test_grad_em_grouped_local_bp.py` and `test_local_bp_kernels.py`:
**160 passed, 7 skipped** (116 CPU tests and 44 CUDA tests). Seven extension
FP32 cases skip its explicit BF16-only rejection. Tests cover both grouped
APIs, E1/K1, E8/K2, E64/K8, uneven routing, empty experts, normalization on/off,
zero selected weights with nonzero q, frozen parameters, absent input gradients,
zero upstream signals, two successive MoEs, router-only VJPs, and GEMM counts.

Maximum parameter differences from the unchanged grouped GE parameter path
(same upstream signal), over normalization on/off and both grouped APIs:

| Case | FC1 parameters | FC2 parameters | Router parameters | Input vs BP |
|---|---:|---:|---:|---:|
| CPU FP32, E8/K2, D16/H32 | 2.086163e-7 | 0 | 0 | 0 |
| CPU BF16, E8/K2, D16/H32 | 0.00390625 | 0 | 0 | 0 |
| GH200 BF16, E8/K2, D768/H1536 | 0.001953125 | 0 | 0 | 0 |
| GH200 BF16, E64/K8, D768/H384 | 0.0009765625 | 0 | 0 | 0 |

Outputs match grouped standard/global exactly. A separate direct comparison
loaded the previous grouped implementation from `1928123`: E1/K1, E8/K2 and
E64/K8, both normalization settings, identical state/input/upstream, 14 tokens.
It confirmed exactly equal forward/input/FC2/router results and the GPU FC1
differences above; all E1/K1 differences were zero. Parameter keys also matched.

Full suite: **813 passed, 9 skipped, 6 failed**. Five unchanged launcher-parser
tests fail with `BASH_SOURCE[0]: unbound variable`; all five reproduce in an
extracted `1928123` baseline. The sixth is the unchanged bias profiler test
missing CUDA kernel events in the full suite; its isolated rerun passed.
Global Grad-EM and ordinary BP regression tests passed. `git diff --check` passed.

## Timing and memory

One complete packed MoE layer; D768, 65,536 tokens, BF16 activations / FP32
parameters, eta=0.1, native `MOE_GMM_IMPLEMENTATION=torch`. E8 uses H1536/K2;
E64 uses H384/K8. Each mode ran in a fresh process, with 10 warmups and 30
measured iterations, device-wide fences around forward/backward, no profiler
or optimizer. Peaks include model/input/upstream/gradients and all measured
iterations; allocator peaks reset after warmup and cache clear. Backend call
counting and isolated dgrad measurements run after recording timing/peaks.

| Geometry | Mode | Forward median, ms | Backward median, ms | Peak allocated, GiB | Peak reserved, GiB |
|---|---|---:|---:|---:|---:|
| E8/K2 | Standard BP | 3.489 | 4.119 | 2.518 | 3.004 |
| E8/K2 | Global GE | 3.454 | 4.122 | 2.518 | 3.004 |
| E8/K2 | Previous local BP | 3.712 | 6.023 | 3.270 | 3.756 |
| E8/K2 | Mixed local BP | 3.498 | 3.770 | 2.726 | 3.178 |
| E64/K8 | Standard BP | 6.799 | 6.033 | 4.634 | 4.656 |
| E64/K8 | Global GE | 6.769 | 6.035 | 4.634 | 4.656 |
| E64/K8 | Previous local BP | 8.816 | 8.478 | 5.591 | 6.156 |
| E64/K8 | Mixed local BP | 6.871 | 5.880 | 4.392 | 4.658 |

Measured backward time fell 37.4% / 30.6% for E8 / E64; allocated peaks fell
0.543 / 1.199 GiB. These are layer measurements, not training throughput.
The remaining local-specific work is pointwise hidden rescaling, simultaneous
BP/GE signal buffers, and the ordinary router VJP alongside the GE router
signal. Activation fusion and graph removal also affect time/memory, so this
change is not solely the removal of one GEMM. Packing/count synchronization
and expert GEMMs remain shared costs across modes.

Reproduce the current implementation on an idle allocated GH200:

```bash
source scripts/vista/env.sh
uv run --no-sync python -m pytest -q -rs -s \
  tests/test_grad_em_grouped_local_bp.py tests/test_local_bp_kernels.py
for geometry in e8 e64; do
  for mode in standard global local_bp; do
    uv run --no-sync python -m tools.benchmark_grad_em \
      --implementation torch --geometry "$geometry" --mode "$mode" \
      --output "/tmp/mixed-bp-${geometry}-${mode}.json" || break
  done
done
```

For the previous local-BP baseline, run the same benchmark script with its
package imports directed to an extracted `1928123` package. Session JSON
artifacts are `/tmp/mixed-bp-current-{e8,e64}-{standard,global,local_bp}.json`
and `/tmp/mixed-bp-old-{e8,e64}-local_bp.json`; they include individual samples
and backend call counts. Contended measurements from the previous allocation
were discarded and are not used here.
