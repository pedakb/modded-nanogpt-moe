# TensorBoard training diagnostics

Requires the existing rank-zero TensorBoard writer (`TB_ROOT` nonempty). Optional
TOML settings, with these defaults (existing experiment files need no edits):

```toml
[diagnostics]
scalar_interval = 10
histogram_interval = 0
during_nsys = false
```

`scalar_interval = 0` disables all new training diagnostics. Histograms are
disabled by default; use e.g. 100 for a lower-frequency histogram. A nonzero
histogram interval must be a multiple of the scalar interval. Nsight disables
diagnostics for the **entire run**, unless `during_nsys = true` explicitly opts
in. That opt-in contaminates profile timings with diagnostics; use only when
intended. `TRAINING_BENCHMARK=1` always disables diagnostics, even with that
opt-in. Nonzero ranks and runs without a writer never instantiate the observer.

## Metrics and meanings

Sampled training diagnostics use the number of **completed optimizer updates**
(10, 20, ... by default); resuming preserves that cadence. All TensorBoard tags
use only `metric/`, `opt/`, `router/`, and `perf/`. Layer names are zero-padded
`l00`, `l01`, ..., `l11`. Historical event files are not migrated and no legacy
aliases are emitted; resuming an older run retains its historical old tags.

| Prefix | Metrics |
| --- | --- |
| `opt/router/l00/` | `param_norm`, `grad_norm`, `grad_ratio`, `update_norm`, `update_ratio`, plus `param_rms`, `grad_rms`, `update_rms`, `dlogit_rms` (actual local logit gradients, when backward traverses the router) |
| `opt/expert/l00/fc1/` (also `fc2`) | Five L2/relative metrics with `_min`, `_med`, `_mean`, `_max`, `_std`, `_p10`, `_p90`. `param_rms`, `grad_rms`, `update_rms` log only `_p10`, `_med`, `_p90` |
| `opt/attn/`, `opt/embed/`, `opt/head/` | Same parameter norms/ratios and RMS metrics as routers, aggregated per category |
| `opt/compare/l00/fc1/` (also `fc2`) | `router_over_expert_grad_norm`, `router_over_expert_update_ratio` |
| `router/l00/` | `entropy`, `entropy_norm`, `logit_rms`, `top1_prob`, `top1_top2_gap`, `topk_margin`, `topk_margin_med` |
| `router/l00/load/` | `min`, `max`, `mean`, `std` (assignment fractions), `cv`, `entropy`, `entropy_norm`, `zero`, `max_mean`; optional `fraction` histogram |
| `router/l00/margin/` | `lt_001`, `lt_005`, `lt_01`: fractions with margin below 0.01, 0.05, 0.1, respectively |
| `metric/loss/train` | Rank-local summed training cross-entropy divided by rank-local tokens, on sampled updates |
| `metric/loss/val` | Validation loss at the existing validation cadence; no duplicate aliases |
| `opt/lr/adamw/` | `g0`, `g1`, `g2`, generalized to actual group count |
| `opt/lr/muon` | Single-group Muon LR; multiple groups use `opt/lr/muon/g0`, `g1`, ... without an aggregate alias |
| `perf/` | `step_ms`, `train_s`, `tok_s`, from existing training-time accounting |

LR values retain their existing zero-based update index and cadence. Performance
tags retain their existing write points: `train_s` is cumulative training time;
`step_ms` is the cumulative average at updates/resume, and the interval average
at validation. `tok_s` is global batch tokens divided by that same average step
duration (NaN when unavailable). No extra timers or synchronization are added;
these ordinary-training timings are not the synchronized benchmark measurement.

Router norms include weight and bias together. Expert FC1/FC2 norms are weights
only, one norm per independent expert matrix (both ModuleList and packed
layouts). Attention combines all attention **weights** across layers; embedding
and output-head metrics cover their weights. Norms use FP32 reductions. Expert
median is the midpoint median; std is population std. Median is the primary
typical-expert statistic; p10/p90 describe the central 80% band. Percentiles use
linear interpolation at `(E-1)*q` via kthvalue selection, without a full sort or
host scalar reads. NaNs propagate as for the existing summaries. No per-expert-ID
scalar series are created. New RMS metrics add only three percentile series,
not additional min/max/std/histogram series. Stream count depends on layers,
not number of experts.

RMS is the corresponding L2 norm divided by `sqrt(numel)`. Routers include the
weight and bias element counts together. For experts, normalize each independent
matrix **before** computing p10/median/p90; packed `[E,D,H]` weights divide by
`sqrt(D*H)`, not `sqrt(E*D*H)`. Other categories use their combined element count.
All existing L2 quantities and relative ratios are unchanged. RMS reuses their
already-computed norms; no extra parameter/gradient scans or snapshots.

Parameter norm is measured **before** the update. Gradient norm is observed after
all accumulation and the trainer's existing SUM all-reduce, before either
optimizer can mutate gradients. It is the actual gradient used by the trainer,
not additionally averaged or rescaled for logging. Ratios divide by the pre-step
parameter norm. Undefined ratios (zero denominator, e.g. zero-initialized FC2)
are logged as NaN, rather than misleading zeros or arbitrary epsilon-scaled
values. NaNs propagate into expert summaries/comparisons until defined.

Update norm measures `||parameter_after - parameter_before||_2`. Snapshots are
independent detached clones on **sampled updates only**; both AdamW and Muon run
unaltered before the difference is measured. This includes weight decay,
momentum, adaptive scaling, and actual stored-parameter rounding. It is not
`lr * gradient`, and optimizer internals/states are neither read nor changed.
FP32 snapshots are reused as subtraction scratch; low-precision snapshots are
promoted before subtraction. Snapshots are released after emission.

The comparison metrics divide router gradient norm by median expert gradient
norm, and router update/parameter ratio by median expert update/parameter ratio,
separately for FC1 and FC2.

Routing observes **existing** full FP32 probabilities and actual selected IDs,
before optional selected-probability renormalization. Observations are detached,
do not replace routing tensors, and retain no autograd graphs. Full entropy uses
safe clamped logs. `top1_prob` reports the mean maximum probability, without a
duplicate maximum-probability alias.
An additional `topk(k+1)` (at least 2, capped at E) on detached logits supplies
the boundary margin, never a full expert sort. For k=E the boundary does not
exist, so margin tags are omitted; for E=1 the top1-top2 gap is omitted and
normalized entropies are defined as zero.

`logit_rms = sqrt(sum_token population_variance_e(z) / sampled_token_count)`:
subtract each token's mean logit conceptually, then pool centered squared values
over all tokens/experts. It is invariant to a common shift of that token's logits.
Population variance computes the centering without retaining a centered logit
matrix. Pool variances first and take the square root once: do not average
microbatch RMS values. E=1 gives zero. No raw-logit L2 scale metric is added.

`opt/router/l00/dlogit_rms` uses temporary router forward hooks to attach tensor gradient
hooks only during sampled capture. Each backward hook reduces the incoming
gradient to an FP32 sum of squares and increments a Python element count; emission
takes `sqrt(total_squares/total_elements)`. No `retain_grad`, saved logits/full
gradient buffer, extra backward pass, or graph retention. The hook returns None,
leaving gradients unchanged. Hook handles contain weak references, and both
module and tensor hooks are removed in capture's `finally` (also on failure).
No hooks are installed on normal updates or when diagnostics are bypassed.
Gradients are in the actual router-output dtype (including BF16 rounding) and
the trainer's existing summed-loss scale; no per-token rescaling or cross-rank
reduction. If no backward reaches logits (e.g. no-grad), omit the tag rather than
claiming a measured zero. All accumulation passes contribute by element count.

Counts, entropy/probability sums and margins combine **all microbatches in the
sampled update**, not averages of microbatch load statistics. Loads are actual
assignment counts divided by total assignments; empty experts are included.
Margin median uses order-statistic selection, not a full token sort. Optional
histograms record expert param/grad/update norm vectors and load fractions, not
per-token logits or full parameter tensors.

## Overhead and distributed scope

- Non-sampled updates have no diagnostic tensor work, parameter copies, routing
  reductions or transfers. They perform a cadence check; MoE has an inactive
  observer branch. Dense compilation and eager-MoE/compiled-head boundaries stay
  unchanged. Observer callbacks are detached/reset in `finally`, including when
  forward raises. Validation does not collect training routing statistics.
- Sampled updates read monitored parameters/gradients, clone parameters once,
  collect routing statistics, and read post-step parameters. Exact update norms
  need these snapshots; this is **not memory-free**. E64/H384/D768/L12 FP32
  expert snapshots alone take 1.6875 GiB, plus attention, router, embedding and
  head snapshots (roughly 2 GiB total with the standard model/dtypes). Mixed-dtype
  subtraction and routing reductions also require temporary storage.
- Margin retention is one FP32 value per rank-local token per MoE layer, about
  24 MiB for 524288 tokens and 12 layers; emission temporarily concatenates each
  layer's margins. Full probabilities/logits are not retained. Routing reductions
  incur extra launches on sampled microbatches only. Lowering logging frequency
  reduces average overhead but not sampled-update peak memory.
- The follow-up scale metrics add a centered variance and a gradient norm
  reduction per sampled router call, small percentile selections, and divisions
  of existing parameter norms. Persistent extra storage is scalar accumulators,
  tiny cached RMS divisors, and weak hook handles only. FP32 casts/reductions may
  need temporary workspace (e.g. a BF16 logit tensor is ~16 MiB when promoted for
  N=65536/E64), but logits/gradients are not retained across microbatches. Existing
  snapshot size and histogram contents are unchanged; no new device-to-host transfers or
  synchronization calls. Measure CUDA overhead before making runtime claims.
- Scalar values transfer to CPU in one batch per logged update; optional
  histogram vectors use a second batch. No explicit CUDA synchronization or
  per-microbatch `.item()`/CPU transfers are added. Transfers necessarily wait
  for their results. Routine training timing includes diagnostic overhead;
  benchmark timing code and regions are unchanged and bypass all diagnostics.
- Rank 0 alone collects/writes. Routing, logit-gradient RMS and `metric/loss/train` are **rank-local**;
  gradient norms are after the pre-existing all-rank SUM, parameters/updates
  reflect the synchronized model. No new collectives. Do not interpret local
  utilization as whole-world utilization in distributed runs.
- Diagnostics do not consume randomness, register buffers, enter optimizer
  state, or enter checkpoint compatibility configuration. Checkpoint size and
  restore/rotation/purge semantics are unchanged. Settings appear in the printed
  experiment config and diagnostics source is included in source logging.

Local tests exercise deterministic metrics, real AdamW/Muon stored updates,
model/gradient/state parity, cleanup, and benchmark/profile gating. CUDA peak
memory/runtime and event-file inspection remain to be measured on Vista/LS6;
no negligible-overhead claim is made. Increase `scalar_interval` or disable
diagnostics if sampled-update peak memory exceeds a run's available headroom.
