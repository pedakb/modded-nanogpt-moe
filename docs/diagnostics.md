# TensorBoard training diagnostics

TensorBoard requires the existing rank-zero writer (`TB_ROOT` nonempty). The
diagnostic settings remain:

```toml
[diagnostics]
scalar_interval = 10
histogram_interval = 0
during_nsys = false
```

Production experiments can set `scalar_interval = 25`. A value of zero disables
the sampled heavy diagnostics but does not disable the ordinary loss, learning
rate, validation, or performance scalars. `histogram_interval` remains accepted
for configuration compatibility, but the compact diagnostics emit no histogram
series.

Benchmark mode always bypasses heavy diagnostics. Nsight bypasses them for the
whole run unless `during_nsys = true`; opting in contaminates profile timing.
Nonzero ranks and runs without a TensorBoard writer do not construct an observer.

## Scalar surface

Ordinary trainer scalars are:

| Tag | Cadence |
| --- | --- |
| `metric/loss/train` | Every completed optimizer update |
| `metric/loss/val` | Existing validation cadence |
| `perf/step_ms`, `perf/tok_s` | Existing update, validation, and resume write points |
| `opt/lr/adamw/g0`, `g1`, `g2`; `opt/lr/muon` | Every optimizer update |

`perf/train_s` is not emitted. Training loss is the rank-local summed
cross-entropy divided by rank-local tokens. LR values retain their existing
zero-based update index; the other training scalars use completed-update steps.

Heavy diagnostics are sampled on completed updates divisible by
`scalar_interval`. The model-wide tags are:

- `opt/global/param_rms`
- `opt/global/grad_rms`
- `opt/global/update_rms`
- `opt/global/update_ratio`

Only representative transformer layers `l00`, `l05`, and `l11` emit per-layer
heavy metrics. Each emits:

- `opt/router/lXX/{param_rms,grad_rms,update_rms,update_ratio,dlogit_rms,sens_range_med}`
- `opt/expert/lXX/{param_rms_med,grad_rms_med,update_rms_med,update_ratio_med}`
- `router/lXX/{entropy_norm,entropy_post_norm,logit_range_med,topk_margin_med}`
- `router/lXX/load/{cv,zero}`

The standard 12-layer, three-AdamW-group run therefore has exactly 60 unique
scalar series: 8 ordinary series, 4 global diagnostics, and 16 diagnostics for
each of three representative layers. No legacy aliases are emitted and old
event files are not migrated.

## Optimization definitions

Global metrics include every trainable model parameter included in optimization.
They use true parameter-count weighting, not averages of module-level RMS values:

```text
param_rms    = sqrt(sum(theta^2) / total_numel)
grad_rms     = sqrt(sum(grad^2) / total_numel)
update_rms   = sqrt(sum(delta^2) / total_numel)
update_ratio = sqrt(sum(delta^2)) / sqrt(sum(theta^2))
```

Parameter values are measured before the update. Gradients are measured after
accumulation and the trainer's existing SUM all-reduce, before either optimizer
can mutate them. Missing gradients contribute zero while their parameter count
remains in the global denominator.

Update values are actual stored changes (`parameter_after - parameter_before`).
They therefore include momentum, adaptive scaling, weight decay, and stored-dtype
rounding. The observer does not infer updates from learning rates or inspect
optimizer state. Undefined relative updates from a zero denominator are NaN.

Router metrics combine router weight and bias. Their RMS values divide by the
combined element count, and their update ratio uses the combined squared sums.

Expert metrics preserve the former inclusion rule: FC1 and FC2 **weights** are
included, while expert biases are not. For each expert, FC1 and FC2 squared sums
and element counts are combined first:

```text
expert_param_rms[e] =
    sqrt((sum(fc1_theta[e]^2) + sum(fc2_theta[e]^2)) /
         (numel(fc1[e]) + numel(fc2[e])))
```

Gradient RMS, actual-update RMS, and relative update are combined the same way.
Only then is the midpoint median taken across experts. This applies identically
to ModuleList and packed layouts; it is not an average of FC1 and FC2 RMS values.

## Routing definitions

Routing observes the existing full FP32 probabilities, selected IDs, and
selected weights after optional Top-K renormalization. It does not replace or
modify routing tensors.

- `entropy_norm` is the mean per-token entropy of the full E-way routing
  probabilities divided by `log(E)`; it is zero for `E=1`.
- `entropy_post_norm` is the mean per-token entropy of the normalized selected
  Top-K weights divided by `log(K)`; it is zero for `K=1`.
- `logit_range_med` is the midpoint median across tokens of each token's
  maximum router logit minus its minimum router logit.
- `topk_margin_med` is the midpoint median of the `k` versus `k+1` logit
  boundary. It uses `topk(k+1)`, not a full sort, and is omitted for `k=E`.
- `load/cv` is the population standard deviation of assignment fractions divided
  by their mean.
- `load/zero` counts experts receiving no assignments across the sampled update.
- `dlogit_rms` is the FP32 RMS of actual local router-logit gradients across all
  sampled microbatches. Temporary hooks retain scalar sums and counts, not logits,
  gradients, or autograd graphs. The tag is omitted if backward never reaches the
  logits.
- `sens_range_med` is the midpoint median across tokens of
  `max_i(dL/da_i)-min_i(dL/da_i)` on the selected support. Standard backprop
  observes the gradient already produced for normalized Top-K weights. Grad-EM
  records its already-computed FP32 `v` from the custom backward kernel/reference;
  neither path performs a second backward or recomputes expert dot products.

Counts, entropy sums, ranges, margins, and gradient sums combine all
microbatches in the sampled optimizer update. Routing and gradient metrics are
rank-local.

## Cost and state

Non-sampled updates do no heavy diagnostic parameter scans, snapshots, routing
reductions, hooks, or batched transfers. They only perform the cadence check and
the ordinary training-loss logging requested for every update.

On sampled updates, exact global update metrics still require one detached clone
of every optimized parameter. Representative router/expert groups reuse those
same snapshots rather than making additional copies. For the standard packed
E64 model the snapshot remains roughly 2 GiB, so reducing cadence lowers average
cost but not sampled-update peak memory.

Compared with the former diagnostics, only 3 of 12 layers perform routing
observation, boundary top-k, margin retention, and logit-gradient hooks. The
observer no longer computes raw L2 series, min/max/mean/std/p10/p90 summaries,
separate FC1/FC2 summaries, router comparison metrics, centered-logit RMS,
threshold fractions, load entropy/extrema, attention/embedding/head groups, or
histograms. The full-model parameter/gradient/update reductions remain necessary
for the four global metrics.

Sampled scalar values, including training loss, use one batched device-to-host
transfer. No explicit CUDA synchronization or new distributed collective is
added. Diagnostics consume no randomness, register no buffers, and do not enter
model, optimizer, checkpoint, or compatibility state.
