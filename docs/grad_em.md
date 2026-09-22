# Grad-EM: reference contract and Stage 2A CPU integration

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
device/distributed setup** until the optimized Stage-2B kernel is implemented.

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

## Stage 2B (not implemented)

The CPU boundary deliberately materializes `[T,K,D]`, never `[T,E,D]`. It rejects
non-CPU tensors before combining; the MoE also guards before router/expert
execution. There is no memory-heavy GPU fallback. Stage 2B must use the existing
sorted outputs/metadata for an efficient CUDA implementation, retain the
existing fused forward, and compare its gradients/casts against the oracle and
CPU integration tests. CUDA correctness, memory and performance remain pending.

CPU verification:

```bash
uv run --no-sync python -m pytest -q tests/test_grad_em.py
uv run --no-sync python -m pytest -q tests/test_grad_em_integration.py
uv run --no-sync python -m pytest -q tests/test_package.py
git diff --check
```
