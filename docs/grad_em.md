# Grad-EM: Stage 1 reference contract

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
future production integration must handle casts at the activation boundary.
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
and is fixed (no schedule). Existing TOMLs remain unchanged. Stage 1 accepts
Grad-EM config for reference work but **training rejects it before CUDA setup**
until the production backward is implemented.

Resolved experiment/checkpoint configs record both fields. Compatibility
checks interpret missing legacy fields as `"standard"` / `0.1`, without
mutating the checkpoint. Explicit mode/eta mismatches are rejected. No model
state keys or checkpoint format version change.

## Future integration (not implemented)

The boundary is `MoE._forward_grouped_gemm`'s call to
`combine_expert_outputs(out_sorted, topk_weights, order)` in `model.py`, currently
backed by `_combine.py::_FusedCombine` on CUDA. A separate opt-in custom combine
must retain access to full router logits and fixed support, use the existing
sorted-output metadata to compute FP32 v/q, return q*g to the existing grouped
expert graph, and return q_tilde-p **directly to logits**, not through the
ordinary top-k-weight/softmax Jacobian. Standard combine remains unchanged.
No grouped-GEMM, routing selection, bias, optimizer, or kernel changes are
needed in Stage 1. Future kernel tests must compare against this oracle and
cover router gradients, rounding/casts, empty experts, and imbalanced support.

CPU verification:

```bash
uv run --no-sync python -m pytest -q tests/test_grad_em.py
uv run --no-sync python -m pytest -q tests/test_package.py
git diff --check
```
