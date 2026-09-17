"""
Standalone correctness harness for genuine dropless grouped-GEMM MoE expert
execution, compared against the existing per-expert Python-loop MoE class
(train_gpt_simple.MoE -- imported unmodified, not reimplemented, not
touched). Does not change train_gpt_simple.py or its (already-integrated,
working) training path.

Candidate backend: fanshiqing/grouped_gemm, PyPI package `nv-grouped-gemm`
(NOT the tgale96 original, NOT padded bmm, NOT TransformerEngine -- see the
INSTALL section below for exactly why, based on source inspection). Nothing
is installed by writing this file.

Routing/packing/combine (softmax -> top-k -> optional renormalize -> pack ->
[expert compute] -> unpack -> weight -> sum) are implemented ONCE, in plain
PyTorch, in GroupedGemmMoE below, and are completely independent of which
GEMM backend computes the two expert matmuls: a `gmm_fn(a, b, batch_sizes,
trans_b)` callable is injected. Two are provided:

  - reference_gmm: pure PyTorch (a per-group loop of `torch.matmul`), works
    on CPU or CUDA, no extension required. Mathematically equivalent to
    grouped_gemm.ops.gmm by construction, NOT a claim about the real CUDA
    kernels' correctness. This is what --backend reference exercises.
  - real_gmm: a thin wrapper around the real `grouped_gemm.ops.gmm`
    (fanshiqing/grouped_gemm). Requires CUDA and the package installed --
    see INSTALL. The prior indexed-bias implementation passed BF16
    correctness on an LS6 A100 for all eight correctness cases (both
    layouts, including empty experts), with a profiler-observed CUTLASS
    GemmGrouped execution. Changes after that result require rerunning
    --mode validate. Performance is not implied by correctness;
    --mode benchmark, --mode indexing, and --mode profile are opt-in.

No GPU-capability assumption (SM80 or otherwise) is hardcoded anywhere in
GroupedGemmMoE or in the pack/combine logic: `torch.cuda.get_device_capability()`
is read and logged at runtime in log_environment() so the identical code path
runs on LS6 (A100, SM80) and Vista (GH200, SM90) alike. Which underlying
kernel grouped_gemm's C++ extension actually dispatches to (CUTLASS vs a
cuBLAS multi-stream loop) is a property of how THAT PACKAGE was built and
called, not something this script selects -- see BACKEND DISPATCH NOTES.

================================================================================
BACKEND DISPATCH NOTES (source-derived expectations; profile_gmm_dispatch()
below additionally captures REAL kernel-name evidence at runtime when
--backend grouped_gemm is used on CUDA -- printed notes alone are not
runtime verification, and are not treated as such anywhere in this script)

Read directly from fanshiqing/grouped_gemm @ v1.1.4.post8,
grouped_gemm/csrc/grouped_gemm.cu and grouped_gemm/ops.py:

  gmm(a, b, batch_sizes, trans_b) -> GroupedGemm.apply(a, b, batch_sizes, trans_b)
    forward:        backend.gmm(a,    b,    batch_sizes, trans_a=False, trans_b=trans_b)
    input grad:     backend.gmm(grad, b,    batch_sizes, trans_a=False, trans_b=not trans_b)
    weight grad:    backend.gmm(lhs,  rhs,  batch_sizes, trans_a=True,  trans_b=False)
                    (lhs, rhs) = (grad, a) if trans_b else (a, grad)

  C++ dispatch (grouped_gemm.cu), for a call with a given trans_b:
    #if !defined(GROUPED_GEMM_DEVICE_CAPABILITY) || GROUPED_GEMM_DEVICE_CAPABILITY != 80
        CublasGroupedGemm(a, b, c, batch_sizes, trans_b);   // always, any trans_b
    #else
        if (trans_b) CublasGroupedGemm(...);                // trans_b=True -> cuBLAS
        else         CutlassGroupedGemm(a, b, c, batch_sizes); // trans_b=False -> CUTLASS
    #endif

  CUTLASS eligibility therefore depends on trans_b, not on "forward" or
  "dgrad" as fixed labels -- whichever of {forward, input-grad} is called
  with trans_b=False is the one eligible for CUTLASS (and ONLY if the
  extension was built with GROUPED_GEMM_DEVICE_CAPABILITY=80 defined -- see
  INSTALL, this is easy to accidentally suppress). Since input-grad always
  flips trans_b relative to forward (`trans_b=not trans_b`), forward and
  input-grad can never both be CUTLASS-eligible in the same call: choosing
  trans_b for the forward call is a choice of WHICH of the two you make
  eligible.

  This harness therefore runs BOTH configurations (GroupedGemmMoE's
  forward_trans_b flag), not just one:
    forward_trans_b=True  (natural nn.Linear-style weights [E, out, in],
                           x @ W.T): forward itself is ALWAYS cuBLAS;
                           input-grad becomes CUTLASS-eligible.
    forward_trans_b=False (weights pre-transposed to [E, in, out] via a
                           differentiable, .contiguous() stacked view of
                           the SAME per-expert Parameters -- see
                           GroupedGemmMoE._stacked_weight): forward itself
                           becomes CUTLASS-eligible; input-grad becomes
                           cuBLAS.
  Weight-grad (trans_a=True) is unconditional in both cases: it dispatches
  to a wholly separate cuBLAS-only "variable-K" path
  (CublasGroupedGemmVariableK) with no CUTLASS implementation in this
  source, on any capability, regardless of trans_b. Do not describe all
  three GEMMs as CUTLASS in either configuration -- at most one of the
  three ever is, and only on capability 8.0.

================================================================================
INSTALL (LS6 / A100 / SM80)

Package: `nv-grouped-gemm` on PyPI (source: github.com/fanshiqing/grouped_gemm).
NOT the same as PyPI `grouped-gemm` (tgale96's original -- a different,
less complete fork; do not substitute it).

Verified by inspecting PyPI + GitHub directly (both re-checked while writing
this file, not assumed from memory):
  - Latest PyPI release: 1.1.4.post8 (sdist only, no prebuilt wheels on PyPI
    itself -- `pip index`/PyPI JSON API shows one .tar.gz per release).
  - That release DOES include the "Include cutlass into sdist" packaging fix
    (PR #23, commit df0224a155) -- verified via GitHub's compare API:
    df0224a155 is an ancestor of the v1.1.4.post8 tag (2 commits behind it:
    a version bump and a CI change). An earlier, unrelated GitHub issue
    (Megatron-LM #2541) about a missing-CUTLASS-headers build failure on
    this package is NOT applicable to this pinned version.
  - setup.py needs torch importable at build time (it inspects
    torch.cuda.get_device_capability()/TORCH_CUDA_ARCH_LIST to decide which
    architectures to compile for) -- install with --no-build-isolation
    against the project's existing torch 2.11+cu128, do not let it pull a
    fresh torch into an isolated build env.
  - Supports CUDA major versions 11/12/13; our cu128 install is fine.
    Requires C++17 (a reasonably modern host compiler; whatever LS6's
    `module load gcc/...`-provided toolchain already satisfies for building
    the rest of the torch/cu128 stack should be sufficient).
  - CUDA architectures list in setup.py includes both "8.0" (A100) and "9.0"
    (Hopper/GH200). IMPORTANT, corrected from an earlier draft of this
    harness: setup.py's exact logic (quoted, not paraphrased) is

        env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if env_arch_list:
            device_capability = ""          # <- falsy
        else:
            device_capability = torch.cuda.get_device_capability()
            device_capability = f"{device_capability[0]}{device_capability[1]}"
        if device_capability:               # <- skipped whenever TORCH_CUDA_ARCH_LIST is set
            nvcc_flags.extend([..., f"-DGROUPED_GEMM_DEVICE_CAPABILITY={device_capability}"])

    i.e. setting TORCH_CUDA_ARCH_LIST at all -- to "8.0", to anything --
    makes device_capability an empty string and SKIPS defining
    -DGROUPED_GEMM_DEVICE_CAPABILITY entirely, which permanently forces the
    cuBLAS branch for every trans_b (see BACKEND DISPATCH NOTES): the exact
    opposite of what a "validate genuine CUTLASS grouped-GEMM" build needs.
    Do NOT set TORCH_CUDA_ARCH_LIST for this build. Instead leave it unset
    and run the install itself on a node with the target GPU actually
    visible, so torch.cuda.get_device_capability() succeeds and returns the
    real value -- see prerequisite checks below.

Exact commands (run from the repository root, on an LS6 node with an A100
actually visible -- e.g. inside a SLURM GPU allocation, NOT a plain login
node, since auto-detection now requires it):

    # 0. Prerequisite checks -- run these FIRST; if either fails, the build
    #    below will silently produce a cuBLAS-only extension instead of
    #    failing loudly, so do not skip this.
    nvidia-smi -L
    uv run --no-sync python -c \\
      "import torch; print('cuda available:', torch.cuda.is_available()); \\
       print('capability:', torch.cuda.get_device_capability())"
    # expected: cuda available: True / capability: (8, 0)
    nvcc --version   # expect a CUDA 12.x toolkit, matching the project's cu128 torch build

    # 1. Force a from-source build (skip the prebuilt-wheel lookup, which
    #    would target a torch/cu tag combination we don't have anyway),
    #    bypass uv's cache so a previous build/wheel with different flags
    #    can't be silently reused, and capture full build output to a log:
    unset TORCH_CUDA_ARCH_LIST
    GROUPED_GEMM_FORCE_BUILD=TRUE \\
    uv pip install --no-build-isolation --no-cache -v "nv-grouped-gemm==1.1.4.post8" \\
      2>&1 | tee /tmp/grouped_gemm_build.log

    # 2. Verify the capability macro actually landed in a real compiler
    #    invocation (not just present somewhere incidental in the log):
    grep -- "-DGROUPED_GEMM_DEVICE_CAPABILITY=80" /tmp/grouped_gemm_build.log \\
      || echo "MACRO NOT FOUND -- build did not target sm_80, do not proceed"

    # 3. Confirm this didn't change already-resolved project dependencies
    #    (in particular torch, which nv-grouped-gemm's install_requires
    #    lists unpinned):
    uv pip list | grep -E "^torch |^numpy |^absl-py "
    # torch must still read 2.11.0; note whatever numpy/absl-py versions
    # landed (both were previously absent from the project env).

If uv does not stream the underlying setup.py/nvcc compiler lines even with
-v (its own verbosity flags control uv's logging, not necessarily
subprocess passthrough, and this has not been confirmed either way since
this environment has no CUDA to test the actual build against), fall back
to building directly with pip inside the same venv for this one
verification step -- pip's own -v reliably prints the literal compiler
command lines:

    .venv/bin/pip install --no-build-isolation --no-cache-dir -v \\
      "nv-grouped-gemm==1.1.4.post8" 2>&1 | tee /tmp/grouped_gemm_build.log

This installs into the project's existing venv only -- it does NOT touch
pyproject.toml/uv.lock (no `uv add`), per instruction to not modify project
dependencies yet. If/when this is promoted to a real dependency, it should
become a Linux/CUDA-only extra (mirroring the cu128/cu129 split already in
pyproject.toml), since it cannot build at all on macOS.

--------------------------------------------------------------------------------
Vista / GH200 / SM90 -- documented for later, UNVERIFIED, do not treat as tested

  - Same package, same pin, same "leave TORCH_CUDA_ARCH_LIST unset, build on
    a node with the target GPU visible" approach -- on Vista that means
    device_capability auto-detects to "90" and the macro becomes
    -DGROUPED_GEMM_DEVICE_CAPABILITY=90:

        GROUPED_GEMM_FORCE_BUILD=TRUE \\
        uv pip install --no-build-isolation --no-cache -v "nv-grouped-gemm==1.1.4.post8" \\
          2>&1 | tee /tmp/grouped_gemm_build_vista.log
        grep -- "-DGROUPED_GEMM_DEVICE_CAPABILITY=90" /tmp/grouped_gemm_build_vista.log

  - No prebuilt wheels exist for linux_aarch64 on this package's GitHub
    Releases (its wheel-fetch step targets x86_64 build matrices); expect a
    from-source build on Vista rather than a wheel download.
  - Per BACKEND DISPATCH NOTES above, even with the capability macro defined
    as 90, the CUTLASS branch is only taken `#else` when
    GROUPED_GEMM_DEVICE_CAPABILITY == 80 specifically -- capability 90 still
    falls into the unconditional-cuBLAS branch for every trans_b. Defining
    the macro correctly on Vista confirms the BUILD targeted the right GPU;
    it does not make CUTLASS reachable there, per this source.
  - The pinned v1.1.4.post8 tag predates three commits currently on `main`
    (Feb-Apr 2026), one of which (d1f3194079, "Record allocator streams in
    SM90 cuBLAS fallback", merged via PR #26) is specifically an SM90 cuBLAS
    fix NOT included in 1.1.4.post8. If Vista validation surfaces a
    correctness or stream-ordering problem that A100 does not show, the
    fallback plan is to pin instead to a specific commit at or after
    d1f3194079 (e.g. `git+https://github.com/fanshiqing/grouped_gemm@efe8c40eaf`,
    the PR #26 merge commit), not to silently patch around it.
  - Do not consider Vista support validated until this harness has actually
    been run there with --backend grouped_gemm and passed.
================================================================================
"""
import argparse
import copy
import gc
import hashlib
import os
from pathlib import Path
import re
import statistics
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gpt_simple as tgs  # noqa: E402  (imports MoE/Linear unmodified)


# ------------------------------------------------------------------------- #
# GEMM backends
# ------------------------------------------------------------------------- #

def reference_gmm(a, b, batch_sizes, trans_b=False):
    """Pure-PyTorch stand-in for grouped_gemm.ops.gmm: same call signature
    (a: [sum(batch_sizes), K] 2D, b: [num_groups, ...] 3D, batch_sizes: 1D
    sizes along a's dim 0), same per-group semantics. CPU-or-CUDA, any
    dtype. Used to validate the pack/unpack/combine/autograd scaffolding
    independent of any CUDA extension -- NOT a claim about the real
    grouped_gemm CUDA kernels' correctness."""
    assert a.dim() == 2 and b.dim() == 3
    outs = []
    start = 0
    for i in range(batch_sizes.numel()):
        cnt = int(batch_sizes[i])
        a_i = a[start:start + cnt]
        b_i = b[i].t() if trans_b else b[i]
        outs.append(a_i @ b_i)
        start += cnt
    return torch.cat(outs, dim=0) if outs else a.new_zeros(0, b.size(1) if trans_b else b.size(2))


def real_gmm(a, b, batch_sizes, trans_b=False):
    """Thin wrapper around the real fanshiqing/grouped_gemm CUDA extension.
    batch_sizes MUST be an int64 CPU tensor -- the C++ extension asserts
    this (grouped_gemm/csrc/grouped_gemm.cu reads it via
    batch_sizes.data_ptr<int64_t>() on host), so it is never moved to the
    activation's device here. This assert (not a silent .cpu() call here)
    is deliberate: the sync must already have happened by the time this
    function is reached -- see GroupedGemmMoE.forward."""
    import grouped_gemm  # local import: only required when this backend is selected
    assert batch_sizes.device.type == "cpu", "grouped_gemm requires batch_sizes on CPU"
    assert batch_sizes.dtype == torch.int64, "grouped_gemm requires int64 batch_sizes"
    return grouped_gemm.ops.gmm(a, b, batch_sizes, trans_b=trans_b)


BACKENDS = {"reference": reference_gmm, "grouped_gemm": real_gmm}


# ------------------------------------------------------------------------- #
# Model under test
# ------------------------------------------------------------------------- #

class GroupedGemmMoE(nn.Module):
    """route -> pack assignments by expert -> grouped FC1 -> bias -> ReLU^2
    -> grouped FC2 -> bias -> routing weight -> combine.

    Router softmax / top-k / optional renormalize semantics are copied
    verbatim from train_gpt_simple.MoE.forward, so output semantics match
    exactly -- only expert execution differs. Ordinary per-expert
    nn.Parameters (one fc/proj weight+bias per expert, same shapes as
    train_gpt_simple.MLP's Linear submodules) are stored in ParameterLists;
    gmm_fn is called on torch.stack(...) views of these, which is
    differentiable and routes gradients back to each individual Parameter
    (verified below in check_reference_backend_gradients_flow_to_parameters).
    No per-token weight copy is ever made -- only per-token *activation*
    rows are gathered; weights are stacked once, per expert (size E).

    forward_trans_b selects which of the two GEMMs-per-call configurations
    described in BACKEND DISPATCH NOTES is exercised:
      True  (default): natural nn.Linear-style storage [E, out, in] is
             stacked as-is and gmm() is called with trans_b=True (x @ W.T).
      False: the SAME per-expert Parameters are stacked, then
             .transpose(-2, -1).contiguous()'d to [E, in, out], and gmm() is
             called with trans_b=False. Both the transpose and the
             .contiguous() copy are differentiable, so autograd still
             reaches the original [out, in] Parameters -- this is not a
             second, independent set of weights, just a different view fed
             to the GEMM, kept contiguous because CUTLASS-style kernels
             generally require well-defined (row/column-major) strides, not
             an arbitrary transposed view."""

    def __init__(self, dim, num_experts, top_k, normalize_topk=True, gmm_fn=reference_gmm,
                 forward_trans_b=True, profile_stages=False):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_topk = normalize_topk
        self.gmm_fn = gmm_fn
        self.forward_trans_b = forward_trans_b
        self.profile_stages = profile_stages
        hdim = 4 * dim
        self.router = tgs.Linear(dim, num_experts)
        self.fc_weight = nn.ParameterList(nn.Parameter(torch.empty(hdim, dim)) for _ in range(num_experts))
        self.fc_bias = nn.ParameterList(nn.Parameter(torch.empty(hdim)) for _ in range(num_experts))
        self.proj_weight = nn.ParameterList(nn.Parameter(torch.empty(dim, hdim)) for _ in range(num_experts))
        self.proj_bias = nn.ParameterList(nn.Parameter(torch.empty(dim)) for _ in range(num_experts))

    def load_from_moe(self, moe: "tgs.MoE"):
        """Copy weights from an existing train_gpt_simple.MoE instance so
        both implementations start from byte-identical parameters -- this
        is what makes routing decisions identical by construction, rather
        than by separately re-implementing/re-checking the router."""
        self.router.load_state_dict(moe.router.state_dict())
        for i, expert in enumerate(moe.experts):
            self.fc_weight[i].data.copy_(expert.fc.weight.data)
            self.fc_bias[i].data.copy_(expert.fc.bias.data)
            self.proj_weight[i].data.copy_(expert.proj.weight.data)
            self.proj_bias[i].data.copy_(expert.proj.bias.data)

    def _stacked_weight(self, param_list):
        w = torch.stack(list(param_list))  # [E, out, in], differentiable stack of the Parameters
        if self.forward_trans_b:
            return w
        # Differentiable, contiguous transposed view -- gradients still flow
        # back to the original [out, in] Parameters through both ops.
        return w.transpose(-2, -1).contiguous()  # [E, in, out]

    def _scope(self, name):
        return torch.profiler.record_function(name) if self.profile_stages else nullcontext()

    def forward(self, x):
        B, T, D = x.shape
        x = x.reshape(-1, D)
        N = x.shape[0]
        device = x.device

        with self._scope("moe.routing"):
            router_logits = self.router(x)
            routing_weights = F.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_experts = routing_weights.topk(self.top_k, dim=-1)
            if self.normalize_topk:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.type_as(x)

        # ---- pack assignments by expert (backend-independent) ----
        with self._scope("moe.pack"):
            flat_experts = topk_experts.reshape(-1)                                    # [N*k]
            flat_tokens = torch.arange(N, device=device).repeat_interleave(self.top_k)  # [N*k]
            order = torch.argsort(flat_experts, stable=True)
            sorted_experts = flat_experts[order]
            x_sorted = x[flat_tokens[order]]  # gather of ACTIVATIONS; never a weight copy

        # Required CPU int64 token-count metadata for grouped_gemm, and the
        # one explicit device-to-host sync that produces it (see real_gmm's
        # docstring and INSTALL notes above). bincount runs on `device`;
        # only the resulting length-E tensor is synced to host, not the
        # (potentially large) activations.
        with self._scope("moe.count_and_cpu_transfer"):
            batch_sizes = torch.bincount(sorted_experts, minlength=self.num_experts).to(torch.int64).cpu()

        with self._scope("moe.fc1_weight_stack_and_layout"):
            fc_w = self._stacked_weight(self.fc_weight).type_as(x_sorted)
            fc_b = torch.stack(list(self.fc_bias)).type_as(x_sorted)        # [E, 4D]
        with self._scope("moe.fc2_weight_stack_and_layout"):
            proj_w = self._stacked_weight(self.proj_weight).type_as(x_sorted)
            proj_b = torch.stack(list(self.proj_bias)).type_as(x_sorted)    # [E, D]

        with self._scope("moe.fc1_grouped_gemm"):
            h = self.gmm_fn(x_sorted, fc_w, batch_sizes, trans_b=self.forward_trans_b)
        with self._scope("moe.fc1_segmented_bias"):
            h = tgs.add_bias_by_expert_segments(
                h, fc_b, batch_sizes)
        with self._scope("moe.activation"):
            h = h.relu().square()
        with self._scope("moe.fc2_grouped_gemm"):
            out_sorted = self.gmm_fn(
                h.type_as(x_sorted), proj_w, batch_sizes,
                trans_b=self.forward_trans_b)
        with self._scope("moe.fc2_segmented_bias"):
            out_sorted = tgs.add_bias_by_expert_segments(
                out_sorted, proj_b, batch_sizes)

        # ---- unpermute + weighted combine (backend-independent) ----
        with self._scope("moe.combine"):
            out_flat = torch.empty_like(out_sorted)
            out_flat[order] = out_sorted
            out = (out_flat.view(N, self.top_k, D) * topk_weights.unsqueeze(-1)).sum(dim=1)
        return out.view(B, T, D)


# ------------------------------------------------------------------------- #
# Comparison harness
# ------------------------------------------------------------------------- #

def _biased_router_excluding(router: nn.Linear, excluded_experts, bias_value=-1.0e4):
    """Force specific experts to never be selected, regardless of x, by
    driving their router logit far negative -- deterministic 'empty expert'
    construction for requirement 3, independent of router weight randomness."""
    with torch.no_grad():
        for e in excluded_experts:
            router.bias[e] = bias_value


def run_case(name, dim, num_experts, top_k, n_tokens, backend_name, device, dtype,
             normalize_topk=True, excluded_experts=(), atol=2e-2, rtol=2e-2, seed=0,
             forward_trans_b=True):
    gmm_fn = BACKENDS[backend_name]
    torch.manual_seed(seed)

    moe = tgs.MoE(dim, num_experts=num_experts, top_k=top_k, normalize_topk=normalize_topk).to(device)
    for p in moe.parameters():
        p.data = torch.empty_like(p.data, dtype=dtype).normal_(std=0.05)
    if excluded_experts:
        _biased_router_excluding(moe.router, excluded_experts)

    gg = GroupedGemmMoE(dim, num_experts, top_k, normalize_topk=normalize_topk, gmm_fn=gmm_fn,
                         forward_trans_b=forward_trans_b).to(device)
    gg.load_from_moe(moe)

    torch.manual_seed(seed + 1)
    x_base = torch.randn(2, n_tokens, dim, dtype=dtype, device=device)

    x_moe = x_base.clone().requires_grad_(True)
    out_moe = moe(x_moe)
    out_moe.float().sum().backward()

    x_gg = x_base.clone().requires_grad_(True)
    out_gg = gg(x_gg)
    out_gg.float().sum().backward()

    ok = True

    def check(label, a, b, allow_none=False):
        nonlocal ok
        if a is None and b is None:
            print(f"  [{name}] {label}: both None (skipped, allow_none={allow_none})", flush=True)
            return
        if a is None or b is None:
            print(f"  [{name}] {label}: MISMATCH -- one side is None (moe={a is None}, gg={b is None})",
                  flush=True)
            ok = False
            return
        close = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
        max_diff = (a.float() - b.float()).abs().max().item()
        status = "OK" if close else "MISMATCH"
        print(f"  [{name}] {label}: {status} (max_abs_diff={max_diff:.6f})", flush=True)
        ok = ok and close

    check("output", out_moe, out_gg)
    check("grad_x", x_moe.grad, x_gg.grad)
    check("grad_router.weight", moe.router.weight.grad, gg.router.weight.grad)
    check("grad_router.bias", moe.router.bias.grad, gg.router.bias.grad)

    for i in range(num_experts):
        expert = moe.experts[i]
        is_excluded = i in excluded_experts
        # Requirement 4: zero vs. absent gradients for unused experts, made explicit.
        if is_excluded:
            none_as_expected = expert.fc.weight.grad is None
            zero_not_none = (gg.fc_weight[i].grad is not None
                              and torch.count_nonzero(gg.fc_weight[i].grad) == 0)
            print(f"  [{name}] expert{i} (excluded): loop-MoE grad is None: {none_as_expected} "
                  f"(expected True -- known loop-dispatch limitation); "
                  f"grouped-gemm grad is present AND all-zero: {zero_not_none} "
                  f"(expected True -- this is the property being validated)", flush=True)
            ok = ok and none_as_expected and zero_not_none
            continue
        check(f"expert{i}.fc.weight", expert.fc.weight.grad, gg.fc_weight[i].grad)
        check(f"expert{i}.fc.bias", expert.fc.bias.grad, gg.fc_bias[i].grad)
        check(f"expert{i}.proj.weight", expert.proj.weight.grad, gg.proj_weight[i].grad)
        check(f"expert{i}.proj.bias", expert.proj.bias.grad, gg.proj_bias[i].grad)

    print(f"[{name}] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


CASES = [
    dict(name="E1_k1", dim=16, num_experts=1, top_k=1, n_tokens=20),
    dict(name="E8_k2_random_uneven", dim=16, num_experts=8, top_k=2, n_tokens=37, seed=3),
    dict(name="E8_k2_forced_empty_experts", dim=16, num_experts=8, top_k=2, n_tokens=37,
         excluded_experts=(2, 5), seed=4),
    dict(name="E8_k2_normalize_false", dim=16, num_experts=8, top_k=2, n_tokens=37,
         normalize_topk=False, seed=5),
]


# The first case exactly matches one training MoE layer's flattened token and
# matrix dimensions. The E=8 case intentionally uses 1/16 as many tokens so
# all three implementations can be investigated quickly before committing to
# a full training-step profile.
BENCHMARK_CASES = {
    "train_e1_k1": dict(name="train_e1_k1", batch=64, seq_len=1024, dim=768,
                         num_experts=1, top_k=1, seed=101),
    "small_e8_k2": dict(name="small_e8_k2", batch=4, seq_len=1024, dim=768,
                         num_experts=8, top_k=2, seed=202),
}

BENCHMARK_CONFIGS = (
    ("loop", None),
    ("grouped_trans_b_false", False),
    ("grouped_trans_b_true", True),
)


def _make_benchmark_inputs(case, activation_dtype):
    """Create one CPU source of truth reused by every sequential config.

    Production keeps MoE parameters in FP32 (`GPT(...).cuda()`) and receives
    BF16 residual-stream activations from the BF16 embedding. Linear.forward
    and the grouped path cast weights/biases to the activation dtype on every
    call. Preserve that mixed-dtype behavior here instead of moving the model
    itself to BF16.
    """
    generator = torch.Generator(device="cpu").manual_seed(case["seed"])
    canonical = tgs.MoE(case["dim"], num_experts=case["num_experts"],
                        top_k=case["top_k"], normalize_topk=True).to(dtype=torch.float32)
    with torch.no_grad():
        for p in canonical.parameters():
            p.copy_(torch.randn(p.shape, generator=generator, dtype=torch.float32) * 0.02)
    x_cpu = torch.randn(case["batch"], case["seq_len"], case["dim"],
                        generator=generator, dtype=activation_dtype)
    return canonical, x_cpu


def _canonical_parameter_pairs(canonical, candidate, config_name):
    if config_name == "loop":
        canonical_params = list(canonical.named_parameters())
        candidate_params = list(candidate.named_parameters())
        assert [n for n, _ in canonical_params] == [n for n, _ in candidate_params]
        return [(name, expected, actual) for (name, expected), (_, actual)
                in zip(canonical_params, candidate_params, strict=True)]

    pairs = []
    for name, expected in canonical.router.named_parameters():
        pairs.append((f"router.{name}", expected, dict(candidate.router.named_parameters())[name]))
    for i, expert in enumerate(canonical.experts):
        pairs.extend((
            (f"experts.{i}.fc.weight", expert.fc.weight, candidate.fc_weight[i]),
            (f"experts.{i}.fc.bias", expert.fc.bias, candidate.fc_bias[i]),
            (f"experts.{i}.proj.weight", expert.proj.weight, candidate.proj_weight[i]),
            (f"experts.{i}.proj.bias", expert.proj.bias, candidate.proj_bias[i]),
        ))
    return pairs


def _assert_identical_benchmark_parameters(canonical, candidate, config_name):
    """Fail before timing if a config did not receive byte-identical weights."""
    for name, expected, actual in _canonical_parameter_pairs(canonical, candidate, config_name):
        assert expected.shape == actual.shape, name
        assert expected.dtype == actual.dtype == torch.float32, name
        actual_cpu = actual.detach().cpu()
        assert torch.equal(expected.detach(), actual_cpu), name


def _make_benchmark_model(canonical, case, config_name, forward_trans_b, device,
                          profile_stages=False):
    if config_name == "loop":
        model = copy.deepcopy(canonical).to(device=device)
    else:
        model = GroupedGemmMoE(
            case["dim"], case["num_experts"], case["top_k"], normalize_topk=True,
            gmm_fn=real_gmm, forward_trans_b=forward_trans_b,
            profile_stages=profile_stages,
        ).to(device=device, dtype=torch.float32)
        model.load_from_moe(canonical)
    _assert_identical_benchmark_parameters(canonical, model, config_name)
    return model


@torch.no_grad()
def _routing_snapshot(router, x, top_k):
    flat_x = x.reshape(-1, x.shape[-1])
    probabilities = F.softmax(router(flat_x).float(), dim=-1)
    weights, experts = probabilities.topk(top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).type_as(flat_x)
    return experts.cpu(), weights.cpu()


def _assert_identical_routing(snapshot, expected, config_name):
    if expected is None:
        return snapshot
    assert torch.equal(snapshot[0], expected[0]), f"{config_name}: top-k expert indices differ"
    assert torch.equal(snapshot[1], expected[1]), f"{config_name}: top-k routing weights differ"
    return expected


def _cuda_measure(fn, prepare, warmup, iterations):
    """Wall-clock CUDA work with device-wide boundaries.

    torch.cuda.synchronize() waits for all streams on the device, including
    the auxiliary streams created inside nv-grouped-gemm. CUDA events on the
    current stream alone would undercount that work.
    """
    for _ in range(warmup):
        prepare()
        fn()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    samples_ms = []
    for _ in range(iterations):
        prepare()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1e3)
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return samples_ms, peak_gib


def _print_latency(label, samples_ms, peak_gib):
    print(f"  {label}: median={statistics.median(samples_ms):.3f} ms "
          f"mean={statistics.mean(samples_ms):.3f} ms "
          f"min={min(samples_ms):.3f} ms max={max(samples_ms):.3f} ms "
          f"peak_allocated={peak_gib:.3f} GiB n={len(samples_ms)}", flush=True)


def _release_cuda_tensors():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def benchmark_one_configuration(canonical, x_cpu, case, config_name, forward_trans_b,
                                dtype, warmup, iterations, expected_routing):
    print(f"[{case['name']}:{config_name}] allocating", flush=True)
    model = _make_benchmark_model(canonical, case, config_name, forward_trans_b,
                                  device="cuda")
    x = x_cpu.to(device="cuda").requires_grad_(True)
    assert x.dtype == dtype
    assert torch.equal(x.detach().cpu(), x_cpu), f"{config_name}: input copy differs"
    snapshot = _routing_snapshot(model.router, x, case["top_k"])
    expected_routing = _assert_identical_routing(snapshot, expected_routing, config_name)
    counts = torch.bincount(snapshot[0].reshape(-1), minlength=case["num_experts"])
    print(f"  exact weights/input/routing verified; assignment_counts={counts.tolist()}", flush=True)

    last_out = None

    def forward_only():
        nonlocal last_out
        last_out = model(x)

    def clear_forward():
        nonlocal last_out
        last_out = None

    forward_ms, forward_peak = _cuda_measure(forward_only, clear_forward, warmup, iterations)
    _print_latency("forward", forward_ms, forward_peak)
    last_out = None
    _release_cuda_tensors()

    grad_out = torch.randn_like(x)

    def forward_backward():
        nonlocal last_out
        last_out = model(x)
        last_out.backward(grad_out)

    def clear_gradients():
        nonlocal last_out
        last_out = None
        x.grad = None
        for p in model.parameters():
            p.grad = None

    fwd_bwd_ms, fwd_bwd_peak = _cuda_measure(
        forward_backward, clear_gradients, warmup, iterations)
    _print_latency("forward+backward", fwd_bwd_ms, fwd_bwd_peak)

    del last_out, grad_out, x, model, snapshot
    _release_cuda_tensors()
    return expected_routing, dict(
        forward_median_ms=statistics.median(forward_ms),
        forward_backward_median_ms=statistics.median(fwd_bwd_ms),
        forward_peak_gib=forward_peak,
        forward_backward_peak_gib=fwd_bwd_peak,
    )


def run_benchmarks(args, dtype):
    if args.backend != "grouped_gemm" or args.device != "cuda":
        raise SystemExit("--mode benchmark requires --backend grouped_gemm --device cuda")
    selected = BENCHMARK_CASES if args.benchmark_case == "all" \
        else {args.benchmark_case: BENCHMARK_CASES[args.benchmark_case]}
    print("BENCHMARK MODE: eager isolated layers; profiler is disabled in this process.", flush=True)
    print("Latency is wall time between device-wide synchronizations and includes extension "
          "auxiliary-stream work. Configurations run sequentially.", flush=True)
    for case in selected.values():
        n_tokens = case["batch"] * case["seq_len"]
        print("=" * 100, flush=True)
        print(f"case={case['name']} N={n_tokens} D={case['dim']} hidden={4 * case['dim']} "
              f"E={case['num_experts']} k={case['top_k']} "
              f"activation_dtype={dtype} parameter_dtype={torch.float32}", flush=True)
        canonical, x_cpu = _make_benchmark_inputs(case, dtype)
        expected_routing = None
        results = {}
        for config_name, forward_trans_b in BENCHMARK_CONFIGS:
            expected_routing, results[config_name] = benchmark_one_configuration(
                canonical, x_cpu, case, config_name, forward_trans_b, dtype,
                args.warmup, args.iterations, expected_routing)
        loop_result = results["loop"]
        print(f"[{case['name']}] median latency comparison (ratio vs loop):", flush=True)
        for config_name, _ in BENCHMARK_CONFIGS:
            result = results[config_name]
            print(f"  {config_name}: forward={result['forward_median_ms']:.3f} ms "
                  f"({result['forward_median_ms'] / loop_result['forward_median_ms']:.3f}x), "
                  f"forward+backward={result['forward_backward_median_ms']:.3f} ms "
                  f"({result['forward_backward_median_ms'] / loop_result['forward_backward_median_ms']:.3f}x)",
                  flush=True)
        del results, expected_routing, x_cpu, canonical
        _release_cuda_tensors()


def _measure_autograd_implementation(make_output, clear_gradients, grad_out, warmup, iterations):
    """Measure forward, isolated backward, and combined forward+backward."""
    output = None

    def clear_forward():
        nonlocal output
        output = None
        clear_gradients()

    def forward():
        nonlocal output
        output = make_output()

    forward_ms, _ = _cuda_measure(forward, clear_forward, warmup, iterations)
    output = None
    _release_cuda_tensors()

    def prepare_backward():
        nonlocal output
        output = None
        clear_gradients()
        output = make_output()

    def backward():
        output.backward(grad_out)

    backward_ms, _ = _cuda_measure(backward, prepare_backward, warmup, iterations)
    output = None
    _release_cuda_tensors()

    def forward_backward():
        nonlocal output
        output = make_output()
        output.backward(grad_out)

    fwd_bwd_ms, peak_gib = _cuda_measure(
        forward_backward, clear_forward, warmup, iterations)
    output = None
    _release_cuda_tensors()
    return dict(
        forward_ms=statistics.median(forward_ms),
        backward_ms=statistics.median(backward_ms),
        forward_backward_ms=statistics.median(fwd_bwd_ms),
        peak_gib=peak_gib,
    )


def _print_indexing_comparison(label, results):
    baseline = results[next(iter(results))]
    print(f"[{label}] medians and ratios vs {next(iter(results))}:", flush=True)
    for name, result in results.items():
        print(f"  {name}: forward={result['forward_ms']:.3f} ms "
              f"({result['forward_ms'] / baseline['forward_ms']:.3f}x), "
              f"backward={result['backward_ms']:.3f} ms "
              f"({result['backward_ms'] / baseline['backward_ms']:.3f}x), "
              f"forward+backward={result['forward_backward_ms']:.3f} ms "
              f"({result['forward_backward_ms'] / baseline['forward_backward_ms']:.3f}x), "
              f"peak={result['peak_gib']:.3f} GiB", flush=True)


def _make_indexing_metadata(case):
    n_tokens = case["batch"] * case["seq_len"]
    assignments = n_tokens * case["top_k"]
    if case["num_experts"] == 1:
        counts_list = [assignments]
    else:
        # Deliberately uneven, sums to 8192 for the provided E=8/k=2 case.
        counts_list = [2048, 1536, 1280, 1024, 768, 640, 512, 384]
        assert len(counts_list) == case["num_experts"] and sum(counts_list) == assignments
    counts_cpu = torch.tensor(counts_list, dtype=torch.int64)
    sorted_experts_cpu = torch.repeat_interleave(
        torch.arange(case["num_experts"], dtype=torch.int64), counts_cpu)
    generator = torch.Generator(device="cpu").manual_seed(case["seed"] + 1000)
    shuffled_experts = sorted_experts_cpu[torch.randperm(assignments, generator=generator)]
    order_cpu = torch.argsort(shuffled_experts, stable=True)
    flat_tokens_cpu = torch.arange(n_tokens).repeat_interleave(case["top_k"])
    pack_index_cpu = flat_tokens_cpu[order_cpu]
    inverse_order_cpu = torch.empty_like(order_cpu)
    inverse_order_cpu[order_cpu] = torch.arange(assignments)
    return dict(
        counts_cpu=counts_cpu,
        counts_cuda=counts_cpu.cuda(),
        sorted_experts=sorted_experts_cpu.cuda(),
        order=order_cpu.cuda(),
        inverse_order=inverse_order_cpu.cuda(),
        pack_index=pack_index_cpu.cuda(),
        n_tokens=n_tokens,
        assignments=assignments,
    )


def benchmark_bias_add_candidates(case, metadata, width, dtype, warmup, iterations):
    label = f"{case['name']}:bias_width_{width}"
    results = {}
    for candidate_name in ("advanced_index", "index_select", "repeat_interleave", "segment_add_cat"):
        torch.manual_seed(case["seed"] + width)
        values = torch.randn(metadata["assignments"], width, device="cuda", dtype=dtype,
                             requires_grad=True)
        bias_params = [torch.randn(width, device="cuda", dtype=torch.float32, requires_grad=True)
                       for _ in range(case["num_experts"])]
        grad_out = torch.randn_like(values)

        def make_output():
            bias = torch.stack(bias_params).type_as(values)
            if candidate_name == "advanced_index":
                return values + bias[metadata["sorted_experts"]]
            if candidate_name == "index_select":
                return values + bias.index_select(0, metadata["sorted_experts"])
            if candidate_name == "repeat_interleave":
                expanded = torch.repeat_interleave(
                    bias, metadata["counts_cuda"], dim=0,
                    output_size=metadata["assignments"])
                return values + expanded
            return tgs.add_bias_by_expert_segments(values, bias, metadata["counts_cpu"])

        def clear_gradients():
            values.grad = None
            for bias_param in bias_params:
                bias_param.grad = None

        results[candidate_name] = _measure_autograd_implementation(
            make_output, clear_gradients, grad_out, warmup, iterations)
        del grad_out, bias_params, values
        _release_cuda_tensors()
    _print_indexing_comparison(label, results)


def benchmark_pack_candidates(case, metadata, dtype, warmup, iterations):
    results = {}
    for candidate_name in ("advanced_index", "index_select"):
        torch.manual_seed(case["seed"] + 2000)
        source = torch.randn(metadata["n_tokens"], case["dim"], device="cuda", dtype=dtype,
                             requires_grad=True)
        grad_out = torch.randn(metadata["assignments"], case["dim"], device="cuda", dtype=dtype)

        def make_output():
            if candidate_name == "advanced_index":
                return source[metadata["pack_index"]]
            return source.index_select(0, metadata["pack_index"])

        def clear_gradients():
            source.grad = None

        results[candidate_name] = _measure_autograd_implementation(
            make_output, clear_gradients, grad_out, warmup, iterations)
        del grad_out, source
        _release_cuda_tensors()
    _print_indexing_comparison(f"{case['name']}:activation_pack", results)


def benchmark_unpermute_candidates(case, metadata, dtype, warmup, iterations):
    results = {}
    for candidate_name in ("index_put", "inverse_index_select"):
        torch.manual_seed(case["seed"] + 3000)
        source = torch.randn(metadata["assignments"], case["dim"], device="cuda", dtype=dtype,
                             requires_grad=True)
        grad_out = torch.randn_like(source)

        def make_output():
            if candidate_name == "index_put":
                output = torch.empty_like(source)
                output[metadata["order"]] = source
                return output
            return source.index_select(0, metadata["inverse_order"])

        def clear_gradients():
            source.grad = None

        results[candidate_name] = _measure_autograd_implementation(
            make_output, clear_gradients, grad_out, warmup, iterations)
        del grad_out, source
        _release_cuda_tensors()
    _print_indexing_comparison(f"{case['name']}:unpermute", results)


def run_indexing_benchmarks(args, dtype):
    if args.device != "cuda":
        raise SystemExit("--mode indexing requires --device cuda")
    selected = BENCHMARK_CASES if args.benchmark_case == "all" \
        else {args.benchmark_case: BENCHMARK_CASES[args.benchmark_case]}
    print("INDEXING MODE: isolated eager PyTorch operators; no grouped GEMM calls and no profiler.",
          flush=True)
    print("Each implementation is measured sequentially with device-wide synchronization.", flush=True)
    for case in selected.values():
        metadata = _make_indexing_metadata(case)
        print("=" * 100, flush=True)
        print(f"case={case['name']} N={metadata['n_tokens']} D={case['dim']} "
              f"assignments={metadata['assignments']} E={case['num_experts']} "
              f"k={case['top_k']} counts={metadata['counts_cpu'].tolist()}", flush=True)
        benchmark_pack_candidates(case, metadata, dtype, args.warmup, args.iterations)
        benchmark_bias_add_candidates(
            case, metadata, 4 * case["dim"], dtype, args.warmup, args.iterations)
        benchmark_bias_add_candidates(
            case, metadata, case["dim"], dtype, args.warmup, args.iterations)
        benchmark_unpermute_candidates(case, metadata, dtype, args.warmup, args.iterations)
        del metadata
        _release_cuda_tensors()


def log_environment(args):
    print("=" * 100, flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"device={torch.cuda.get_device_name(0)} capability=sm_{cap[0]}{cap[1]}", flush=True)
    else:
        print("CUDA not available on this machine -- only --backend reference can run here.", flush=True)
    print(f"mode={args.mode} backend={args.backend} device={args.device} dtype={args.dtype}", flush=True)
    if args.backend == "grouped_gemm":
        import grouped_gemm
        print(f"grouped_gemm module file: {getattr(grouped_gemm, '__file__', 'unknown')}", flush=True)
        print(f"grouped_gemm version attr: {getattr(grouped_gemm, '__version__', 'unknown (not exported)')}",
              flush=True)
        print("This run exercises the REAL CUDA extension; the selected mode controls which "
              "forward_trans_b configs run (see BACKEND DISPATCH NOTES). Weight-grad is always "
              "cuBLAS regardless; forward and input-grad each become CUTLASS-eligible in "
              "exactly one of the two configs, and only if this extension was built with "
              "GROUPED_GEMM_DEVICE_CAPABILITY=80 defined (see INSTALL). These are source-"
              "derived EXPECTATIONS, not runtime facts -- profile_gmm_dispatch() below is "
              "the actual runtime evidence.", flush=True)
    else:
        print("This run uses the pure-PyTorch reference backend: it validates pack/unpack/"
              "combine/autograd scaffolding, NOT the real grouped_gemm CUDA kernels. "
              "Use --backend grouped_gemm on CUDA to exercise the installed extension.",
              flush=True)
    print("=" * 100, flush=True)


def _pick_timing_attr(avgs):
    """Return the name of the self-time-in-device attribute actually
    supported by the installed torch version's FunctionEventAvg, preferring
    the current (non-deprecated) name. torch 2.11 deprecated
    self_cuda_time_total in favor of the device-agnostic
    self_device_time_total (confirmed directly against this project's
    installed torch: `cuda_time` raises `FutureWarning: ... please use
    device_time instead`). Returns None if NEITHER attribute exists, so
    callers report timing as unavailable instead of silently treating
    missing data as zero."""
    if not avgs:
        return None
    for name in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(avgs[0], name):
            return name
    return None


def _report_kernel_evidence(avgs, label):
    names = [e.key for e in avgs]
    cutlass_hits = [n for n in names if "cutlass" in n.lower()]
    cublas_hits = [n for n in names if "cublas" in n.lower() or "gemm" in n.lower()]
    print(f"[profile:{label}] {len(names)} distinct op/kernel names recorded", flush=True)
    print(f"[profile:{label}] names containing 'cutlass': {cutlass_hits or '(none)'}", flush=True)
    print(f"[profile:{label}] names containing 'cublas'/'gemm': {cublas_hits or '(none)'}", flush=True)

    timing_attr = _pick_timing_attr(avgs)
    if timing_attr is None:
        print(f"[profile:{label}] TIMING DATA UNAVAILABLE -- this torch version's "
              f"FunctionEventAvg exposes neither self_device_time_total nor "
              f"self_cuda_time_total. Reported as unavailable, not as zero.", flush=True)
    else:
        note = " (deprecated name; self_device_time_total not found on this torch version)" \
            if timing_attr == "self_cuda_time_total" else ""
        print(f"[profile:{label}] timing field used: {timing_attr}{note}", flush=True)
        top = sorted(avgs, key=lambda e: getattr(e, timing_attr), reverse=True)[:10]
        print(f"[profile:{label}] top-10 ops by {timing_attr}:", flush=True)
        for e in top:
            print(f"    {e.key}  {timing_attr}={getattr(e, timing_attr)}  count={e.count}", flush=True)
    return dict(cutlass_observed=bool(cutlass_hits), cublas_or_gemm_observed=bool(cublas_hits),
                timing_available=timing_attr is not None)


def profile_gmm_dispatch(gmm_fn, dim, num_experts, n_tokens_per_expert, forward_trans_b, device, dtype):
    """Run gmm_fn's forward and backward SEPARATELY, each under its own
    torch.profiler session, and report which CUDA kernels actually executed
    in each phase -- REAL runtime evidence for which branch (CUTLASS vs
    cuBLAS) was taken, rather than the source-derived expectations in
    BACKEND DISPATCH NOTES. Forward and backward are profiled separately
    (not one combined trace) because backward launches TWO different
    kernels -- the input-gradient GEMM and the separate weight-gradient
    "variable-K" GEMM -- and conflating them with forward's kernel into one
    trace would make it impossible to attribute evidence to the right call.
    Only meaningful for --backend grouped_gemm on CUDA; main() does not call
    this for --backend reference (plain torch.matmul, no dispatch ambiguity
    to confirm)."""
    from torch.profiler import ProfilerActivity, profile

    hdim = 4 * dim
    batch_sizes = torch.full((num_experts,), n_tokens_per_expert, dtype=torch.int64)  # CPU, per real_gmm's contract
    total = int(batch_sizes.sum())
    a = torch.randn(total, dim, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(num_experts, hdim, dim, dtype=dtype, device=device, requires_grad=True)
    b_arg = w if forward_trans_b else w.detach().transpose(-2, -1).contiguous().requires_grad_(True)

    # Warm up OUTSIDE any profiler context (first-call cuBLAS handle /
    # workspace allocation, first-call CUTLASS kernel selection), so the
    # profiled runs below reflect steady-state dispatch, not one-time setup.
    warm_out = gmm_fn(a, b_arg, batch_sizes, trans_b=forward_trans_b)
    warm_out.sum().backward()
    a.grad = None
    b_arg.grad = None
    torch.cuda.synchronize()

    label = f"forward_trans_b={forward_trans_b}"

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as fwd_prof:
        out = gmm_fn(a, b_arg, batch_sizes, trans_b=forward_trans_b)
        torch.cuda.synchronize()
    fwd_evidence = _report_kernel_evidence(fwd_prof.key_averages(), f"{label} FORWARD")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as bwd_prof:
        out.sum().backward()
        torch.cuda.synchronize()
    bwd_evidence = _report_kernel_evidence(bwd_prof.key_averages(), f"{label} BACKWARD")

    return dict(forward=fwd_evidence, backward=bwd_evidence)


def report_extension_binary_and_build_flags(build_log):
    """Report binary identity and flags from the captured *actual* build.

    Python extension modules do not embed a standardized copy of their nvcc
    command line, so setup.py defaults are not evidence of how the installed
    .so was built. A verbose build log is the source of truth here. If none is
    supplied (or it contains no nvcc command), say so rather than inferring.
    """
    import grouped_gemm

    package_dir = Path(grouped_gemm.__file__).resolve().parent
    shared_objects = sorted(package_dir.rglob("*.so"))
    print("=" * 100, flush=True)
    print("Installed grouped_gemm binary identity:", flush=True)
    if not shared_objects:
        print(f"  NO .so FOUND under {package_dir}", flush=True)
    for so_path in shared_objects:
        digest = hashlib.sha256()
        with so_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        print(f"  {so_path} sha256={digest.hexdigest()}", flush=True)

    if build_log is None:
        print("Build optimization flags: UNVERIFIED (--build-log was not supplied).", flush=True)
        return
    log_path = Path(build_log)
    if not log_path.is_file():
        print(f"Build optimization flags: UNVERIFIED ({log_path} does not exist).", flush=True)
        return
    lines = log_path.read_text(errors="replace").splitlines()
    nvcc_lines = [line.strip() for line in lines if re.search(r"(^|\s)(nvcc|[^ ]*/nvcc)(\s|$)", line)]
    if not nvcc_lines:
        print(f"Build optimization flags: UNVERIFIED (no nvcc invocation in {log_path}).", flush=True)
        return

    joined = "\n".join(nvcc_lines)
    flag_patterns = (
        r"(?<!\S)-O(?:0|1|2|3|s|fast)(?!\S)",
        r"(?<!\S)--use_fast_math(?!\S)",
        r"(?<!\S)-G(?!\S)",
        r"(?<!\S)-g(?!\S)",
        r"(?<!\S)-lineinfo(?!\S)",
        r"(?<!\S)-DNDEBUG(?!\S)",
        r"-DGROUPED_GEMM_DEVICE_CAPABILITY=\d+",
        r"(?:-gencode|--generate-code)(?:=|\s+)[^\s]+",
    )
    flags = sorted({match.group(0) for pattern in flag_patterns
                    for match in re.finditer(pattern, joined)})
    debug_device_code = bool(re.search(r"(?<!\S)-G(?!\S)", joined))
    print(f"Build-log evidence: {log_path.resolve()} ({len(nvcc_lines)} nvcc invocation(s))", flush=True)
    print(f"  observed optimization/codegen flags: {flags or '(none matched)'}", flush=True)
    print(f"  debug device code (-G) observed: {debug_device_code}", flush=True)
    print(f"  fast math observed: {'--use_fast_math' in joined}", flush=True)
    print("  nvcc command(s), verbatim:", flush=True)
    for line in nvcc_lines:
        print(f"    {line}", flush=True)


@torch.no_grad()
def report_production_dtype_conversions(moe, x):
    """Execute the production stacking/layout/cast expressions and report copies."""
    x_flat = x.reshape(-1, x.shape[-1])
    print("Production dtype/layout conversion probe (same expressions as train_gpt_simple.MoE):",
          flush=True)
    print(f"  activation={x_flat.dtype}; router.weight={moe.router.weight.dtype}", flush=True)
    for label, params in (
        ("fc.weight", [e.fc.weight for e in moe.experts]),
        ("proj.weight", [e.proj.weight for e in moe.experts]),
    ):
        stacked = torch.stack(params)
        transposed_contiguous = stacked.transpose(-2, -1).contiguous()
        cast = transposed_contiguous.type_as(x_flat)
        print(f"  {label}: stored={params[0].dtype} stacked={stacked.dtype} "
              f"stacked_shape={tuple(stacked.shape)} trans_b=False_shape={tuple(cast.shape)} "
              f"transpose_contiguous_copy={transposed_contiguous.data_ptr() != stacked.data_ptr()} "
              f"type_as_copy={cast.data_ptr() != transposed_contiguous.data_ptr()}", flush=True)
        del cast, transposed_contiguous, stacked
    for label, params in (
        ("fc.bias", [e.fc.bias for e in moe.experts]),
        ("proj.bias", [e.proj.bias for e in moe.experts]),
    ):
        stacked = torch.stack(params)
        cast = stacked.type_as(x_flat)
        print(f"  {label}: stored={params[0].dtype} stacked={stacked.dtype} "
              f"type_as_copy={cast.data_ptr() != stacked.data_ptr()}", flush=True)
        del cast, stacked
    torch.cuda.synchronize()


def _profiled_loop_forward(moe, x):
    """Production loop forward split only into profiler annotation ranges."""
    B, T, D = x.shape
    flat_x = x.reshape(-1, D)
    with torch.profiler.record_function("moe.routing"):
        router_logits = moe.router(flat_x)
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_experts = routing_weights.topk(moe.top_k, dim=-1)
        if moe.normalize_topk:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.type_as(flat_x)
    with torch.profiler.record_function("moe.loop_expert_dispatch_and_combine"):
        out = flat_x.new_zeros(flat_x.shape)
        for expert_idx, expert in enumerate(moe.experts):
            token_idx, slot_idx = torch.where(topk_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            out.index_add_(0, token_idx,
                           expert(flat_x[token_idx]) * topk_weights[token_idx, slot_idx, None])
    return out.view(B, T, D)


def _profile_total_timing_attr(avgs):
    if not avgs:
        return None
    for name in ("device_time_total", "cuda_time_total"):
        if hasattr(avgs[0], name):
            return name
    return None


def _report_full_layer_profile(prof, label):
    avgs = prof.key_averages()
    timing_attr = _profile_total_timing_attr(avgs)
    print(f"[profile:{label}] annotated stage totals (profiler overhead present; not benchmark data):",
          flush=True)
    stages = [event for event in avgs
              if event.key.startswith("moe.") and event.key != "moe.full_forward_backward"]
    for event in stages:
        device_us = getattr(event, timing_attr) if timing_attr is not None else None
        print(f"  {event.key}: cpu_total_us={event.cpu_time_total:.3f} "
              f"device_total_us={device_us if device_us is not None else 'unavailable'} "
              f"count={event.count}", flush=True)
    if stages:
        host_dominant = max(stages, key=lambda event: event.cpu_time_total)
        print(f"  largest annotated forward host range: {host_dominant.key} "
              f"({host_dominant.cpu_time_total:.3f} us)", flush=True)
        if timing_attr is not None:
            device_dominant = max(stages, key=lambda event: getattr(event, timing_attr))
            print(f"  largest annotated forward device range: {device_dominant.key} "
                  f"({getattr(device_dominant, timing_attr):.3f} us)", flush=True)
    print(f"[profile:{label}] top-10 ops/ranges by self CPU time:", flush=True)
    for event in sorted(avgs, key=lambda item: item.self_cpu_time_total, reverse=True)[:10]:
        print(f"  {event.key}: self_cpu_time_total={event.self_cpu_time_total:.3f} "
              f"count={event.count}", flush=True)
    _report_kernel_evidence(avgs, label)


def profile_full_layer(model, x, grad_out, case_name, config_name, trace_dir, warmup):
    from torch.profiler import ProfilerActivity, profile

    def call_model():
        return _profiled_loop_forward(model, x) if config_name == "loop" else model(x)

    for _ in range(warmup):
        out = call_model()
        out.backward(grad_out)
        x.grad = None
        for p in model.parameters():
            p.grad = None
        torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        with torch.profiler.record_function("moe.full_forward_backward"):
            out = call_model()
            out.backward(grad_out)
        torch.cuda.synchronize()
    trace_path = trace_dir / f"{case_name}_{config_name}_full_layer.json"
    prof.export_chrome_trace(str(trace_path))
    _report_full_layer_profile(prof, f"{config_name} FULL_LAYER")
    print(f"[profile:{config_name}] chrome trace: {trace_path}", flush=True)


def _profile_raw_gmm_phase(case_name, phase, gemm_name, in_dim, out_dim, batch_sizes,
                           forward_trans_b, dtype, trace_dir):
    """Profile exactly one native forward, input-grad, or weight-grad call.

    Do not obtain dgrad/wgrad by calling the custom autograd Function's
    backward: nv-grouped-gemm computes both there. Invoke the same backend.gmm
    calls used by ops.py directly so each trace contains only the named phase.
    """
    from torch.profiler import ProfilerActivity, profile
    from grouped_gemm import backend

    total = int(batch_sizes.sum())
    a = torch.randn(total, in_dim, device="cuda", dtype=dtype)
    natural_w = torch.randn(len(batch_sizes), out_dim, in_dim, device="cuda", dtype=dtype)
    b = natural_w if forward_trans_b else natural_w.transpose(-2, -1).contiguous()
    grad_out = torch.randn(total, out_dim, device="cuda", dtype=dtype) if phase != "forward" else None

    def phase_call():
        if phase == "forward":
            return backend.gmm(a, b, batch_sizes, False, forward_trans_b)
        if phase == "input_grad":
            return backend.gmm(grad_out, b, batch_sizes, False, not forward_trans_b)
        # Exact GroupedGemm.backward weight-gradient argument order from ops.py.
        lhs, rhs = (grad_out, a) if forward_trans_b else (a, grad_out)
        return backend.gmm(lhs, rhs, batch_sizes, True, False)

    warm_out = phase_call()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        out = phase_call()
        torch.cuda.synchronize()

    layout = f"trans_b_{str(forward_trans_b).lower()}"
    label = f"{case_name}_{layout}_{gemm_name}_{phase}"
    trace_path = trace_dir / f"{label}.json"
    prof.export_chrome_trace(str(trace_path))
    avgs = prof.key_averages()
    _report_kernel_evidence(avgs, label)
    print(f"[profile:{label}] chrome trace: {trace_path}", flush=True)
    timing_attr = _pick_timing_attr(avgs)
    if timing_attr is None:
        summary = dict(label=label, device_work_us=None, top_device_op=None, top_device_us=None)
    else:
        top = max(avgs, key=lambda event: getattr(event, timing_attr))
        summary = dict(
            label=label,
            # Sum of kernel/device work, not wall time: auxiliary streams can overlap.
            device_work_us=sum(getattr(event, timing_attr) for event in avgs),
            top_device_op=top.key,
            top_device_us=getattr(top, timing_attr),
        )
    del out, warm_out, grad_out, b, natural_w, a
    _release_cuda_tensors()
    return summary


def profile_raw_gmm_phases(case, batch_sizes, forward_trans_b, dtype, trace_dir):
    print("Raw extension phase profiles: each trace contains only one forward, input-gradient, "
          "or weight-gradient phase; profiler timings are not benchmark timings.", flush=True)
    summaries = []
    for gemm_name, in_dim, out_dim in (
        ("fc1", case["dim"], 4 * case["dim"]),
        ("fc2", 4 * case["dim"], case["dim"]),
    ):
        for phase in ("forward", "input_grad", "weight_grad"):
            summaries.append(_profile_raw_gmm_phase(
                case["name"], phase, gemm_name, in_dim, out_dim,
                batch_sizes, forward_trans_b, dtype, trace_dir))
    available = [summary for summary in summaries if summary["device_work_us"] is not None]
    if available:
        print("Raw phase device-work summary (summed kernel time, not wall time; streams may overlap):",
              flush=True)
        for summary in sorted(available, key=lambda item: item["device_work_us"], reverse=True):
            print(f"  {summary['label']}: device_work_us={summary['device_work_us']:.3f} "
                  f"largest_op={summary['top_device_op']} "
                  f"largest_op_us={summary['top_device_us']:.3f}", flush=True)


def run_profiles(args, dtype):
    if args.backend != "grouped_gemm" or args.device != "cuda":
        raise SystemExit("--mode profile requires --backend grouped_gemm --device cuda")
    case = BENCHMARK_CASES[args.profile_case]
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    canonical, x_cpu = _make_benchmark_inputs(case, dtype)

    # Probe the actual production expressions independently of the diagnostic
    # GroupedGemmMoE class. This is setup work, outside every timed/profiled region.
    production_probe = copy.deepcopy(canonical).to(device="cuda")
    x_probe = x_cpu.to(device="cuda")
    report_production_dtype_conversions(production_probe, x_probe)
    del x_probe, production_probe
    _release_cuda_tensors()

    configs = BENCHMARK_CONFIGS if args.profile_config == "all" else tuple(
        config for config in BENCHMARK_CONFIGS if config[0] == args.profile_config)
    expected_routing = None
    for config_name, forward_trans_b in configs:
        print("=" * 100, flush=True)
        print(f"PROFILE MODE case={case['name']} config={config_name}", flush=True)
        model = _make_benchmark_model(canonical, case, config_name, forward_trans_b,
                                      device="cuda", profile_stages=True)
        x = x_cpu.to(device="cuda").requires_grad_(True)
        snapshot = _routing_snapshot(model.router, x, case["top_k"])
        expected_routing = _assert_identical_routing(snapshot, expected_routing, config_name)
        batch_sizes = torch.bincount(snapshot[0].reshape(-1),
                                     minlength=case["num_experts"]).to(torch.int64)
        grad_out = torch.randn_like(x)
        profile_full_layer(model, x, grad_out, case["name"], config_name,
                           trace_dir, args.profile_warmup)
        del grad_out, snapshot, x, model
        _release_cuda_tensors()
        if config_name != "loop" and not args.skip_raw_gmm_profiles:
            profile_raw_gmm_phases(case, batch_sizes, forward_trans_b, dtype, trace_dir)
        del batch_sizes

    del expected_routing, x_cpu, canonical
    _release_cuda_tensors()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "benchmark", "indexing", "profile"],
                    default="validate",
                    help="performance modes are opt-in; validate preserves the original harness")
    ap.add_argument("--backend", choices=list(BACKENDS), default="reference")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--benchmark-case", choices=["all", *BENCHMARK_CASES], default="all")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--profile-case", choices=list(BENCHMARK_CASES), default="train_e1_k1")
    ap.add_argument("--profile-config", choices=["all", *(name for name, _ in BENCHMARK_CONFIGS)],
                    default="all")
    ap.add_argument("--profile-warmup", type=int, default=1)
    ap.add_argument("--trace-dir", default="/tmp/grouped_gemm_profiles")
    ap.add_argument("--skip-raw-gmm-profiles", action="store_true",
                    help="profile only the annotated full layer, not isolated FC1/FC2 fwd/dgrad/wgrad")
    ap.add_argument("--build-log", help="verbose nvcc build log used to report actual optimization flags")
    args = ap.parse_args()

    if args.backend == "grouped_gemm" and args.device != "cuda":
        raise SystemExit("--backend grouped_gemm requires --device cuda")
    if args.warmup < 1 or args.iterations < 1 or args.profile_warmup < 1:
        raise SystemExit("warmup and iteration counts must be positive")

    log_environment(args)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if args.backend == "grouped_gemm":
        report_extension_binary_and_build_flags(args.build_log)
    if args.mode == "benchmark":
        run_benchmarks(args, dtype)
        return
    if args.mode == "indexing":
        run_indexing_benchmarks(args, dtype)
        return
    if args.mode == "profile":
        run_profiles(args, dtype)
        return

    all_ok = True
    # Both configurations from BACKEND DISPATCH NOTES: forward_trans_b=True
    # (natural nn.Linear layout, forward is cuBLAS / input-grad is
    # CUTLASS-eligible) and forward_trans_b=False (pre-transposed weights,
    # forward is CUTLASS-eligible / input-grad is cuBLAS) -- kept side by
    # side rather than picking one, per instruction.
    for forward_trans_b in (True, False):
        print(f"--- forward_trans_b={forward_trans_b} ---", flush=True)
        for case in CASES:
            ok = run_case(
                name=f"{case['name']}_transb{forward_trans_b}", dim=case["dim"],
                num_experts=case["num_experts"], top_k=case["top_k"], n_tokens=case["n_tokens"],
                normalize_topk=case.get("normalize_topk", True),
                excluded_experts=case.get("excluded_experts", ()),
                backend_name=args.backend, device=args.device, dtype=dtype,
                atol=args.atol, rtol=args.rtol, seed=case.get("seed", 0),
                forward_trans_b=forward_trans_b,
            )
            all_ok = all_ok and ok

    if args.backend == "grouped_gemm":
        print("=" * 100, flush=True)
        print("Runtime kernel-dispatch evidence (profiler-based, not source-inference):", flush=True)
        for forward_trans_b in (True, False):
            profile_gmm_dispatch(BACKENDS[args.backend], dim=64, num_experts=8, n_tokens_per_expert=64,
                                  forward_trans_b=forward_trans_b, device=args.device, dtype=dtype)

    print("=" * 100, flush=True)
    if not all_ok:
        print("VALIDATION FAILED", flush=True)
        raise SystemExit(1)
    print("VALIDATION PASSED", flush=True)


if __name__ == "__main__":
    main()
