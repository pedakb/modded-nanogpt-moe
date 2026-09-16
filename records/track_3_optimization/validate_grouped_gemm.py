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
    kernels' correctness. This is what --backend reference exercises, and
    is what has actually been run (on this machine, CPU, no GPU available)
    while writing this harness.
  - real_gmm: a thin wrapper around the real `grouped_gemm.ops.gmm`
    (fanshiqing/grouped_gemm). Requires CUDA and the package installed --
    see INSTALL. This is the actual milestone validation and is PENDING: it
    has not been run anywhere yet.

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
INSTALL (LS6 / A100 / SM80 -- validate this first; nothing has been run)

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
import os
import sys

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
                 forward_trans_b=True):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_topk = normalize_topk
        self.gmm_fn = gmm_fn
        self.forward_trans_b = forward_trans_b
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

    def forward(self, x):
        B, T, D = x.shape
        x = x.reshape(-1, D)
        N = x.shape[0]
        device = x.device

        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_experts = routing_weights.topk(self.top_k, dim=-1)
        if self.normalize_topk:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.type_as(x)

        # ---- pack assignments by expert (backend-independent) ----
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
        batch_sizes = torch.bincount(sorted_experts, minlength=self.num_experts).to(torch.int64).cpu()

        fc_w = self._stacked_weight(self.fc_weight).type_as(x_sorted)
        fc_b = torch.stack(list(self.fc_bias)).type_as(x_sorted)        # [E, 4D]
        proj_w = self._stacked_weight(self.proj_weight).type_as(x_sorted)
        proj_b = torch.stack(list(self.proj_bias)).type_as(x_sorted)    # [E, D]

        h = self.gmm_fn(x_sorted, fc_w, batch_sizes, trans_b=self.forward_trans_b) + fc_b[sorted_experts]
        h = h.relu().square()
        out_sorted = (self.gmm_fn(h.type_as(x_sorted), proj_w, batch_sizes, trans_b=self.forward_trans_b)
                      + proj_b[sorted_experts])

        # ---- unpermute + weighted combine (backend-independent) ----
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


def log_environment(args):
    print("=" * 100, flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"device={torch.cuda.get_device_name(0)} capability=sm_{cap[0]}{cap[1]}", flush=True)
    else:
        print("CUDA not available on this machine -- only --backend reference can run here.", flush=True)
    print(f"backend={args.backend} device={args.device} dtype={args.dtype}", flush=True)
    if args.backend == "grouped_gemm":
        import grouped_gemm
        print(f"grouped_gemm module file: {getattr(grouped_gemm, '__file__', 'unknown')}", flush=True)
        print(f"grouped_gemm version attr: {getattr(grouped_gemm, '__version__', 'unknown (not exported)')}",
              flush=True)
        print("This run exercises the REAL CUDA extension, in BOTH forward_trans_b configs "
              "(see the module docstring's BACKEND DISPATCH NOTES): weight-grad is always "
              "cuBLAS regardless; forward and input-grad each become CUTLASS-eligible in "
              "exactly one of the two configs, and only if this extension was built with "
              "GROUPED_GEMM_DEVICE_CAPABILITY=80 defined (see INSTALL). These are source-"
              "derived EXPECTATIONS, not runtime facts -- profile_gmm_dispatch() below is "
              "the actual runtime evidence.", flush=True)
    else:
        print("This run uses the pure-PyTorch reference backend: it validates pack/unpack/"
              "combine/autograd scaffolding, NOT the real grouped_gemm CUDA kernels. "
              "The real-CUDA-kernel validation is PENDING until run with "
              "--backend grouped_gemm on a CUDA machine with the package installed.",
              flush=True)
    print("=" * 100, flush=True)


def profile_gmm_dispatch(gmm_fn, dim, num_experts, n_tokens_per_expert, forward_trans_b, device, dtype):
    """Run gmm_fn's forward AND backward under torch.profiler and report
    which CUDA kernels actually executed -- REAL runtime evidence for which
    branch (CUTLASS vs cuBLAS) was taken, rather than the source-derived
    expectations in BACKEND DISPATCH NOTES. Only meaningful for
    --backend grouped_gemm on CUDA; main() does not call this for
    --backend reference (plain torch.matmul, no dispatch ambiguity to
    confirm)."""
    from torch.profiler import ProfilerActivity, profile

    hdim = 4 * dim
    batch_sizes = torch.full((num_experts,), n_tokens_per_expert, dtype=torch.int64)  # CPU, per real_gmm's contract
    total = int(batch_sizes.sum())
    a = torch.randn(total, dim, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(num_experts, hdim, dim, dtype=dtype, device=device, requires_grad=True)
    b_arg = w if forward_trans_b else w.detach().transpose(-2, -1).contiguous().requires_grad_(True)

    def call():
        out = gmm_fn(a, b_arg, batch_sizes, trans_b=forward_trans_b)
        out.sum().backward()

    call()  # warm up (first-call cuBLAS handle / workspace allocation) before profiling
    a.grad = None
    b_arg.grad = None

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()

    avgs = prof.key_averages()
    names = [e.key for e in avgs]
    cutlass_hits = [n for n in names if "cutlass" in n.lower()]
    cublas_hits = [n for n in names if "cublas" in n.lower() or "gemm" in n.lower()]

    label = f"forward_trans_b={forward_trans_b}"
    print(f"[profile:{label}] {len(names)} distinct op/kernel names recorded", flush=True)
    print(f"[profile:{label}] names containing 'cutlass': {cutlass_hits or '(none)'}", flush=True)
    print(f"[profile:{label}] names containing 'cublas'/'gemm': {cublas_hits or '(none)'}", flush=True)
    top = sorted(avgs, key=lambda e: getattr(e, "self_cuda_time_total", 0), reverse=True)[:10]
    print(f"[profile:{label}] top-10 ops by self CUDA time:", flush=True)
    for e in top:
        print(f"    {e.key}  self_cuda_time_total_us={getattr(e, 'self_cuda_time_total', 0)}  count={e.count}",
              flush=True)
    return dict(cutlass_observed=bool(cutlass_hits), cublas_or_gemm_observed=bool(cublas_hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS), default="reference")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    args = ap.parse_args()

    if args.backend == "grouped_gemm" and args.device != "cuda":
        raise SystemExit("--backend grouped_gemm requires --device cuda")

    log_environment(args)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
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
