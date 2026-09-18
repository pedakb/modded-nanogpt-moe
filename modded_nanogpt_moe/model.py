"""Active dense and mixture-of-experts model implementation."""

import math
from contextlib import nullcontext

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ._grouped_gemm import expert_gmm, implementation_from_environment

_moe_nsys_capture_active = False


def set_moe_nsys_capture_active(active: bool):
    """Called only at the trainer's Nsight capture boundaries."""
    global _moe_nsys_capture_active
    _moe_nsys_capture_active = active


def nsys_range(enabled: bool, name: str):
    """Return an NVTX range only while the opt-in Nsight capture is active."""
    return torch.cuda.nvtx.range(name) if enabled else nullcontext()


def _register_moe_backward_ranges(out, out_sorted, h_act, h_pre, x_sorted):
    """Bracket enqueue intervals on the single-device autograd worker thread.

    Tensor hooks do not change gradients. Ranges delimit activation-gradient
    boundaries, not exclusive kernel time: autograd can interleave parameter
    and router branches. No tensors are retained by the hook closures.
    """
    if not _moe_nsys_capture_active or not torch.is_grad_enabled():
        return
    boundaries = (out, out_sorted, h_act, h_pre, x_sorted)
    names = ("moe_bw.combine", "moe_bw.fc2", "moe_bw.activation", "moe_bw.fc1")
    range_open = False
    cleanup_queued = False

    def close_range():
        nonlocal range_open
        if range_open:
            torch.cuda.nvtx.range_pop()
            range_open = False

    def finish_backward():
        nonlocal cleanup_queued
        try:
            close_range()
        finally:
            cleanup_queued = False

    def boundary_hook(next_name):
        def hook(gradient):
            nonlocal range_open, cleanup_queued
            close_range()
            if not _moe_nsys_capture_active or next_name is None:
                return
            # Run on the autograd worker after this backward finishes, including
            # partial grad() traversals which may omit the final boundary hook.
            if not cleanup_queued:
                torch.autograd.Variable._execution_engine.queue_callback(finish_backward)
                cleanup_queued = True
            torch.cuda.nvtx.range_push(next_name)
            range_open = True
        return hook

    for index, tensor in enumerate(boundaries):
        if tensor.requires_grad:
            next_name = (
                names[index] if index < len(names)
                and boundaries[index + 1].requires_grad else None)
            tensor.register_hook(boundary_hook(next_name))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        self.num_heads = dim // head_dim
        if self.num_heads < 1:
            raise ValueError(
                f"attention model dimension {dim} must provide at least one "
                f"head of dimension {head_dim}")
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

def resolve_mlp_hidden_dim(model_dim: int, mlp_ratio: float = 4) -> int:
    """Resolve an experiment-level MLP ratio without silently rounding it."""
    if isinstance(mlp_ratio, bool) or not isinstance(mlp_ratio, (int, float)):
        raise ValueError(f"mlp_ratio must be a finite positive number, got {mlp_ratio!r}")
    try:
        ratio = float(mlp_ratio)
    except OverflowError as exc:
        raise ValueError(f"mlp_ratio must be finite and positive, got {mlp_ratio!r}") from exc
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"mlp_ratio must be finite and positive, got {mlp_ratio!r}")
    try:
        product = model_dim * ratio
    except OverflowError as exc:
        raise ValueError(
            f"model_dim * mlp_ratio must be a finite positive integer, got "
            f"{model_dim} * {mlp_ratio!r}") from exc
    if not math.isfinite(product) or product <= 0 or not product.is_integer():
        raise ValueError(
            f"model_dim * mlp_ratio must be a finite positive integer, got "
            f"{model_dim} * {mlp_ratio!r} = {product!r}")
    return int(product)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = 4 * dim if hidden_dim is None else hidden_dim
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim!r}")
        self.fc = Linear(dim, hidden_dim)
        self.proj = Linear(hidden_dim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x


class _ExpertSegmentBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, batch_sizes, sorted_experts):
        ctx.save_for_backward(batch_sizes)
        # index_select is inside a custom forward: its scatter/indexing backward
        # is never built. Save counts only, not the expanded bias or activations.
        return x + bias.index_select(0, sorted_experts)

    @staticmethod
    def backward(ctx, grad_output):
        grad_bias = None
        if ctx.needs_input_grad[1]:
            (batch_sizes,) = ctx.saved_tensors
            if (grad_output.is_cuda and not torch.is_grad_enabled()
                    and grad_output.dtype in (torch.float16, torch.bfloat16, torch.float32)):
                from ._segmented_bias import segmented_bias_grad
                grad_bias = segmented_bias_grad(grad_output, batch_sizes)
            else:
                # CPU/double/reference higher-order path. segment_reduce sums in
                # its input dtype, so low-precision gradients need FP32 promotion.
                values = (grad_output.float() if grad_output.dtype in
                          (torch.float16, torch.bfloat16) else grad_output)
                # Trusted bincount output; unsafe skips GPU .item() validation.
                grad_bias = torch.segment_reduce(
                    values, "sum", lengths=batch_sizes, axis=0, unsafe=True,
                ).to(grad_output.dtype)
        return grad_output if ctx.needs_input_grad[0] else None, grad_bias, None, None


def add_bias_by_expert_segments(x: Tensor, bias: Tensor, batch_sizes: Tensor,
                                sorted_experts: Tensor):
    """Vectorized bias add and contiguous-segment bias-gradient sum.

    Metadata must be the existing sorted assignment IDs and their bincount,
    both on x.device. Zero counts produce present, exactly-zero bias gradients.
    Callers guarantee count/assignment consistency; do not synchronize to check
    GPU tensor values here. All tensor work is independent of the expert count
    in launch count (not in work), with no per-expert Python loop.
    """
    assert x.ndim == bias.ndim == 2 and x.shape[1] == bias.shape[1]
    assert x.dtype == bias.dtype and x.device == bias.device
    assert batch_sizes.shape == (bias.shape[0],) and batch_sizes.dtype == torch.int64
    assert sorted_experts.shape == (x.shape[0],) and sorted_experts.dtype == torch.int64
    assert batch_sizes.device == sorted_experts.device == x.device
    return _ExpertSegmentBias.apply(x, bias, batch_sizes, sorted_experts)


def combine_expert_outputs(out_sorted: Tensor, topk_weights: Tensor, order: Tensor):
    """Combine a trusted sorted-row -> assignment permutation and routed values.

    CUDA uses a compact assignment -> sorted-row lookup constructed once from
    order, then reuses it in backward. No routing/sort is recomputed. CPU and
    double precision retain the exact eager reference for portable checks.
    """
    assert out_sorted.ndim == topk_weights.ndim == 2 and order.ndim == 1
    n, k = topk_weights.shape
    d = out_sorted.shape[1]
    assert k > 0 and d > 0 and out_sorted.shape[0] == order.numel() == n * k
    assert out_sorted.device == topk_weights.device == order.device
    assert out_sorted.dtype == topk_weights.dtype and order.dtype == torch.int64
    if out_sorted.is_cuda and out_sorted.dtype in (torch.bfloat16, torch.float16, torch.float32):
        from ._combine import fused_combine
        return fused_combine(out_sorted, topk_weights, order)
    out_flat = torch.empty_like(out_sorted)
    out_flat[order] = out_sorted
    return (out_flat.view(n, k, d) * topk_weights.unsqueeze(-1)).sum(dim=1)


class MoE(nn.Module):
    """Top-k routed sparse MoE. Each expert is an MLP identical in architecture,
    init, and dtype behavior to the dense MLP above. num_experts=1, top_k=1 reduces
    exactly to a single MLP: softmax over one logit is always 1, so expert 0 receives
    every token with routing weight 1.

    With the default moe_parameter_layout="modulelist", moe_backend selects
    HOW self.experts is executed, not how it is stored:
      "loop" (default): the original Python loop over experts, unchanged.
      "grouped_gemm": packs tokens by expert and calls the fanshiqing/
        grouped_gemm CUDA extension (package `grouped_gemm`, PyPI
        `nv-grouped-gemm`) for both expert matmuls. self.experts remains the
        exact same nn.ModuleList of MLP for both backends -- same checkpoint
        keys, same Parameter identities, same optimizer grouping by
        p.ndim/p.shape. grouped_gemm only changes how those Parameters are
        *read* at forward time (stacked into one tensor per matmul, freshly
        every call -- see _forward_grouped_gemm's docstring for the two
        costs specific to this that are not present in "loop").

    Experimental moe_parameter_layout="packed" requires grouped_gemm and stores
    FC/projection weights directly as [E,D,H]/[E,H,D], with [E,H]/[E,D] biases.
    It preserves expert initialization and computation but has different state
    keys. Optimizer construction explicitly assigns packed biases to AdamW and
    interprets packed weights as independent transposed expert matrices in Muon.

    Correctness of "grouped_gemm" against "loop" (identical weights,
    identical inputs, including E=8/k=2 with deliberately empty experts) was
    validated in tools/validate_grouped_gemm.py, then
    confirmed on an LS6 A100 in BF16 (all 8 cases, both trans_b layouts, a
    profiler-confirmed CUTLASS GemmGrouped kernel). This class's own
    forward-time packing here is written independently of that harness (no
    import from it) but implements the same validated pipeline against the
    production nn.ModuleList instead of a separate ParameterList."""
    def __init__(self, dim: int, num_experts: int, top_k: int, normalize_topk: bool = True,
                 moe_backend: str = "loop", hidden_dim: int | None = None,
                 moe_parameter_layout: str = "modulelist"):
        super().__init__()
        assert 1 <= top_k <= num_experts
        assert moe_backend in ("loop", "grouped_gemm"), f"unknown moe_backend: {moe_backend!r}"
        if moe_parameter_layout not in ("modulelist", "packed"):
            raise ValueError(f"unknown moe_parameter_layout: {moe_parameter_layout!r}")
        if moe_parameter_layout == "packed" and moe_backend != "grouped_gemm":
            raise ValueError("packed parameters require moe_backend='grouped_gemm'")
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_topk = normalize_topk
        self.moe_backend = moe_backend
        self.moe_parameter_layout = moe_parameter_layout
        self._routing_diagnostics = None  # Transient observer, never checkpointed.
        self.router = Linear(dim, num_experts)
        if moe_parameter_layout == "modulelist":
            self.experts = nn.ModuleList(MLP(dim, hidden_dim) for _ in range(num_experts))
        else:
            hidden_dim = 4 * dim if hidden_dim is None else hidden_dim
            if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
                raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim!r}")
            self.fc_weight = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
            self.fc_bias = nn.Parameter(torch.empty(num_experts, hidden_dim))
            self.proj_weight = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
            self.proj_bias = nn.Parameter(torch.empty(num_experts, dim))
            # Preserve nn.Linear constructor initialization and RNG order exactly.
            # Temporary experts are discarded; none are stored or built in forward.
            with torch.no_grad():
                for index in range(num_experts):
                    expert = MLP(dim, hidden_dim)
                    self.fc_weight[index].copy_(expert.fc.weight.mT)
                    self.fc_bias[index].copy_(expert.fc.bias)
                    self.proj_weight[index].copy_(expert.proj.weight.mT)
                    self.proj_bias[index].copy_(expert.proj.bias)
        self.gmm_implementation = implementation_from_environment() if moe_backend == "grouped_gemm" else "extension"
        self._gmm = None
        if moe_backend == "grouped_gemm" and self.gmm_implementation == "extension":
            # Imported only when this backend is selected -- "loop" (the
            # default) never touches this import, and dense models never
            # construct a MoE at all.
            try:
                import grouped_gemm
            except ImportError as e:
                raise ImportError(
                    "moe_backend='grouped_gemm' requires the `grouped_gemm` package "
                    "(PyPI: nv-grouped-gemm) to be importable in this environment. It is "
                    "NOT a project dependency (not in pyproject.toml/uv.lock) and must be "
                    "installed manually -- see the INSTALL section in "
                    "tools/validate_grouped_gemm.py for exact, "
                    "pinned, architecture-specific build commands. It also requires CUDA "
                    "(no CPU fallback exists in that extension)."
                ) from e
            self._gmm = grouped_gemm.ops.gmm

    def forward(self, x: Tensor):
        if self.moe_backend == "grouped_gemm":
            return self._forward_grouped_gemm(x)
        return self._forward_loop(x)

    def _forward_loop(self, x: Tensor):
        B, T, D = x.shape
        x = x.view(-1, D)

        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_experts = routing_weights.topk(self.top_k, dim=-1)
        if self._routing_diagnostics is not None:
            self._routing_diagnostics(router_logits.detach(), routing_weights.detach(), topk_experts.detach())
        if self.normalize_topk:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.type_as(x)

        out = x.new_zeros(x.shape)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(topk_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            out.index_add_(0, token_idx, expert(x[token_idx]) * topk_weights[token_idx, slot_idx, None])
        return out.view(B, T, D)

    def _forward_grouped_gemm(self, x: Tensor):
        """route -> pack assignments by expert -> grouped FC1 -> bias ->
        ReLU^2 -> grouped FC2 -> bias -> routing weight -> combine. Router
        softmax/top-k/renormalize and the final weighted-combine are
        byte-for-byte the same computation as _forward_loop -- only expert
        execution differs. Dropless: every one of the N*top_k assignments is
        computed, no capacity limit, no token dropping, no padding.

        Backend-specific costs (no speedup claim is made here):
          - batch_sizes.cpu() below is a required device-to-host
            synchronization -- grouped_gemm's C++ extension asserts its
            token-count tensor is CPU-resident int64. This happens once per
            MoE layer per forward call (so once per layer per microbatch,
            i.e. num_layers times per training step).
          - in the default ModuleList layout, four torch.stack(...) calls (two also
            .transpose(-2,-1).contiguous()) rebuild the [E, ...] weight
            tensors from self.experts fresh on every call rather than
            caching them across calls -- O(num_experts) extra allocation
            and copy per matmul per layer per call. Packed layout avoids these
            reconstructions, but retains master-parameter-to-activation dtype casts.
        """
        B, T, D = x.shape
        x = x.view(-1, D)
        N = x.shape[0]
        device = x.device

        with nsys_range(_moe_nsys_capture_active, "moe.router_topk"):
            router_logits = self.router(x)
            routing_weights = F.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_experts = routing_weights.topk(self.top_k, dim=-1)
            if self._routing_diagnostics is not None:
                self._routing_diagnostics(router_logits.detach(), routing_weights.detach(), topk_experts.detach())
            if self.normalize_topk:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.type_as(x)

        # ---- pack assignments by expert ----
        with nsys_range(_moe_nsys_capture_active, "moe.pack"):
            flat_experts = topk_experts.reshape(-1)                                    # [N*k]
            flat_tokens = torch.arange(N, device=device).repeat_interleave(self.top_k)  # [N*k]
            order = torch.argsort(flat_experts, stable=True)
            sorted_experts = flat_experts[order]
            x_sorted = x[flat_tokens[order]]  # gather of ACTIVATIONS; never a weight copy

            # Required CPU int64 token-count metadata -- explicit D2H sync, see docstring.
            batch_sizes_device = torch.bincount(sorted_experts, minlength=self.num_experts).to(torch.int64)
            batch_sizes = batch_sizes_device.cpu()
            # Reuse counts for native grouped GEMM; no new host transfer or sort.
            offsets = (batch_sizes_device.cumsum(0, dtype=torch.int32)
                       if self.gmm_implementation == "torch" else None)

        # Both layouts supply contiguous [E,in,out] weights for trans_b=False.
        # Keep the existing range name for comparison; packed only casts dtype.
        with nsys_range(_moe_nsys_capture_active, "moe.stack_params"):
            if self.moe_parameter_layout == "packed":
                # Preserve FP32 master parameters and the existing activation-dtype
                # conversion, without stack/transpose/contiguous reconstruction.
                fc_w = self.fc_weight.type_as(x_sorted)
                fc_b = self.fc_bias.type_as(x_sorted)
                proj_w = self.proj_weight.type_as(x_sorted)
                proj_b = self.proj_bias.type_as(x_sorted)
            else:
                fc_w = torch.stack([e.fc.weight for e in self.experts]).transpose(-2, -1).contiguous().type_as(x_sorted)
                fc_b = torch.stack([e.fc.bias for e in self.experts]).type_as(x_sorted)
                proj_w = torch.stack([e.proj.weight for e in self.experts]).transpose(-2, -1).contiguous().type_as(x_sorted)
                proj_b = torch.stack([e.proj.bias for e in self.experts]).type_as(x_sorted)

        with nsys_range(_moe_nsys_capture_active, "moe.fc1"):
            h_pre = add_bias_by_expert_segments(
                expert_gmm(x_sorted, fc_w, batch_sizes, offsets, self.gmm_implementation,
                           self._gmm, "fc1", _moe_nsys_capture_active),
                fc_b, batch_sizes_device, sorted_experts)
        with nsys_range(_moe_nsys_capture_active, "moe.activation"):
            h_act = h_pre.relu().square()
        with nsys_range(_moe_nsys_capture_active, "moe.fc2"):
            out_sorted = add_bias_by_expert_segments(
                expert_gmm(h_act.type_as(x_sorted), proj_w, batch_sizes, offsets, self.gmm_implementation,
                           self._gmm, "fc2", _moe_nsys_capture_active),
                proj_b, batch_sizes_device, sorted_experts)

        # ---- unpermute + weighted combine ----
        # NOTE on optimizer semantics, not just numerics: when an expert
        # gets zero tokens (batch_sizes[e] == 0), "loop" above never touches
        # that expert's Parameters, so their .grad stays None after
        # backward() -- which crashes this file's `assert p.grad is not
        # None` in the training loop today (a known, still-unfixed
        # limitation of "loop" for num_experts > 1). "grouped_gemm" instead
        # produces a PRESENT, exactly-zero .grad for that expert (verified
        # on an LS6 A100). That avoids the crash, but is NOT equivalent to
        # "no update happened": both optimizers used in this file apply
        # weight decay unconditionally (regardless of the gradient value),
        # and Muon additionally computes an orthogonalized update from
        # momentum.lerp_(grad=0, 1-mu) -- which decays prior momentum
        # toward zero rather than leaving it untouched, and can still
        # produce a NONZERO parameter update from residual momentum alone.
        # Whether that is benign or causes drift for experts that are
        # repeatedly starved of tokens has not been checked empirically.
        with nsys_range(_moe_nsys_capture_active, "moe.combine"):
            out = combine_expert_outputs(out_sorted, topk_weights, order)
            out = out.view(B, T, D)
        if _moe_nsys_capture_active:
            _register_moe_backward_ranges(out, out_sorted, h_act, h_pre, x_sorted)
        return out

class Block(nn.Module):
    def __init__(self, dim: int, mlp_type: str = "dense", num_experts: int = 1,
                 top_k: int = 1, normalize_topk: bool = True, moe_backend: str = "loop",
                 hidden_dim: int | None = None, moe_parameter_layout: str = "modulelist"):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        if mlp_type == "dense":
            self.mlp = MLP(dim, hidden_dim)
        elif mlp_type == "moe":
            self.mlp = MoE(dim, num_experts=num_experts, top_k=top_k, normalize_topk=normalize_topk,
                            moe_backend=moe_backend, hidden_dim=hidden_dim,
                            moe_parameter_layout=moe_parameter_layout)
        else:
            raise ValueError(f"unknown mlp_type: {mlp_type!r}")
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, mlp_type: str = "dense",
                 num_experts: int = 1, top_k: int = 1, normalize_topk: bool = True,
                 moe_backend: str = "loop", mlp_ratio: float = 4,
                 moe_parameter_layout: str = "modulelist"):
        super().__init__()
        if moe_parameter_layout not in ("modulelist", "packed"):
            raise ValueError(f"unknown moe_parameter_layout: {moe_parameter_layout!r}")
        if moe_parameter_layout == "packed" and (mlp_type != "moe" or moe_backend != "grouped_gemm"):
            raise ValueError("packed parameters require grouped_gemm MoE")
        hidden_dim = resolve_mlp_hidden_dim(model_dim, mlp_ratio)
        self.model_dim = model_dim
        self.mlp_ratio = mlp_ratio
        self.hidden_dim = hidden_dim
        self.moe_parameter_layout = moe_parameter_layout
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            Block(model_dim, mlp_type=mlp_type, num_experts=num_experts, top_k=top_k,
                  normalize_topk=normalize_topk, moe_backend=moe_backend,
                  hidden_dim=hidden_dim, moe_parameter_layout=moe_parameter_layout)
            for _ in range(num_layers)
        ])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


@torch.no_grad()
def initialize_model_parameters(model):
    """Trainer initialization, preserving the reference expert RNG draw order."""
    packed = {
        module.router.bias: module for module in model.modules()
        if isinstance(module, MoE) and module.moe_parameter_layout == "packed"
    }
    packed_parameters = {
        p for module in packed.values()
        for p in (module.fc_weight, module.fc_bias, module.proj_weight, module.proj_bias)
    }
    for name, p in model.named_parameters():
        if p in packed_parameters:
            continue
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")
        if p in packed:
            module = packed[p]
            # Draw each FC weight in its original contiguous [H,D] order, after
            # router initialization, exactly where ModuleList traversal drew it.
            for index in range(module.num_experts):
                weight = torch.empty_like(module.fc_weight[index].mT,
                                          memory_format=torch.contiguous_format)
                weight.normal_(std=0.33**0.5 / weight.size(-1)**0.5)
                module.fc_weight[index].copy_(weight.mT)
                module.fc_bias[index].zero_()
                module.proj_weight[index].zero_()
                module.proj_bias[index].zero_()


def eager_prefix(model: "GPT", inputs: Tensor) -> Tensor:
    """GPT.forward's embed -> norm1 -> blocks section, run eagerly. Used in
    place of whole-model compilation for mlp_type="moe": the expert dispatch
    loop inside Block/MoE.forward contains a data-dependent-shaped op inside
    a Python loop, which torch.compile cannot partially graph-break out of --
    it skips compiling GPT.forward entirely (dense and MoE alike), dragging
    the much larger head/loss computation into eager mode with it. Running
    this prefix eagerly (unchanged from GPT.forward, no dispatch/routing
    changes) and compiling only the head/loss tail in isolation (see
    make_head_loss) avoids that, verified on an A100: MoE(E=1,k=1) eval peak
    allocated dropped from an OOM (>37 GiB) to 6.944 GiB."""
    x = model.norm1(model.embed(inputs))
    for block in model.blocks:
        x = block(x)
    return x


def make_head_loss(model: "GPT"):
    """GPT.forward's tail -- norm2 -> proj -> float -> softcap ->
    cross_entropy(reduction="sum") -- as a standalone callable closing over
    model.norm2/model.proj directly (the same nn.Parameter objects; no
    duplication), meant to be wrapped in torch.compile(fullgraph=True) and
    called on eager_prefix's output. Byte-for-byte the same computation,
    dtype casts, and loss reduction/scaling as GPT.forward -- this function
    exists so the tail can be compiled independently of the (uncompiled)
    block loop, not to change what is computed."""
    def head_loss(x: Tensor, targets: Tensor) -> Tensor:
        logits = model.proj(model.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")
    return head_loss
