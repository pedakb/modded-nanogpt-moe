"""Active dense and mixture-of-experts model implementation."""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

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


def add_bias_by_expert_segments(x: Tensor, bias: Tensor, batch_sizes: Tensor):
    """Add one expert bias to each contiguous packed-token segment.

    `batch_sizes` is the CPU int64 metadata already required by grouped_gemm.
    Splitting by those counts and broadcasting each row avoids the repeated
    advanced-index backward in `bias[sorted_experts]`. Empty experts still
    participate through a zero-length chunk, so their bias gradient is a
    present, exactly-zero tensor rather than None. This is deliberately
    generic for every expert count; there is no E=1 special case.
    """
    assert batch_sizes.device.type == "cpu" and batch_sizes.dtype == torch.int64
    counts = batch_sizes.tolist()
    assert len(counts) == bias.shape[0] and sum(counts) == x.shape[0]
    return torch.cat([
        x_segment + bias_row
        for x_segment, bias_row in zip(x.split(counts, dim=0), bias.unbind(0), strict=True)
    ], dim=0)


class MoE(nn.Module):
    """Top-k routed sparse MoE. Each expert is an MLP identical in architecture,
    init, and dtype behavior to the dense MLP above. num_experts=1, top_k=1 reduces
    exactly to a single MLP: softmax over one logit is always 1, so expert 0 receives
    every token with routing weight 1.

    moe_backend selects HOW self.experts is executed, not how it is stored:
      "loop" (default): the original Python loop over experts, unchanged.
      "grouped_gemm": packs tokens by expert and calls the fanshiqing/
        grouped_gemm CUDA extension (package `grouped_gemm`, PyPI
        `nv-grouped-gemm`) for both expert matmuls. self.experts remains the
        exact same nn.ModuleList of MLP for both backends -- same checkpoint
        keys, same Parameter identities, same optimizer grouping by
        p.ndim/p.shape. grouped_gemm only changes how those Parameters are
        *read* at forward time (stacked into one tensor per matmul, freshly
        every call -- see _forward_grouped_gemm's docstring for the two
        costs specific to this that are not present in "loop", neither
        benchmarked yet).

    Correctness of "grouped_gemm" against "loop" (identical weights,
    identical inputs, including E=8/k=2 with deliberately empty experts) was
    validated in records/track_3_optimization/validate_grouped_gemm.py, then
    confirmed on an LS6 A100 in BF16 (all 8 cases, both trans_b layouts, a
    profiler-confirmed CUTLASS GemmGrouped kernel). This class's own
    forward-time packing here is written independently of that harness (no
    import from it) but implements the same validated pipeline against the
    production nn.ModuleList instead of a separate ParameterList."""
    def __init__(self, dim: int, num_experts: int, top_k: int, normalize_topk: bool = True,
                 moe_backend: str = "loop", hidden_dim: int | None = None):
        super().__init__()
        assert 1 <= top_k <= num_experts
        assert moe_backend in ("loop", "grouped_gemm"), f"unknown moe_backend: {moe_backend!r}"
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_topk = normalize_topk
        self.moe_backend = moe_backend
        self.router = Linear(dim, num_experts)
        self.experts = nn.ModuleList(MLP(dim, hidden_dim) for _ in range(num_experts))
        if moe_backend == "grouped_gemm":
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
                    "records/track_3_optimization/validate_grouped_gemm.py for exact, "
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

        Two costs specific to this backend that _forward_loop does not have,
        NEITHER benchmarked yet (no speedup or memory-feasibility claim is
        made anywhere in this module):
          - batch_sizes.cpu() below is a required device-to-host
            synchronization -- grouped_gemm's C++ extension asserts its
            token-count tensor is CPU-resident int64. This happens once per
            MoE layer per forward call (so once per layer per microbatch,
            i.e. num_layers times per training step).
          - the four torch.stack(...) calls (two of which also
            .transpose(-2,-1).contiguous()) rebuild the [E, ...] weight
            tensors from self.experts fresh on every call rather than
            caching them across calls -- O(num_experts) extra allocation
            and copy per matmul per layer per call.
        """
        B, T, D = x.shape
        x = x.view(-1, D)
        N = x.shape[0]
        device = x.device

        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_experts = routing_weights.topk(self.top_k, dim=-1)
        if self.normalize_topk:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.type_as(x)

        # ---- pack assignments by expert ----
        flat_experts = topk_experts.reshape(-1)                                    # [N*k]
        flat_tokens = torch.arange(N, device=device).repeat_interleave(self.top_k)  # [N*k]
        order = torch.argsort(flat_experts, stable=True)
        sorted_experts = flat_experts[order]
        x_sorted = x[flat_tokens[order]]  # gather of ACTIVATIONS; never a weight copy

        # Required CPU int64 token-count metadata -- explicit D2H sync, see docstring.
        batch_sizes = torch.bincount(sorted_experts, minlength=self.num_experts).to(torch.int64).cpu()

        # Differentiable, contiguous transposed weight stacks (trans_b=False):
        # gradients flow back to self.experts[i].fc/proj.weight exactly as
        # under "loop" -- these are views/copies built fresh per call, not
        # new stored Parameters. trans_b=False is the layout validated end
        # to end on an LS6 A100 with a profiler-confirmed CUTLASS kernel.
        fc_w = torch.stack([e.fc.weight for e in self.experts]).transpose(-2, -1).contiguous().type_as(x_sorted)
        fc_b = torch.stack([e.fc.bias for e in self.experts]).type_as(x_sorted)
        proj_w = torch.stack([e.proj.weight for e in self.experts]).transpose(-2, -1).contiguous().type_as(x_sorted)
        proj_b = torch.stack([e.proj.bias for e in self.experts]).type_as(x_sorted)

        h = add_bias_by_expert_segments(
            self._gmm(x_sorted, fc_w, batch_sizes, trans_b=False), fc_b, batch_sizes)
        h = h.relu().square()
        out_sorted = add_bias_by_expert_segments(
            self._gmm(h.type_as(x_sorted), proj_w, batch_sizes, trans_b=False), proj_b, batch_sizes)

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
        out_flat = torch.empty_like(out_sorted)
        out_flat[order] = out_sorted
        out = (out_flat.view(N, self.top_k, D) * topk_weights.unsqueeze(-1)).sum(dim=1)
        return out.view(B, T, D)

class Block(nn.Module):
    def __init__(self, dim: int, mlp_type: str = "dense", num_experts: int = 1,
                 top_k: int = 1, normalize_topk: bool = True, moe_backend: str = "loop",
                 hidden_dim: int | None = None):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        if mlp_type == "dense":
            self.mlp = MLP(dim, hidden_dim)
        elif mlp_type == "moe":
            self.mlp = MoE(dim, num_experts=num_experts, top_k=top_k, normalize_topk=normalize_topk,
                            moe_backend=moe_backend, hidden_dim=hidden_dim)
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
                 moe_backend: str = "loop", mlp_ratio: float = 4):
        super().__init__()
        hidden_dim = resolve_mlp_hidden_dim(model_dim, mlp_ratio)
        self.model_dim = model_dim
        self.mlp_ratio = mlp_ratio
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([
            Block(model_dim, mlp_type=mlp_type, num_experts=num_experts, top_k=top_k,
                  normalize_topk=normalize_topk, moe_backend=moe_backend,
                  hidden_dim=hidden_dim)
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
