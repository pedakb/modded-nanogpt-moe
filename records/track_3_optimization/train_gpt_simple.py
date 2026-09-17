"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import platform
import random
import socket
import subprocess
import sys
import tempfile
import uuid
import time
import math
from datetime import datetime, timezone
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


CHECKPOINT_FORMAT_VERSION = 1


def nsys_range(enabled: bool, name: str):
    """Return an NVTX range only while the opt-in Nsight capture is active."""
    return torch.cuda.nvtx.range(name) if enabled else nullcontext()


########################################
#              Dataloader              #
########################################

def _read_data_shard_header(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    return header, num_tokens


def _load_data_shard(file: Path, pin_memory=True):
    _, num_tokens = _read_data_shard_header(file)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=pin_memory)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def _data_shard_identity(file: Path):
    header, num_tokens = _read_data_shard_header(file)
    return {
        "name": file.name,
        "size_bytes": file.stat().st_size,
        "num_tokens": num_tokens,
        "header": header.tolist(),
    }


class DistributedDataLoader:
    """Sequential shard loader with an explicit, relocatable cursor state."""
    def __init__(self, filename_pattern: str, batch_size: int, seq_len=1024,
                 data_root: str | Path | None = None, world_size: int | None = None,
                 rank: int | None = None, device: str | torch.device = "cuda"):
        self.filename_pattern = filename_pattern
        self.data_root = Path.cwd() if data_root is None else Path(data_root)
        self.files = sorted(self.data_root.glob(filename_pattern))
        if not self.files:
            raise FileNotFoundError(
                f"no data shards match {filename_pattern!r} under {self.data_root}")
        distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if world_size is None and distributed else (world_size or 1)
        self.rank = dist.get_rank() if rank is None and distributed else (rank or 0)
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} is invalid for world_size {self.world_size}")
        if batch_size % self.world_size != 0:
            raise ValueError("batch_size must be divisible by world_size")
        self.batch_size = batch_size
        self.local_batch_size = batch_size // self.world_size
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.shard_identities = [_data_shard_identity(file) for file in self.files]
        self.shard_index = 0
        self.pos = 0
        self.tokens = _load_data_shard(
            self.files[self.shard_index], pin_memory=self.device.type == "cuda")

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos + self.batch_size + 1 >= len(self.tokens):
            self.shard_index += 1
            if self.shard_index >= len(self.files):
                raise StopIteration("training data shards exhausted")
            self.tokens = _load_data_shard(
                self.files[self.shard_index], pin_memory=self.device.type == "cuda")
            self.pos = 0
        start = self.pos + self.rank * self.local_batch_size
        buf = self.tokens[start:][:self.local_batch_size + 1]
        inputs = buf[:-1].to(device=self.device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=self.device, dtype=torch.int64, non_blocking=True)
        self.pos += self.batch_size
        return inputs.view(-1, self.seq_len), targets.view(-1, self.seq_len)

    def state_dict(self):
        return {
            "format_version": 1,
            "filename_pattern": self.filename_pattern,
            "shards": self.shard_identities,
            "shard_index": self.shard_index,
            "token_offset": self.pos,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "world_size": self.world_size,
            "rank": self.rank,
            "epoch": 0,
            "shuffle": False,
        }

    def load_state_dict(self, state):
        if state.get("format_version") != 1:
            raise ValueError(f"unsupported data-loader state version: {state.get('format_version')}")
        expected = {
            "filename_pattern": self.filename_pattern,
            "shards": self.shard_identities,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "world_size": self.world_size,
            "rank": self.rank,
            "epoch": 0,
            "shuffle": False,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"incompatible data-loader {key}: checkpoint={state.get(key)!r}, current={value!r}")
        shard_index = int(state["shard_index"])
        token_offset = int(state["token_offset"])
        if not 0 <= shard_index < len(self.files):
            raise ValueError(f"invalid checkpoint shard_index: {shard_index}")
        if token_offset < 0 or token_offset % self.batch_size != 0:
            raise ValueError(f"invalid checkpoint token_offset: {token_offset}")
        self.shard_index = shard_index
        self.pos = token_offset
        self.tokens = _load_data_shard(
            self.files[self.shard_index], pin_memory=self.device.type == "cuda")
        if self.pos > len(self.tokens):
            raise ValueError(
                f"checkpoint token_offset {self.pos} exceeds shard length {len(self.tokens)}")


def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024,
                               data_root: str | Path | None = None):
    return DistributedDataLoader(filename_pattern, batch_size, seq_len, data_root=data_root)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def collect_environment_metadata():
    def git_output(*args):
        result = subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parents[2],
            text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
    }
    if torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
        metadata["cuda_capability"] = list(torch.cuda.get_device_capability())
    return metadata


def atomic_save_checkpoint(payload, checkpoint_dir: str | Path):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_dir / "latest.pt"
    previous = checkpoint_dir / "previous.pt"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=".checkpoint-", suffix=".tmp",
                dir=checkpoint_dir, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        if latest.exists():
            os.replace(latest, previous)
        os.replace(temporary_path, latest)
        directory_fd = os.open(checkpoint_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return latest
    except BaseException:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def validate_checkpoint_config(checkpoint, resolved_config):
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version: {checkpoint.get('format_version')}")
    checkpoint_config = checkpoint.get("resolved_config")
    if checkpoint_config != resolved_config:
        raise ValueError(
            f"checkpoint configuration is incompatible:\n"
            f"checkpoint={checkpoint_config!r}\ncurrent={resolved_config!r}")


def unwrap_model(model):
    """Return the underlying module if a future compile path wraps it."""
    return getattr(model, "_orig_mod", model)


def make_training_checkpoint(model, optimizers, completed_updates, batch_size,
                             resolved_config, train_loader, run_id, trial_idx,
                             training_time, current_segment_time, last_val_step,
                             environment_metadata):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": unwrap_model(model).state_dict(),
        "optimizers": [
            {"name": type(optimizer).__name__, "state": optimizer.state_dict()}
            for optimizer in optimizers
        ],
        "completed_updates": completed_updates,
        "processed_training_tokens": completed_updates * batch_size,
        "resolved_config": resolved_config,
        "data_loader": train_loader.state_dict(),
        "rng": capture_rng_state(),
        "run": {"run_id": str(run_id), "trial_idx": trial_idx},
        "timing": {
            "training_time": training_time,
            "current_segment_time": current_segment_time,
            "last_val_step": last_val_step,
        },
        "environment": environment_metadata,
    }


########################################
#             Architecture             #
########################################

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


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


########################################
#                Setup                 #
########################################

if __name__ == "__main__":
    with open(sys.argv[0]) as f:
        code = f.read() # read the code of this file ASAP, for logging

    # torchrun sets these env variables
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    # this code can be run equivalently with 1, 2, 4, or 8 gpus.
    assert 8 % dist.get_world_size() == 0

    num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "")
    checkpoint_interval = int(os.environ.get("CHECKPOINT_INTERVAL", 0))
    resume_checkpoint_path = os.environ.get("RESUME_CHECKPOINT", "")
    stop_after_value = os.environ.get("STOP_AFTER_COMPLETED_UPDATES", "")
    stop_after_updates = int(stop_after_value) if stop_after_value else None
    checkpointing_requested = bool(
        checkpoint_dir or checkpoint_interval or resume_checkpoint_path
        or stop_after_updates is not None)
    if checkpoint_interval < 0:
        raise ValueError("CHECKPOINT_INTERVAL must be nonnegative")
    if checkpointing_requested:
        if dist.get_world_size() != 1:
            raise ValueError("checkpoint/resume currently requires exactly one GPU")
        if num_trials != 1:
            raise ValueError("checkpoint/resume currently requires exactly one trial")
        if not checkpoint_dir:
            raise ValueError(
                "CHECKPOINT_DIR is required when checkpointing, resuming, or stopping early")
    if stop_after_updates is not None and stop_after_updates <= 0:
        raise ValueError("STOP_AFTER_COMPLETED_UPDATES must be positive")

    resume_checkpoint = None
    if resume_checkpoint_path:
        resume_checkpoint = torch.load(
            resume_checkpoint_path, map_location="cpu", weights_only=False)
        if resume_checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format version: "
                f"{resume_checkpoint.get('format_version')}")
        if resume_checkpoint.get("run", {}).get("trial_idx") != 0:
            raise ValueError("resume currently supports only trial_idx=0")

    seed_value = os.environ.get("SEED_OVERRIDE", "")
    seed = int(seed_value) if seed_value else None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # logging setup
    if dist.get_rank() == 0:
        os.makedirs("logs", exist_ok=True)
        run_id = (resume_checkpoint["run"]["run_id"]
                  if resume_checkpoint is not None else str(uuid.uuid4()))
        logfile = f"logs/{run_id}.txt"
        print(logfile)
    def print0(s, console=False, log=True):
        if dist.get_rank() == 0:
            if console:
                print(s)
            if log:
                with open(logfile, "a") as f:
                    print(s, file=f)

    # we begin by logging this file itself
    print0(code)
    print0("="*100)
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
           + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
    print0("="*100)
    if checkpointing_requested:
        print0(
            f"checkpointing: directory={checkpoint_dir} interval={checkpoint_interval} "
            f"resume={resume_checkpoint_path or 'none'} "
            f"stop_after={stop_after_updates if stop_after_updates is not None else 'none'}",
            console=True,
        )

    val_tokens = 20 * 524288
    batch_size = 8 * 64 * 1024
    data_root = os.environ.get("DATA_ROOT", str(Path.cwd()))
    # (MBS_OVERRIDE: opt-in override for smoke tests; unset -> unchanged default of 64.
    # Global batch is preserved regardless of mbs: the gradient-accumulation loop
    # below runs len(inputs)//mbs microbatches per step, so a smaller mbs means
    # more microbatches accumulated into the same fixed batch_size, not a smaller
    # effective step.)
    mbs = int(os.environ.get("MBS_OVERRIDE", 64))
    val_loader = distributed_data_generator(
        "data/fineweb10B/fineweb_val_*.bin", val_tokens, data_root=data_root)
    val_inputs, val_targets = next(val_loader)

    # MLP architecture: "dense" is the original single MLP; "moe" is a SparseMoE
    # of num_experts experts, routing each token to its top_k experts.
    # (*_OVERRIDE: opt-in overrides for smoke tests; all unset -> unchanged defaults:
    # mlp_type="dense", mlp_ratio=4, num_experts=1, top_k=1, moe_backend="loop")
    mlp_type = os.environ.get("MLP_TYPE_OVERRIDE", "dense")   # "dense" or "moe"
    mlp_ratio = float(os.environ.get("MLP_RATIO_OVERRIDE", 4))
    num_experts = int(os.environ.get("NUM_EXPERTS_OVERRIDE", 1))
    top_k = int(os.environ.get("TOP_K_OVERRIDE", 1))
    normalize_topk = True
    moe_backend = os.environ.get("MOE_BACKEND_OVERRIDE", "loop")   # "loop" or "grouped_gemm"

    # TensorBoard logging is opt-in via a shared root; rank 0 only.
    tb_root = os.environ.get("TB_ROOT", "")
    tb_system = os.environ.get("TB_SYSTEM", "unknown")
    tensorboard_log = bool(tb_root)

    nsys_profile_value = os.environ.get("NSYS_PROFILE", "0")
    if nsys_profile_value not in ("0", "1"):
        raise ValueError(f"NSYS_PROFILE must be 0 or 1, got {nsys_profile_value!r}")
    nsys_profile = nsys_profile_value == "1"
    nsys_warmup_steps = 10
    nsys_active_steps = 2
    if nsys_profile:
        nsys_warmup_steps = int(os.environ.get("NSYS_WARMUP_STEPS", nsys_warmup_steps))
        nsys_active_steps = int(os.environ.get("NSYS_ACTIVE_STEPS", nsys_active_steps))
        if dist.get_world_size() != 1:
            raise ValueError("NSYS_PROFILE=1 currently requires exactly one GPU")
        if num_trials != 1:
            raise ValueError("NSYS_PROFILE=1 currently requires exactly one trial")
        if nsys_warmup_steps < 0:
            raise ValueError("NSYS_WARMUP_STEPS must be nonnegative")
        if nsys_active_steps <= 0:
            raise ValueError("NSYS_ACTIVE_STEPS must be positive")
        print0(
            f"Nsight Systems profiling enabled: warmup_updates={nsys_warmup_steps} "
            f"captured_updates={nsys_warmup_steps + 1}-"
            f"{nsys_warmup_steps + nsys_active_steps}",
            console=True,
        )
    if nsys_profile and checkpointing_requested:
        raise ValueError(
            "combining NSYS_PROFILE with checkpoint/resume is not yet supported")
    model_dim = 768
    model = GPT(vocab_size=50304, num_layers=12, model_dim=model_dim, mlp_type=mlp_type,
                num_experts=num_experts, top_k=top_k, normalize_topk=normalize_topk,
                moe_backend=moe_backend, mlp_ratio=mlp_ratio)
    local_sequences = batch_size // dist.get_world_size() // 1024
    assert local_sequences % mbs == 0
    accumulation_count = local_sequences // mbs
    print0(
        f"configuration: model_dim={model.model_dim} mlp_ratio={float(model.mlp_ratio):g} "
        f"hidden_dim={model.hidden_dim} model_type={mlp_type} moe_backend={moe_backend} "
        f"E={num_experts} k={top_k} microbatch={mbs} "
        f"global_batch={batch_size} accumulation_count={accumulation_count} "
        f"trial_count={num_trials}",
        console=True,
    )
    model.cuda()

    # Compilation strategy, constructed once, outside the trial/step loops:
    # - dense: unchanged whole-model compile (existing baseline, untouched).
    # - moe: the expert dispatch loop forces a graph break inside GPT.forward's
    #   own block loop, which skips compiling the WHOLE frame (including the
    #   much larger head/loss tail) for either mlp_type. Instead, run the
    #   block loop eagerly (dispatch/routing unchanged) and compile only the
    #   head/loss tail, in isolation, with fullgraph=True so any future
    #   internal break in that region fails loudly instead of silently
    #   falling back. Verified on an A100 (eval peak 6.944 GiB vs an OOM).
    if mlp_type == "dense":
        model.compile(dynamic=False)
        def run_forward(inputs, targets):
            return model(inputs, targets)
    elif mlp_type == "moe":
        compiled_head_loss = torch.compile(make_head_loss(model), fullgraph=True, dynamic=False)
        def run_forward(inputs, targets):
            x = eager_prefix(model, inputs)
            return compiled_head_loss(x, targets)
    else:
        raise ValueError(f"unknown mlp_type: {mlp_type!r}")

    environment_metadata = (
        collect_environment_metadata() if checkpointing_requested else None)
    stopped_early = False
    for trial_idx in range(num_trials):


        ########################################
        #       Init & Optim Hyperparams       #
        ########################################

        # we want to minimize this while still reaching 3.28 val loss
        # (TRAIN_STEPS_OVERRIDE: opt-in override for short smoke tests; unset -> unchanged default)
        train_steps = int(os.environ.get("TRAIN_STEPS_OVERRIDE", 3250))
        if train_steps <= 0:
            raise ValueError("TRAIN_STEPS_OVERRIDE must be positive")
        if stop_after_updates is not None and stop_after_updates > train_steps:
            raise ValueError(
                "STOP_AFTER_COMPLETED_UPDATES cannot exceed the total training steps")
        if nsys_profile and train_steps < nsys_warmup_steps + nsys_active_steps:
            raise ValueError(
                f"NSYS_PROFILE capture ends after update "
                f"{nsys_warmup_steps + nsys_active_steps}, but TRAIN_STEPS_OVERRIDE "
                f"requests only {train_steps} updates")

        # initialize model parameters
        for name, p in model.named_parameters():
            w = p.data
            if name.endswith("weight"):
                if "proj" in name:
                    w.zero_()
                elif "embed" in name:
                    w.normal_()  # default torch init
                else:
                    w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
            elif name.endswith("bias"):
                w.zero_()
            elif name.endswith("gains"):
                w.normal_(mean=1, std=0)
            else:
                raise Exception(f"Uninitialized parameter: {name}")

        # create the optimizer(s)
        optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.7),
                            dict(params=[model.proj.weight], lr=0.004),
                            dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)],
                           betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
        optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                          lr=0.025, weight_decay=0.05)
        optimizers = [optimizer1, optimizer2]
        assert set(p for opt in optimizers for group in opt.param_groups
                   for p in group["params"]) == set(model.parameters())
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        # learning rate schedule: stable then decay
        def set_hparams(step, cooldown_frac=0.7):
            progress = step / train_steps
            assert 0 <= progress < 1
            if progress < 1 - cooldown_frac:
                eta = 1.0
            else:
                eta = (1 - progress) / cooldown_frac
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * eta

        resolved_config = {
            "model": {
                "vocab_size": 50304,
                "num_layers": 12,
                "model_dim": model.model_dim,
                "mlp_type": mlp_type,
                "mlp_ratio": float(model.mlp_ratio),
                "hidden_dim": model.hidden_dim,
                "num_experts": num_experts,
                "top_k": top_k,
                "normalize_topk": normalize_topk,
                "moe_backend": moe_backend,
            },
            "training": {
                "sequence_length": 1024,
                "global_batch_tokens": batch_size,
                "microbatch_sequences": mbs,
                "accumulation_count": accumulation_count,
                "validation_tokens": val_tokens,
                "total_steps": train_steps,
                "cooldown_fraction": 0.7,
                "training_shard_pattern": "data/fineweb10B/fineweb_train_*.bin",
                "validation_shard_pattern": "data/fineweb10B/fineweb_val_*.bin",
            },
            "optimizers": {
                "adamw": {
                    "group_lrs": [0.7, 0.004, 0.015],
                    "betas": [0.8, 0.95],
                    "eps": 1e-10,
                    "weight_decay": 0.001,
                    "fused": True,
                },
                "muon": {"lr": 0.025, "weight_decay": 0.05, "mu": 0.95},
            },
            "datasets": {
                "validation_shards": val_loader.shard_identities,
            },
            "seed_override": seed,
        }


        ########################################
        #        Training and Validation       #
        ########################################

        train_loader = distributed_data_generator(
            "data/fineweb10B/fineweb_train_*.bin", batch_size, data_root=data_root)

        completed_updates = 0
        training_time = 0.0
        current_segment_time = 0.0
        last_val_step = 0
        if resume_checkpoint is not None:
            validate_checkpoint_config(resume_checkpoint, resolved_config)
            completed_updates = int(resume_checkpoint["completed_updates"])
            if not 0 <= completed_updates <= train_steps:
                raise ValueError(
                    f"invalid completed update count in checkpoint: {completed_updates}")
            if resume_checkpoint.get("processed_training_tokens") != completed_updates * batch_size:
                raise ValueError("checkpoint processed-training-token count is inconsistent")
            if (stop_after_updates is not None
                    and stop_after_updates <= completed_updates):
                raise ValueError(
                    "STOP_AFTER_COMPLETED_UPDATES must be greater than the restored update count")
            unwrap_model(model).load_state_dict(resume_checkpoint["model"])
            optimizer_states = resume_checkpoint.get("optimizers", [])
            if len(optimizer_states) != len(optimizers):
                raise ValueError("checkpoint optimizer count is incompatible")
            for optimizer, saved_optimizer in zip(optimizers, optimizer_states):
                expected_name = type(optimizer).__name__
                if saved_optimizer.get("name") != expected_name:
                    raise ValueError(
                        f"checkpoint optimizer is incompatible: expected {expected_name}, "
                        f"got {saved_optimizer.get('name')!r}")
                optimizer.load_state_dict(saved_optimizer["state"])
            train_loader.load_state_dict(resume_checkpoint["data_loader"])
            timing = resume_checkpoint["timing"]
            training_time = float(timing["training_time"])
            current_segment_time = float(timing["current_segment_time"])
            last_val_step = int(timing["last_val_step"])
            print0(
                f"Resuming {run_id} from {resume_checkpoint_path}: "
                f"completed_updates={completed_updates} "
                f"processed_training_tokens={resume_checkpoint['processed_training_tokens']}",
                console=True,
            )

        # tensorboard writer: rank 0 only, one run directory per trial, disabled by default
        writer = None
        if tensorboard_log and dist.get_rank() == 0:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(
                tb_root, "modded-nanogpt-moe", tb_system, str(run_id), f"trial_{trial_idx}")
            print0(f"TensorBoard event directory: {tb_dir}", console=True)
            writer_kwargs = {"log_dir": tb_dir}
            if resume_checkpoint is not None:
                # Hide any stale events at or after the restored update. This keeps a
                # reused run directory coherent if work progressed past the checkpoint.
                writer_kwargs["purge_step"] = completed_updates
            writer = SummaryWriter(**writer_kwargs)
            if resume_checkpoint is not None and completed_updates > 0:
                restored_elapsed = training_time + current_segment_time
                writer.add_scalar(
                    "perf/approx_training_time_s", restored_elapsed, completed_updates)
                writer.add_scalar(
                    "perf/step_avg_ms",
                    1000 * restored_elapsed / completed_updates,
                    completed_updates,
                )
                writer.flush()

        for p in model.parameters():
            dist.broadcast(p.detach(), 0)
        if resume_checkpoint is not None:
            # Model/optimizer construction and writer setup may consume randomness.
            # Restore last so the next training update sees the saved RNG streams.
            restore_rng_state(resume_checkpoint["rng"])
        # start the clock
        dist.barrier()
        t0 = time.perf_counter() - current_segment_time
        nsys_capture_active = False
        for step in range(completed_updates, train_steps + 1):

            # --------------- VALIDATION SECTION -----------------
            val_step_freq = 125 if step / train_steps < 0.9 else 25
            if step == train_steps or step % val_step_freq == 0:
                # stop the clock
                dist.barrier()
                time_since_last_val = time.perf_counter() - t0
                step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
                last_val_step = step
                training_time += time_since_last_val
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    assert len(val_inputs) % mbs == 0
                    for i in range(len(val_inputs) // mbs):
                        val_loss += run_forward(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                val_loss /= val_tokens
                print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                       + f" step_avg:{1000*step_avg:.2f}ms"
                       + f" mem_alloc:{torch.cuda.memory_allocated()/2**30:.3f}GiB"
                       + f" mem_alloc_peak:{torch.cuda.max_memory_allocated()/2**30:.3f}GiB"
                       + f" mem_reserved:{torch.cuda.memory_reserved()/2**30:.3f}GiB"
                       + f" mem_reserved_peak:{torch.cuda.max_memory_reserved()/2**30:.3f}GiB",
                       console=True)
                if writer is not None:
                    writer.add_scalar("eval/val_loss", float(val_loss), step)
                    writer.add_scalar("perf/step_avg_ms", 1000 * step_avg, step)
                    writer.flush()
                model.train()
                # start the clock again
                dist.barrier()
                t0 = time.perf_counter()

            if step == train_steps:
                break

            # --------------- TRAINING SECTION -----------------
            if nsys_profile and step == nsys_warmup_steps:
                torch.cuda.synchronize()
                print0(f"Nsight Systems capture starting before update {step + 1}", console=True)
                torch.cuda.profiler.start()
                nsys_capture_active = True

            with nsys_range(nsys_capture_active, f"optimizer_step.update_{step + 1}"):
                with nsys_range(nsys_capture_active, "data_preparation"):
                    inputs, targets = next(train_loader)
                # accumulate across microbatches in case we are running with fewer than 8 gpus
                assert len(inputs) % mbs == 0
                for i in range(len(inputs) // mbs):
                    with nsys_range(nsys_capture_active, f"forward.microbatch_{i}"):
                        loss = run_forward(
                            inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
                    with nsys_range(nsys_capture_active, f"backward.microbatch_{i}"):
                        loss.backward()
                    del loss
                for name, p in model.named_parameters():
                    assert p.grad is not None, name
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                # set optimization hyperparameters and take a step
                set_hparams(step)
                if writer is not None:
                    for opt_idx, opt in enumerate(optimizers):
                        for grp_idx, group in enumerate(opt.param_groups):
                            writer.add_scalar(
                                f"optim/lr_opt{opt_idx}_group{grp_idx}", group["lr"], step)
                for opt_idx, opt in enumerate(optimizers):
                    optimizer_name = type(opt).__name__
                    with nsys_range(
                            nsys_capture_active,
                            f"optimizer_update.index_{opt_idx}.{optimizer_name}"):
                        opt.step()
                model.zero_grad(set_to_none=True)

            if (nsys_capture_active
                    and step + 1 == nsys_warmup_steps + nsys_active_steps):
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                nsys_capture_active = False
                print0(f"Nsight Systems capture ended after update {step + 1}", console=True)
            approx_training_time = training_time + (time.perf_counter() - t0)
            print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
                   + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms"
                   + f" mem_alloc:{torch.cuda.memory_allocated()/2**30:.3f}GiB"
                   + f" mem_alloc_peak:{torch.cuda.max_memory_allocated()/2**30:.3f}GiB"
                   + f" mem_reserved:{torch.cuda.memory_reserved()/2**30:.3f}GiB"
                   + f" mem_reserved_peak:{torch.cuda.max_memory_reserved()/2**30:.3f}GiB",
                   console=True, log=False)
            if writer is not None:
                writer.add_scalar("perf/approx_training_time_s", approx_training_time, step + 1)
                writer.add_scalar("perf/step_avg_ms", 1000 * approx_training_time / (step + 1), step + 1)
                writer.flush()

            completed_updates = step + 1
            save_due = bool(checkpoint_dir) and (
                completed_updates == train_steps
                or (checkpoint_interval > 0
                    and completed_updates % checkpoint_interval == 0)
                or completed_updates == stop_after_updates)
            if save_due:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        raise RuntimeError(
                            f"refusing to checkpoint before gradients are cleared: {name}")
                checkpoint_started = time.perf_counter()
                checkpoint = make_training_checkpoint(
                    model=model,
                    optimizers=optimizers,
                    completed_updates=completed_updates,
                    batch_size=batch_size,
                    resolved_config=resolved_config,
                    train_loader=train_loader,
                    run_id=run_id,
                    trial_idx=trial_idx,
                    training_time=training_time,
                    current_segment_time=checkpoint_started - t0,
                    last_val_step=last_val_step,
                    environment_metadata=environment_metadata,
                )
                saved_path = atomic_save_checkpoint(checkpoint, checkpoint_dir)
                checkpoint_duration = time.perf_counter() - checkpoint_started
                # Checkpoint I/O is bookkeeping rather than training time.
                t0 += checkpoint_duration
                print0(
                    f"Saved checkpoint after update {completed_updates}: {saved_path}",
                    console=True,
                )

            if completed_updates == stop_after_updates:
                if nsys_capture_active:
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                    nsys_capture_active = False
                    print0(
                        f"Nsight Systems capture ended early after update {completed_updates}",
                        console=True,
                    )
                print0(
                    f"Stopped cleanly after requested update {completed_updates}; "
                    f"schedule horizon remains {train_steps}",
                    console=True,
                )
                stopped_early = True
                break

        if writer is not None:
            writer.close()
        if stopped_early:
            break

    dist.destroy_process_group()
