#!/usr/bin/env python3
"""Extract raw fixed-batch Grad-EM quantities from training checkpoints."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from modded_nanogpt_moe.checkpoint import CHECKPOINT_FORMAT_VERSION
from modded_nanogpt_moe.config import DEFAULT_CONFIG
from modded_nanogpt_moe.data import DistributedDataLoader
from modded_nanogpt_moe.model import GPT, MoE, eager_prefix, make_head_loss


SCHEMA_VERSION = 1
_STEP_NAME = re.compile(r"^step_(\d+)\.pt$")


def _torch_load(path, *, mmap=False):
    kwargs = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def checkpoint_identity(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def checkpoint_completed_updates(path):
    checkpoint = _torch_load(path, mmap=True)
    try:
        completed = checkpoint.get("completed_updates")
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ValueError(f"checkpoint has invalid completed_updates: {path}")
        return completed
    finally:
        del checkpoint


def discover_checkpoints(checkpoint_dir, requested_steps=None):
    """Return one canonical path per update, preferring numbered snapshots."""
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
    candidates = []
    for path in checkpoint_dir.iterdir():
        match = _STEP_NAME.match(path.name)
        if path.is_file() and (match or path.name in ("latest.pt", "previous.pt")):
            completed = checkpoint_completed_updates(path)
            exact_numbered = bool(match and int(match.group(1)) == completed)
            priority = 0 if exact_numbered else (1 if match else (2 if path.name == "latest.pt" else 3))
            candidates.append((completed, priority, path.name, path))
    if not candidates:
        raise FileNotFoundError(f"no training checkpoints found in {checkpoint_dir}")

    canonical = {}
    for completed, priority, name, path in sorted(candidates):
        canonical.setdefault(completed, (priority, name, path))
    if requested_steps is not None:
        missing = sorted(set(requested_steps) - set(canonical))
        if missing:
            raise ValueError(f"requested checkpoint steps are missing: {missing}")
        selected = set(requested_steps)
    else:
        selected = set(canonical)
    return [(step, canonical[step][2]) for step in sorted(selected)]


def parse_steps(value):
    if value is None:
        return None
    try:
        steps = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--steps must be comma-separated integers") from error
    if not steps or any(step < 0 for step in steps) or len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("--steps must contain unique nonnegative integers")
    return steps


def tensor_hash(inputs, targets):
    digest = hashlib.sha256()
    for name, tensor in (("inputs", inputs), ("targets", targets)):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_torch_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=f".{destination.name}-", suffix=".tmp",
                dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def atomic_json_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=f".{destination.name}-",
                suffix=".tmp", dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def evaluation_batch_spec(resolved_config, data_root):
    training = resolved_config["training"]
    sequence_length = int(training["sequence_length"])
    microbatch_sequences = int(training["microbatch_sequences"])
    pattern = training["validation_shard_pattern"]
    root = Path(data_root).expanduser().resolve()
    paths = [str(path.resolve()) for path in sorted(root.glob(pattern))]
    return {
        "shape": [microbatch_sequences, sequence_length],
        "input_dtype": "torch.int32",
        "target_dtype": "torch.int64",
        "validation_shard_pattern": pattern,
        "validation_shard_paths": paths,
        "seed": resolved_config.get("seed_override"),
    }


def create_evaluation_batch(resolved_config, data_root):
    spec = evaluation_batch_spec(resolved_config, data_root)
    sequences, sequence_length = spec["shape"]
    loader = DistributedDataLoader(
        spec["validation_shard_pattern"], sequences * sequence_length,
        seq_len=sequence_length, data_root=data_root, world_size=1, rank=0,
        device="cpu")
    inputs, targets = next(loader)
    metadata = dict(spec)
    metadata["loader_state_after_batch"] = loader.state_dict()
    metadata["hash_algorithm"] = "sha256(inputs+targets with names/dtypes/shapes)"
    metadata["token_hash"] = tensor_hash(inputs, targets)
    return {"schema_version": SCHEMA_VERSION, "inputs": inputs, "targets": targets,
            "metadata": metadata}


def load_or_create_evaluation_batch(output_dir, resolved_config, data_root,
                                    create_fn=create_evaluation_batch,
                                    eval_batch_path=None):
    path = (Path(eval_batch_path).expanduser().resolve()
            if eval_batch_path is not None else Path(output_dir) / "eval_batch.pt")
    expected = evaluation_batch_spec(resolved_config, data_root)
    if path.exists():
        payload = _torch_load(path)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"incompatible eval-batch schema: {path}")
        for key in ("shape", "input_dtype", "target_dtype",
                    "validation_shard_pattern", "validation_shard_paths", "seed"):
            if payload.get("metadata", {}).get(key) != expected[key]:
                raise ValueError(f"existing eval batch has incompatible {key}: {path}")
        inputs, targets = payload.get("inputs"), payload.get("targets")
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise ValueError(f"invalid eval-batch tensors: {path}")
        actual_hash = tensor_hash(inputs, targets)
        if payload["metadata"].get("token_hash") != actual_hash:
            raise ValueError(f"eval-batch hash mismatch: {path}")
        return payload
    payload = create_fn(resolved_config, data_root)
    atomic_torch_save(payload, path)
    return payload


def selected_dot_products(incoming_gradient, expert_outputs, order, top_k):
    """Compute <g[token], h[token,slot]> in FP32 without saving [N,K,D]."""
    gradient = incoming_gradient.detach().reshape(-1, incoming_gradient.shape[-1]).float()
    outputs = expert_outputs.detach()
    if order is None:
        return (gradient[:, None, :] * outputs.float()).sum(dim=-1)
    order = order.detach()
    sorted_tokens = torch.div(order, top_k, rounding_mode="floor")
    sorted_values = (
        gradient.index_select(0, sorted_tokens) * outputs.float()).sum(dim=-1)
    flat_values = torch.empty_like(sorted_values)
    flat_values[order] = sorted_values
    return flat_values.view(gradient.shape[0], top_k)


@contextmanager
def capture_moe_intermediates(model, layer_indices=None, *, include_routing_state=False):
    captures = {}
    all_modules = [(index, block.mlp) for index, block in enumerate(model.blocks)
                   if isinstance(block.mlp, MoE)]
    if layer_indices is None:
        modules = all_modules
    else:
        requested = set(layer_indices)
        available = {index for index, _ in all_modules}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"requested layers are not MoE layers: {missing}")
        modules = [(index, module) for index, module in all_modules
                   if index in requested]

    def make_callback(layer_index):
        def callback(router_logits, topk_experts, topk_weights, expert_outputs,
                     order, combined_output, router_input):
            logits = router_logits.detach().float()
            captures[layer_index] = {
                "selected_logits": logits.gather(1, topk_experts.detach()),
                "router_logsumexp": torch.logsumexp(logits, dim=-1),
                "topk_experts": topk_experts.detach(),
                "expert_outputs": expert_outputs.detach(),
                "order": None if order is None else order.detach(),
                "combined_output": combined_output,
            }
            if include_routing_state:
                captures[layer_index].update({
                    "logits": logits,
                    "topk_weights": topk_weights.detach(),
                    "router_input": router_input.detach(),
                })
        return callback

    try:
        for layer_index, module in modules:
            if module._grad_em_diagnostics is not None:
                raise RuntimeError(f"MoE layer {layer_index} already has an offline observer")
            module._grad_em_diagnostics = make_callback(layer_index)
        yield captures, len(modules)
    finally:
        for _, module in modules:
            module._grad_em_diagnostics = None


def finish_captured_layers(loss, captures, top_k):
    """Materialize each layer when its incoming combine gradient is ready.

    Differentiating only to the first MoE output traverses every downstream
    layer without allocating parameter gradients. Hooks see each layer's
    incoming gradient before its combine backward and immediately move the
    compact result to CPU.
    """
    layer_indices = sorted(captures)
    if not layer_indices:
        raise RuntimeError("diagnostic replay captured no MoE layers")
    layers = {}
    handles = []

    def finish_layer(layer_index, incoming_gradient):
        if layer_index in layers:
            return incoming_gradient
        capture = captures[layer_index]
        values = selected_dot_products(
            incoming_gradient, capture["expert_outputs"], capture["order"], top_k)
        layers[layer_index] = {
            "selected_logits": capture["selected_logits"].float().cpu(),
            "v": values.float().cpu(),
            "topk_experts": capture["topk_experts"].to(torch.int32).cpu(),
            "router_logsumexp": capture["router_logsumexp"].float().cpu(),
        }
        # The autograd node retains anything still required by combine
        # backward. Drop the observer's extra GPU references immediately.
        capture.clear()
        return incoming_gradient

    for layer_index in layer_indices:
        output = captures[layer_index]["combined_output"]
        handles.append(output.register_hook(
            lambda gradient, index=layer_index: finish_layer(index, gradient)))
    first_output = captures[layer_indices[0]]["combined_output"]
    try:
        (first_gradient,) = torch.autograd.grad(loss, first_output)
        # Hooks normally include an input requested from autograd.grad. Keep an
        # explicit fallback so this behavior is not an undocumented dependency.
        finish_layer(layer_indices[0], first_gradient)
    finally:
        for handle in handles:
            handle.remove()
    if set(layers) != set(layer_indices):
        missing = sorted(set(layer_indices) - set(layers))
        raise RuntimeError(f"missing incoming gradients for MoE layers: {missing}")
    return layers


def replay_fixed_batch_chunks(model, inputs, targets, process_chunk,
                              analysis_microbatch_sequences=None,
                              compile_head=True, return_token_loss=False):
    """Shared fixed-batch replay driver for offline checkpoint extractors."""
    if inputs.ndim != 2 or targets.shape != inputs.shape:
        raise ValueError("diagnostic inputs and targets must have equal [B,T] shape")
    total_sequences = inputs.shape[0]
    chunk_sequences = analysis_microbatch_sequences or total_sequences
    if (isinstance(chunk_sequences, bool) or not isinstance(chunk_sequences, int)
            or not 1 <= chunk_sequences <= total_sequences):
        raise ValueError(
            "analysis_microbatch_sequences must be between 1 and the saved batch size")

    head_loss = make_head_loss(model, return_token_loss=return_token_loss)
    if compile_head:
        # This is the exact production MoE head/loss path. Compilation fuses
        # the full-vocabulary float/softcap/CE work that OOMs in eager mode.
        head_loss = torch.compile(head_loss, fullgraph=True, dynamic=False)
    device = next(model.parameters()).device
    results = []
    for start in range(0, total_sequences, chunk_sequences):
        stop = min(start + chunk_sequences, total_sequences)
        chunk_inputs = inputs[start:stop].to(device, non_blocking=True)
        chunk_targets = targets[start:stop].to(device, non_blocking=True)
        results.append(process_chunk(model, head_loss, chunk_inputs, chunk_targets))
        del chunk_inputs, chunk_targets
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def replay_diagnostics(model, inputs, targets, analysis_microbatch_sequences=None,
                       compile_head=True):
    """Replay a complete saved batch, optionally in contiguous sequence chunks."""
    def process_chunk(model, head_loss, chunk_inputs, chunk_targets):
        with capture_moe_intermediates(model) as (captures, chunk_expected_layers):
            prefix_output = eager_prefix(model, chunk_inputs)
            loss = head_loss(prefix_output, chunk_targets)
            if len(captures) != chunk_expected_layers:
                raise RuntimeError("not every MoE layer produced an offline capture")
            chunk_layers = finish_captured_layers(
                loss, captures, model.blocks[0].mlp.top_k)
        del captures, prefix_output, loss
        return chunk_layers, chunk_expected_layers

    chunk_results = replay_fixed_batch_chunks(
        model, inputs, targets, process_chunk,
        analysis_microbatch_sequences=analysis_microbatch_sequences,
        compile_head=compile_head)
    chunks = {}
    expected_layers = None
    for chunk_layers, chunk_expected_layers in chunk_results:
        if expected_layers is None:
            expected_layers = chunk_expected_layers
        elif chunk_expected_layers != expected_layers:
            raise RuntimeError("MoE layer count changed between analysis chunks")
        for layer_index, layer in chunk_layers.items():
            destination = chunks.setdefault(layer_index, {
                "selected_logits": [], "v": [], "topk_experts": [],
                "router_logsumexp": [],
            })
            for name, tensor in layer.items():
                destination[name].append(tensor)

    layers = {
        layer_index: {
            name: torch.cat(tensors, dim=0)
            for name, tensors in values.items()
        }
        for layer_index, values in chunks.items()
    }
    return layers, expected_layers


def model_config_from_checkpoint(checkpoint):
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, dict) or not isinstance(resolved.get("model"), dict):
        raise ValueError("checkpoint is missing resolved model configuration")
    config = copy.deepcopy(DEFAULT_CONFIG["model"])
    config.update(resolved["model"])
    # These derived fields are not GPT constructor arguments.
    config.pop("hidden_dim", None)
    if config["mlp_type"] != "moe":
        raise ValueError("Grad-EM diagnostics require an MoE checkpoint")
    return config


def build_replay_model(checkpoint, device):
    config = model_config_from_checkpoint(checkpoint)
    trained_backward = config.get("moe_backward", "standard")
    replay_config = dict(config)
    # Standard and Grad-EM have the identical forward. Standard replay makes g
    # the true loss derivative and captures it before every combine backward.
    replay_config["moe_backward"] = "standard"
    model = GPT(**replay_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, trained_backward


def validate_result(result, expected_layers, num_experts, top_k):
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid result schema")
    layers = result.get("layers")
    if not isinstance(layers, dict) or len(layers) != expected_layers:
        raise ValueError("unexpected number of extracted MoE layers")
    for layer_index, layer in layers.items():
        selected_logits = layer["selected_logits"]
        values = layer["v"]
        topk_experts = layer["topk_experts"]
        logsumexp = layer["router_logsumexp"]
        if selected_logits.ndim != 2 or selected_logits.shape[1] != top_k:
            raise ValueError(f"layer {layer_index}: invalid selected_logits shape")
        if values.shape != selected_logits.shape or topk_experts.shape != selected_logits.shape:
            raise ValueError(f"layer {layer_index}: incompatible [N,K] tensors")
        if logsumexp.shape != selected_logits.shape[:1]:
            raise ValueError(f"layer {layer_index}: invalid router_logsumexp shape")
        if any(tensor.dtype != torch.float32 for tensor in
               (selected_logits, values, logsumexp)):
            raise ValueError(f"layer {layer_index}: floating outputs must be FP32")
        if topk_experts.dtype != torch.int32:
            raise ValueError(f"layer {layer_index}: topk_experts must be int32")
        if not all(torch.isfinite(tensor).all() for tensor in
                   (selected_logits, values, logsumexp)):
            raise ValueError(f"layer {layer_index}: non-finite diagnostic tensor")
        if ((topk_experts < 0).any() or (topk_experts >= num_experts).any()):
            raise ValueError(f"layer {layer_index}: expert index out of range")
    return result


def extract_checkpoint(checkpoint_path, eval_batch, device,
                       analysis_microbatch_sequences=None):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = _torch_load(checkpoint_path, mmap=True)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version: {checkpoint.get('format_version')}")
    completed_updates = checkpoint.get("completed_updates")
    if (isinstance(completed_updates, bool)
            or not isinstance(completed_updates, int) or completed_updates < 0):
        raise ValueError("checkpoint has invalid completed_updates")
    model_config = model_config_from_checkpoint(checkpoint)
    training_config = checkpoint["resolved_config"]["training"]
    expected_shape = [
        int(training_config["microbatch_sequences"]),
        int(training_config["sequence_length"]),
    ]
    if list(eval_batch["inputs"].shape) != expected_shape:
        raise ValueError(
            f"eval batch shape is incompatible with checkpoint: "
            f"expected {expected_shape}, got {list(eval_batch['inputs'].shape)}")
    if (eval_batch["metadata"]["validation_shard_pattern"]
            != training_config["validation_shard_pattern"]):
        raise ValueError("eval batch validation shard pattern is incompatible")
    model, trained_backward = build_replay_model(checkpoint, device)
    del checkpoint
    gc.collect()

    layers, expected_layers = replay_diagnostics(
        model, eval_batch["inputs"], eval_batch["targets"],
        analysis_microbatch_sequences=analysis_microbatch_sequences,
        compile_head=True)
    if expected_layers != model_config["num_layers"]:
        raise RuntimeError("number of MoE layers does not match model configuration")

    result = {
        "schema_version": SCHEMA_VERSION,
        "completed_updates": completed_updates,
        "checkpoint_path": str(checkpoint_path),
        "source_checkpoint": checkpoint_identity(checkpoint_path),
        "eval_batch_hash": eval_batch["metadata"]["token_hash"],
        "trained_moe_backward": trained_backward,
        "replay_moe_backward": "standard",
        "model": {
            "num_layers": model_config["num_layers"],
            "num_experts": model_config["num_experts"],
            "top_k": model_config["top_k"],
        },
        "layers": layers,
    }
    validate_result(
        result, expected_layers, model_config["num_experts"], model_config["top_k"])
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def result_matches(path, completed_updates, source_identity, eval_batch_hash):
    try:
        result = _torch_load(path)
        model = result["model"]
        validate_result(
            result, model["num_layers"], model["num_experts"], model["top_k"])
    except Exception:
        return False
    return (
        result.get("schema_version") == SCHEMA_VERSION
        and result.get("completed_updates") == completed_updates
        and result.get("source_checkpoint") == source_identity
        and result.get("eval_batch_hash") == eval_batch_hash
    )


def git_state(repository_root):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
        text=True, capture_output=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root, check=True, text=True,
        capture_output=True).stdout.strip())
    return commit, dirty


def make_manifest(checkpoints, checkpoint_directory, checkpoint, eval_batch,
                  data_root, device, analysis_microbatch_sequences):
    commit, dirty = git_state(REPOSITORY_ROOT)
    resolved = checkpoint["resolved_config"]
    training = resolved["training"]
    accumulation_count = training.get("accumulation_count")
    if accumulation_count is None:
        accumulation_count = (
            training["global_batch_tokens"] // training["sequence_length"]
            // training["microbatch_sequences"])
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": commit, "dirty": dirty},
        "source_checkpoint_directory": (
            str(Path(checkpoint_directory).expanduser().resolve())
            if checkpoint_directory is not None else None),
        "checkpoints": [
            {"path": str(path.resolve()), "completed_updates": step}
            for step, path in checkpoints
        ],
        "completed_outputs": [],
        "resolved_model_config": model_config_from_checkpoint(checkpoint),
        "resolved_training_config": training,
        "validation_shard_paths": eval_batch["metadata"]["validation_shard_paths"],
        "data_root": str(Path(data_root).expanduser().resolve()),
        "eval_batch_shape": eval_batch["metadata"]["shape"],
        "eval_batch_hash": eval_batch["metadata"]["token_hash"],
        "analysis_microbatch_sequences": analysis_microbatch_sequences,
        "loss": {
            "definition": "cross_entropy(logits, targets, reduction='sum') after logit softcap",
            "token_normalization": "none",
            "accumulation_count": accumulation_count,
            "accumulation_scaling": "none; microbatch gradients are summed",
            "microbatch_loss_multiplier": 1.0,
        },
        "model_compute_dtype": "torch.bfloat16 activations with FP32 logits/loss",
        "saved_diagnostic_dtypes": {
            "selected_logits": "torch.float32", "v": "torch.float32",
            "topk_experts": "torch.int32", "router_logsumexp": "torch.float32",
        },
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def configure_logging(output_dir):
    logger = logging.getLogger("grad-em-extractor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(Path(output_dir) / "analysis.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=parse_steps)
    parser.add_argument(
        "--analysis-microbatch-sequences", type=int,
        help="replay the saved batch in contiguous sequence chunks")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.steps is not None and args.checkpoint_dir is None:
        parser.error("--steps requires --checkpoint-dir")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for checkpoint extraction")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    if args.checkpoint_dir is not None:
        checkpoints = discover_checkpoints(args.checkpoint_dir, args.steps)
    else:
        path = args.checkpoint.expanduser().resolve()
        checkpoints = [(checkpoint_completed_updates(path), path)]

    first_checkpoint = _torch_load(checkpoints[0][1], mmap=True)
    resolved_config = first_checkpoint.get("resolved_config")
    if not isinstance(resolved_config, dict) or "training" not in resolved_config:
        raise ValueError("checkpoint is missing resolved training configuration")
    data_root = os.environ.get("DATA_ROOT", str(Path.cwd()))
    eval_batch = load_or_create_evaluation_batch(
        output_dir, resolved_config, data_root)
    saved_sequences = eval_batch["inputs"].shape[0]
    analysis_microbatch_sequences = (
        args.analysis_microbatch_sequences or saved_sequences)
    if (args.analysis_microbatch_sequences is not None
            and not 1 <= args.analysis_microbatch_sequences <= saved_sequences):
        parser.error(
            "--analysis-microbatch-sequences must be between 1 and "
            f"the saved batch size ({saved_sequences})")
    device = torch.device("cuda")
    manifest = make_manifest(
        checkpoints, args.checkpoint_dir, first_checkpoint, eval_batch,
        data_root, device, analysis_microbatch_sequences)
    del first_checkpoint
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as file:
            previous_manifest = json.load(file)
        if (previous_manifest.get("schema_version") == SCHEMA_VERSION
                and previous_manifest.get("eval_batch_hash")
                == manifest["eval_batch_hash"]):
            manifest["created_at"] = previous_manifest.get(
                "created_at", manifest["created_at"])
            manifest["completed_outputs"] = previous_manifest.get(
                "completed_outputs", [])
        elif not args.overwrite:
            raise ValueError(
                "existing manifest is incompatible; use --overwrite to replace it")
    atomic_json_save(manifest, manifest_path)

    for completed_updates, checkpoint_path in checkpoints:
        destination = output_dir / f"step_{completed_updates:06d}.pt"
        identity = checkpoint_identity(checkpoint_path)
        if destination.exists() and not args.overwrite:
            if result_matches(
                    destination, completed_updates, identity,
                    eval_batch["metadata"]["token_hash"]):
                logger.info("[skip] step %d", completed_updates)
                manifest["completed_outputs"].append(destination.name)
                continue
            raise FileExistsError(
                f"existing result does not match this extraction; use --overwrite: {destination}")
        logger.info("[run ] step %d from %s", completed_updates, checkpoint_path)
        result = extract_checkpoint(
            checkpoint_path, eval_batch, device,
            analysis_microbatch_sequences=analysis_microbatch_sequences)
        if result["completed_updates"] != completed_updates:
            raise ValueError("checkpoint metadata changed during extraction")
        atomic_torch_save(result, destination)
        manifest["completed_outputs"].append(destination.name)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json_save(manifest, manifest_path)
        logger.info("[done] step %d", completed_updates)

    manifest["completed_outputs"] = sorted(set(manifest["completed_outputs"]))
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json_save(manifest, manifest_path)


if __name__ == "__main__":
    main()
