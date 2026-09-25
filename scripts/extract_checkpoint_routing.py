#!/usr/bin/env python3
"""Extract reusable routing and expert-geometry data from MoE checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from modded_nanogpt_moe.checkpoint import CHECKPOINT_FORMAT_VERSION
from modded_nanogpt_moe.model import eager_prefix
from scripts.extract_grad_em_diagnostics import (
    _torch_load,
    atomic_json_save,
    atomic_torch_save,
    build_replay_model,
    capture_moe_intermediates,
    checkpoint_completed_updates,
    checkpoint_identity,
    discover_checkpoints,
    load_or_create_evaluation_batch,
    model_config_from_checkpoint,
    parse_steps,
    replay_fixed_batch_chunks,
    selected_dot_products,
)


SCHEMA_VERSION = 1
TOKEN_METADATA_SCHEMA_VERSION = 1


def parse_layers(value):
    try:
        layers = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--layers must be comma-separated integers") from error
    if not layers or any(layer < 0 for layer in layers) or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("--layers must contain unique nonnegative integers")
    return layers


def expert_gram(expert_outputs, order, top_k):
    """Return H H^T / d in token/slot order without retaining H."""
    outputs = expert_outputs.detach()
    if order is not None:
        token_order = torch.empty_like(outputs)
        token_order[order.detach()] = outputs
        outputs = token_order.view(-1, top_k, outputs.shape[-1])
    elif outputs.ndim != 3 or outputs.shape[1] != top_k:
        raise ValueError("unpacked expert outputs must have [T,K,d] shape")
    dimension = outputs.shape[-1]
    outputs = outputs.float()
    return torch.bmm(outputs, outputs.transpose(1, 2)).div_(dimension)


def router_input_statistics(router_input):
    """Return norms, sum, and unnormalized X^T X using FP32 reductions."""
    values = router_input.detach().reshape(-1, router_input.shape[-1]).float()
    return (
        torch.linalg.vector_norm(values, dim=-1),
        values.sum(dim=0),
        values.transpose(0, 1).matmul(values),
    )


@contextmanager
def capture_routing_intermediates(model, layer_indices):
    """Install the existing offline observer for only the requested layers."""
    with capture_moe_intermediates(
            model, layer_indices=layer_indices,
            include_routing_state=True) as captured:
        yield captured


def finish_routing_layers(loss, captures, top_k):
    """Reduce every requested layer from one common task-loss backward traversal."""
    layer_indices = sorted(captures)
    if not layer_indices:
        raise RuntimeError("routing replay captured no MoE layers")
    layers = {}
    handles = []

    def finish_layer(layer_index, incoming_gradient):
        if layer_index in layers:
            return incoming_gradient
        capture = captures[layer_index]
        gradient = incoming_gradient.detach().reshape(
            -1, incoming_gradient.shape[-1]).float()
        output_grad_norm = torch.linalg.vector_norm(
            gradient, dim=-1).float().cpu()
        del gradient
        sensitivity = selected_dot_products(
            incoming_gradient, capture["expert_outputs"], capture["order"], top_k
        ).float().cpu()
        gram = expert_gram(
            capture["expert_outputs"], capture["order"], top_k).float().cpu()
        input_norm, input_sum, input_cross = router_input_statistics(
            capture["router_input"])
        layers[layer_index] = {
            "logits": capture["logits"].float().cpu(),
            "topk_experts": capture["topk_experts"].to(torch.int32).cpu(),
            "topk_weights": capture["topk_weights"].float().cpu(),
            "sensitivity": sensitivity,
            "output_grad_norm": output_grad_norm,
            "expert_gram": gram,
            "router_input_norm": input_norm.float().cpu(),
            "_router_input_sum": input_sum.float().cpu(),
            "_router_input_cross": input_cross.float().cpu(),
        }
        capture.clear()
        return incoming_gradient

    for layer_index in layer_indices:
        output = captures[layer_index]["combined_output"]
        handles.append(output.register_hook(
            lambda gradient, index=layer_index: finish_layer(index, gradient)))
    first_output = captures[layer_indices[0]]["combined_output"]
    try:
        (first_gradient,) = torch.autograd.grad(loss, first_output)
        finish_layer(layer_indices[0], first_gradient)
    finally:
        for handle in handles:
            handle.remove()
    if set(layers) != set(layer_indices):
        missing = sorted(set(layer_indices) - set(layers))
        raise RuntimeError(f"missing incoming gradients for MoE layers: {missing}")
    return layers


def replay_routing(model, inputs, targets, layer_indices,
                   analysis_microbatch_sequences=None, compile_head=True):
    """Replay one fixed batch once, optionally in contiguous sequence chunks."""
    def process_chunk(model, head_loss, chunk_inputs, chunk_targets):
        with capture_routing_intermediates(
                model, layer_indices) as (captures, observed_layer_count):
            prefix_output = eager_prefix(model, chunk_inputs)
            loss, token_loss = head_loss(prefix_output, chunk_targets)
            if (len(captures) != observed_layer_count
                    or observed_layer_count != len(layer_indices)):
                raise RuntimeError("not every requested MoE layer produced a capture")
            chunk_layers = finish_routing_layers(
                loss, captures, model.blocks[layer_indices[0]].mlp.top_k)
        result = (
            loss.detach().double().cpu(), token_loss.detach().float().cpu(),
            chunk_layers)
        del captures, prefix_output, loss, token_loss
        return result

    chunk_results = replay_fixed_batch_chunks(
        model, inputs, targets, process_chunk,
        analysis_microbatch_sequences=analysis_microbatch_sequences,
        compile_head=compile_head, return_token_loss=True)
    layer_chunks = {}
    token_loss_chunks = []
    total_loss = torch.zeros((), dtype=torch.float64)
    for chunk_loss, token_loss, chunk_layers in chunk_results:
        token_loss_chunks.append(token_loss)
        total_loss += chunk_loss
        for layer_index, layer in chunk_layers.items():
            destination = layer_chunks.setdefault(layer_index, {})
            for name, tensor in layer.items():
                destination.setdefault(name, []).append(tensor)

    token_count = inputs.numel()
    layers = {}
    for layer_index, values in layer_chunks.items():
        layer = {}
        for name, chunks in values.items():
            if name == "_router_input_sum":
                layer["router_input_mean"] = torch.stack(chunks).sum(0).div(token_count)
            elif name == "_router_input_cross":
                layer["router_input_second_moment"] = (
                    torch.stack(chunks).sum(0).div(token_count))
            else:
                layer[name] = torch.cat(chunks, dim=0)
        layers[layer_index] = layer
    return total_loss.float(), torch.cat(token_loss_chunks), layers


def evaluation_token_metadata(eval_batch):
    inputs = eval_batch["inputs"].detach().cpu().contiguous()
    targets = eval_batch["targets"].detach().cpu().contiguous()
    sequences, positions = inputs.shape
    return {
        "schema_version": TOKEN_METADATA_SCHEMA_VERSION,
        "eval_batch_hash": eval_batch["metadata"]["token_hash"],
        "shape": [sequences, positions],
        "flattening": "sequence-major row-major; token=sequence_index*sequence_length+token_position",
        "input_ids": inputs.reshape(-1).to(torch.int32),
        "target_ids": targets.reshape(-1).to(torch.int64),
        "sequence_index": torch.arange(sequences, dtype=torch.int32).repeat_interleave(positions),
        "token_position": torch.arange(positions, dtype=torch.int32).repeat(sequences),
    }


def save_or_validate_token_metadata(path, eval_batch):
    expected = evaluation_token_metadata(eval_batch)
    path = Path(path)
    if path.exists():
        actual = _torch_load(path)
        if (actual.get("schema_version") != TOKEN_METADATA_SCHEMA_VERSION
                or actual.get("eval_batch_hash") != expected["eval_batch_hash"]):
            raise ValueError(f"incompatible fixed-batch token metadata: {path}")
        for name in ("input_ids", "target_ids", "sequence_index", "token_position"):
            if not torch.equal(actual.get(name), expected[name]):
                raise ValueError(f"fixed-batch token metadata differs for {name}: {path}")
        return actual
    atomic_torch_save(expected, path)
    return expected


def add_router_parameters(model, layers):
    for layer_index, layer in layers.items():
        router = model.blocks[layer_index].mlp.router
        layer["router_weight"] = router.weight.detach().float().cpu().clone()
        layer["router_bias"] = (
            None if router.bias is None else router.bias.detach().float().cpu().clone())


def validate_artifact(artifact):
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid routing artifact schema")
    metadata = artifact.get("metadata", {})
    loss = artifact.get("loss")
    if (not isinstance(loss, torch.Tensor) or loss.shape != ()
            or loss.dtype != torch.float32 or not torch.isfinite(loss)):
        raise ValueError("loss must be a finite FP32 scalar")
    token_loss = artifact.get("token_loss")
    if not isinstance(token_loss, torch.Tensor) or token_loss.ndim != 1:
        raise ValueError("token_loss must have [T] shape")
    if token_loss.dtype != torch.float32 or not torch.isfinite(token_loss).all():
        raise ValueError("token_loss must be finite FP32")
    token_count = token_loss.numel()
    experts, top_k, dimension = metadata.get("E"), metadata.get("K"), metadata.get("d")
    expected_layers = metadata.get("layers")
    layers = artifact.get("layers")
    if not isinstance(layers, dict) or list(layers) != expected_layers:
        raise ValueError("artifact layers do not match metadata")
    for name, layer in layers.items():
        expected = {
            "logits": (token_count, experts),
            "topk_experts": (token_count, top_k),
            "topk_weights": (token_count, top_k),
            "sensitivity": (token_count, top_k),
            "output_grad_norm": (token_count,),
            "expert_gram": (token_count, top_k, top_k),
            "router_input_norm": (token_count,),
            "router_input_mean": (dimension,),
            "router_input_second_moment": (dimension, dimension),
            "router_weight": (experts, dimension),
        }
        for field, shape in expected.items():
            tensor = layer.get(field)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"{name}.{field} must have shape {shape}")
            wanted_dtype = torch.int32 if field == "topk_experts" else torch.float32
            if tensor.dtype != wanted_dtype:
                raise ValueError(f"{name}.{field} must have dtype {wanted_dtype}")
            if field != "topk_experts" and not torch.isfinite(tensor).all():
                raise ValueError(f"{name}.{field} contains non-finite values")
        bias = layer.get("router_bias")
        if bias is not None:
            if (bias.shape != (experts,) or bias.dtype != torch.float32
                    or not torch.isfinite(bias).all()):
                raise ValueError(
                    f"{name}.router_bias must be None or finite FP32 [{experts}]")
        ids = layer["topk_experts"]
        if (ids < 0).any() or (ids >= experts).any():
            raise ValueError(f"{name}.topk_experts contains an out-of-range ID")
    return artifact


def extract_checkpoint(checkpoint_path, eval_batch, device, layer_indices,
                       analysis_microbatch_sequences=None, compile_head=True):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = _torch_load(checkpoint_path, mmap=True)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version: {checkpoint.get('format_version')}")
    step = checkpoint.get("completed_updates")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint has invalid completed_updates")
    model_config = model_config_from_checkpoint(checkpoint)
    missing = [index for index in layer_indices
               if index < 0 or index >= model_config["num_layers"]]
    if missing:
        raise ValueError(f"requested layers are outside this model: {missing}")
    training = checkpoint["resolved_config"]["training"]
    expected_shape = [int(training["microbatch_sequences"]), int(training["sequence_length"])]
    if list(eval_batch["inputs"].shape) != expected_shape:
        raise ValueError(
            f"eval batch shape is incompatible with checkpoint: expected {expected_shape}, "
            f"got {list(eval_batch['inputs'].shape)}")
    if (eval_batch["metadata"]["validation_shard_pattern"]
            != training["validation_shard_pattern"]):
        raise ValueError("eval batch validation shard pattern is incompatible")

    run = checkpoint.get("run", {})
    environment = checkpoint.get("environment", {})
    source = checkpoint_identity(checkpoint_path)
    model, trained_backward = build_replay_model(checkpoint, device)
    del checkpoint
    gc.collect()

    loss, token_loss, integer_layers = replay_routing(
        model, eval_batch["inputs"], eval_batch["targets"], layer_indices,
        analysis_microbatch_sequences=analysis_microbatch_sequences,
        compile_head=compile_head)
    add_router_parameters(model, integer_layers)
    named_layers = {f"l{index:02d}": integer_layers[index] for index in layer_indices}
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "step": step,
            "checkpoint": source,
            "run_id": run.get("run_id"),
            "trial_idx": run.get("trial_idx"),
            "git_commit": environment.get("git_commit"),
            "eval_batch_hash": eval_batch["metadata"]["token_hash"],
            "token_alignment": "eval_tokens.pt; sequence-major row-major",
            "layers": list(named_layers),
            "layer_indices": list(layer_indices),
            "E": model_config["num_experts"],
            "K": model_config["top_k"],
            "d": model_config["model_dim"],
            "normalize_topk": model_config["normalize_topk"],
            "moe_backward": trained_backward,
            "replay_moe_backward": "standard",
            "grad_em_eta": model_config.get("grad_em_eta"),
            "sensitivity_loss_scaling": {
                "definition": "d(sum token cross-entropy after logit softcap)/d routing weight",
                "reduction": "sum",
                "token_normalization": "none",
                "accumulation_scaling": "none; one saved microbatch is replayed",
                "eta_scaling": "none",
            },
            "expert_gram_normalization": "H H^T / d",
            "saved_dtypes": {
                "floating_tensors": "torch.float32",
                "topk_experts": "torch.int32",
                "target_ids": "torch.int64",
            },
        },
        "loss": loss,
        "token_loss": token_loss,
        "layers": named_layers,
    }
    validate_artifact(artifact)
    del model, integer_layers
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return artifact


def artifact_matches(path, step, source, eval_batch_hash, layer_indices):
    try:
        artifact = validate_artifact(_torch_load(path))
    except Exception:
        return False
    metadata = artifact["metadata"]
    return (
        metadata.get("step") == step
        and metadata.get("checkpoint") == source
        and metadata.get("eval_batch_hash") == eval_batch_hash
        and metadata.get("layer_indices") == layer_indices
    )


def configure_logging(output_dir):
    logger = logging.getLogger("checkpoint-routing-extractor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    for handler in (
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / "routing_analysis.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--steps", type=parse_steps)
    parser.add_argument("--layers", type=parse_layers, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-batch", type=Path,
        help="shared eval_batch.pt path; defaults to OUTPUT_DIR/eval_batch.pt")
    parser.add_argument(
        "--analysis-microbatch-sequences", type=int,
        help="replay the fixed batch in contiguous sequence chunks")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.steps is not None and args.checkpoint_dir is None:
        parser.error("--steps requires --checkpoint-dir")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for checkpoint extraction")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    checkpoints = (
        discover_checkpoints(args.checkpoint_dir, args.steps)
        if args.checkpoint_dir is not None else
        [(checkpoint_completed_updates(args.checkpoint), args.checkpoint.expanduser().resolve())]
    )
    first_checkpoint = _torch_load(checkpoints[0][1], mmap=True)
    resolved = first_checkpoint.get("resolved_config")
    if not isinstance(resolved, dict) or "training" not in resolved:
        raise ValueError("checkpoint is missing resolved training configuration")
    data_root = os.environ.get("DATA_ROOT", str(Path.cwd()))
    eval_batch = load_or_create_evaluation_batch(
        output_dir, resolved, data_root, eval_batch_path=args.eval_batch)
    eval_batch_path = (
        args.eval_batch.expanduser().resolve()
        if args.eval_batch is not None else output_dir / "eval_batch.pt")
    token_metadata_path = eval_batch_path.with_name("eval_tokens.pt")
    save_or_validate_token_metadata(token_metadata_path, eval_batch)
    del first_checkpoint

    manifest_path = output_dir / "routing_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "eval_batch_path": str(eval_batch_path),
        "eval_token_metadata_path": str(token_metadata_path),
        "eval_batch_hash": eval_batch["metadata"]["token_hash"],
        "layer_indices": args.layers,
        "outputs": [],
    }
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as file:
            previous = json.load(file)
        compatible = (
            previous.get("schema_version") == SCHEMA_VERSION
            and previous.get("eval_batch_hash") == manifest["eval_batch_hash"]
            and previous.get("layer_indices") == args.layers)
        if not compatible and not args.overwrite:
            raise ValueError("existing routing manifest is incompatible; use --overwrite")
        if compatible:
            manifest["outputs"] = previous.get("outputs", [])
    atomic_json_save(manifest, manifest_path)

    for step, checkpoint_path in checkpoints:
        destination = output_dir / f"step_{step:06d}.pt"
        source_identity = checkpoint_identity(checkpoint_path)
        if destination.exists() and not args.overwrite:
            if artifact_matches(
                    destination, step, source_identity,
                    eval_batch["metadata"]["token_hash"], args.layers):
                logger.info("[skip] checkpoint=%s step=%d", checkpoint_path, step)
                continue
            raise FileExistsError(
                f"existing result does not match this extraction; use --overwrite: {destination}")
        artifact = extract_checkpoint(
            checkpoint_path, eval_batch, torch.device("cuda"), args.layers,
            analysis_microbatch_sequences=args.analysis_microbatch_sequences,
            compile_head=True)
        atomic_torch_save(artifact, destination)
        size_mib = destination.stat().st_size / (1024 ** 2)
        logger.info(
            "[done] checkpoint=%s step=%d layers=%s eval_batch_hash=%s "
            "output=%s size=%.2f MiB",
            checkpoint_path, step, ",".join(map(str, args.layers)),
            eval_batch["metadata"]["token_hash"], destination, size_mib)
        manifest["outputs"] = sorted(set(manifest["outputs"] + [destination.name]))
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json_save(manifest, manifest_path)
        del artifact
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
