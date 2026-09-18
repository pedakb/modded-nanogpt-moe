"""Compare two completed training checkpoints and their next data batches."""

import argparse
import gc
import hashlib
from pathlib import Path

import numpy as np
import torch

from modded_nanogpt_moe.data import DistributedDataLoader


def _update_digest(digest, value):
    digest.update(type(value).__name__.encode())
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _update_digest(digest, item)
    elif value is None or isinstance(value, (str, int, float, bool, bytes)):
        digest.update(repr(value).encode())
    else:
        raise TypeError(f"unsupported checkpoint value for comparison: {type(value)}")


def _digest(value):
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _load_summary(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    summary = {
        "format_version": checkpoint["format_version"],
        "completed_updates": checkpoint["completed_updates"],
        "processed_training_tokens": checkpoint["processed_training_tokens"],
        "resolved_config": checkpoint["resolved_config"],
        "data_loader": checkpoint["data_loader"],
        "model_digest": _digest(checkpoint["model"]),
        "optimizer_digest": _digest(checkpoint["optimizers"]),
        "rng_digest": _digest(checkpoint["rng"]),
        "learning_rates": [
            [group["lr"] for group in optimizer["state"]["param_groups"]]
            for optimizer in checkpoint["optimizers"]
        ],
    }
    del checkpoint
    gc.collect()
    return summary


def _next_batch(loader_state, data_root):
    loader = DistributedDataLoader(
        loader_state["filename_pattern"],
        loader_state["batch_size"],
        loader_state["seq_len"],
        data_root=data_root,
        world_size=loader_state["world_size"],
        rank=loader_state["rank"],
        device="cpu",
    )
    loader.load_state_dict(loader_state)
    inputs, targets = next(loader)
    result = inputs.clone(), targets.clone()
    del loader
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_a", type=Path)
    parser.add_argument("checkpoint_b", type=Path)
    parser.add_argument("--data-root-a", type=Path, default=Path.cwd())
    parser.add_argument("--data-root-b", type=Path, default=Path.cwd())
    args = parser.parse_args()

    first = _load_summary(args.checkpoint_a)
    second = _load_summary(args.checkpoint_b)
    checks = [
        ("format version", first["format_version"] == second["format_version"]),
        ("completed update counter", first["completed_updates"] == second["completed_updates"]),
        ("processed token counter", first["processed_training_tokens"] == second["processed_training_tokens"]),
        ("resolved configuration", first["resolved_config"] == second["resolved_config"]),
        ("data-loader cursor and shard identities", first["data_loader"] == second["data_loader"]),
        ("learning rates", first["learning_rates"] == second["learning_rates"]),
        ("model state (byte-exact SHA-256)", first["model_digest"] == second["model_digest"]),
        ("optimizer states (byte-exact SHA-256)", first["optimizer_digest"] == second["optimizer_digest"]),
        ("RNG states (byte-exact SHA-256)", first["rng_digest"] == second["rng_digest"]),
    ]

    try:
        batch_a = _next_batch(first["data_loader"], args.data_root_a)
        batch_b = _next_batch(second["data_loader"], args.data_root_b)
        next_batch_matches = all(torch.equal(a, b) for a, b in zip(batch_a, batch_b))
    except Exception as error:
        print(f"[FAIL] next training batch: {error}")
        next_batch_matches = False
    else:
        print(f"[{'PASS' if next_batch_matches else 'FAIL'}] next training batch")

    for label, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
    passed = next_batch_matches and all(result for _, result in checks)
    print(f"CHECKPOINT COMPARISON: {'PASS' if passed else 'FAIL'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
