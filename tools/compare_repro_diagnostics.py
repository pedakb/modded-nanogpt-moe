"""Quantitatively compare opt-in trainer reproducibility diagnostics."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch


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
        digest.update(repr(value).encode())


def _digest(value):
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _flatten_tensors(value, path=""):
    tensors = {}
    if isinstance(value, torch.Tensor):
        tensors[path] = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        tensors[path] = torch.from_numpy(value.copy())
    elif isinstance(value, dict):
        for key, item in value.items():
            tensors.update(_flatten_tensors(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            tensors.update(_flatten_tensors(item, f"{path}[{index}]"))
    return tensors


def _tensor_difference(first, second, chunk_size=1_000_000):
    if first.dtype != second.dtype or first.shape != second.shape:
        return {
            "compatible": False,
            "first": f"{first.dtype}{tuple(first.shape)}",
            "second": f"{second.dtype}{tuple(second.shape)}",
        }
    if torch.equal(first, second):
        return {"compatible": True, "exact": True, "relative_l2": 0.0, "max_abs": 0.0}
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    difference_sq = 0.0
    reference_sq = 0.0
    max_abs = 0.0
    for start in range(0, first_flat.numel(), chunk_size):
        first_chunk = first_flat[start:start + chunk_size].to(torch.float64)
        second_chunk = second_flat[start:start + chunk_size].to(torch.float64)
        difference = first_chunk - second_chunk
        difference_sq += difference.square().sum().item()
        reference_sq += first_chunk.square().sum().item()
        if difference.numel():
            max_abs = max(max_abs, difference.abs().max().item())
    relative_l2 = difference_sq**0.5 / reference_sq**0.5 if reference_sq else (
        0.0 if difference_sq == 0 else float("inf"))
    return {
        "compatible": True,
        "exact": False,
        "relative_l2": relative_l2,
        "max_abs": max_abs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    first = torch.load(args.first, map_location="cpu", weights_only=False)
    second = torch.load(args.second, map_location="cpu", weights_only=False)
    print(f"first:  {args.first} stage={first.get('stage')} updates={first.get('completed_updates')}")
    print(f"second: {args.second} stage={second.get('stage')} updates={second.get('completed_updates')}")

    for section in (
        "model", "optimizers", "gradients", "losses", "inputs", "targets",
        "rng", "data_loader",
    ):
        if section in first or section in second:
            matches = section in first and section in second and _digest(first[section]) == _digest(second[section])
            print(f"[{'MATCH' if matches else 'DIFFER'}] {section}")

    runtime_keys = sorted(set(first.get("runtime", {})) | set(second.get("runtime", {})))
    for key in runtime_keys:
        first_value = first.get("runtime", {}).get(key)
        second_value = second.get("runtime", {}).get(key)
        if first_value != second_value:
            print(f"[CONTROL DIFFERENCE] runtime.{key}: {first_value!r} != {second_value!r}")

    first_tensors = _flatten_tensors(first)
    second_tensors = _flatten_tensors(second)
    tensor_paths = sorted(set(first_tensors) | set(second_tensors))
    differences = []
    missing = []
    for path in tensor_paths:
        if path not in first_tensors or path not in second_tensors:
            missing.append(path)
            continue
        difference = _tensor_difference(first_tensors[path], second_tensors[path])
        if not difference.get("exact", False):
            differences.append((path, difference))

    compatible_differences = [
        item for item in differences if item[1].get("compatible", False)]
    compatible_differences.sort(
        key=lambda item: item[1]["relative_l2"], reverse=True)
    print(
        f"tensor summary: total={len(tensor_paths)} differing={len(differences)} "
        f"missing={len(missing)}")
    for path in missing[:args.top]:
        print(f"missing tensor: {path}")
    for path, difference in compatible_differences[:args.top]:
        print(
            f"{path}: relative_l2={difference['relative_l2']:.9e} "
            f"max_abs={difference['max_abs']:.9e}")
    for path, difference in differences:
        if not difference.get("compatible", False):
            print(
                f"{path}: incompatible {difference['first']} != "
                f"{difference['second']}")

    exact = not differences and not missing
    print(f"TENSOR COMPARISON: {'MATCH' if exact else 'DIFFER'}")
    raise SystemExit(0 if exact else 1)


if __name__ == "__main__":
    main()
