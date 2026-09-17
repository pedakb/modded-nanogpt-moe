"""Versioned training checkpoints, RNG state, and compatibility restoration."""

import os
import platform
import random
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


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
            ["git", *args], cwd=Path(__file__).resolve().parents[1],
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


def make_repro_diagnostic(stage, model, optimizers, train_loader,
                          completed_updates, runtime, include_training_state=True,
                          **extra):
    """Capture exact trainer state only for opt-in reproducibility diagnosis."""
    payload = {
        "stage": stage,
        "completed_updates": completed_updates,
        "data_loader": train_loader.state_dict(),
        "rng": capture_rng_state(),
        "runtime": runtime,
    }
    if include_training_state:
        payload["model"] = unwrap_model(model).state_dict()
        payload["optimizers"] = [
            {"name": type(optimizer).__name__, "state": optimizer.state_dict()}
            for optimizer in optimizers
        ]
    payload.update(extra)
    return _clone_repro_value(payload)


def _clone_repro_value(value):
    """Make a detached CPU snapshot that later training cannot mutate."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_repro_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_repro_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_repro_value(item) for item in value)
    return value


def save_repro_diagnostic(payload, diagnostics_dir: str | Path, filename: str):
    """Atomically write a named diagnostic without rotating training checkpoints."""
    diagnostics_dir = Path(diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    destination = diagnostics_dir / filename
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=f".{filename}-", suffix=".tmp",
                dir=diagnostics_dir, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        return destination
    except BaseException:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise

def restore_training_checkpoint(checkpoint, resolved_config, model, optimizers,
                                train_loader, train_steps, batch_size,
                                stop_after_updates=None):
    validate_checkpoint_config(checkpoint, resolved_config)
    completed_updates = int(checkpoint["completed_updates"])
    if not 0 <= completed_updates <= train_steps:
        raise ValueError(
            f"invalid completed update count in checkpoint: {completed_updates}")
    if checkpoint.get("processed_training_tokens") != completed_updates * batch_size:
        raise ValueError("checkpoint processed-training-token count is inconsistent")
    if stop_after_updates is not None and stop_after_updates <= completed_updates:
        raise ValueError(
            "STOP_AFTER_COMPLETED_UPDATES must be greater than the restored update count")
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer_states = checkpoint.get("optimizers", [])
    if len(optimizer_states) != len(optimizers):
        raise ValueError("checkpoint optimizer count is incompatible")
    for optimizer, saved_optimizer in zip(optimizers, optimizer_states):
        expected_name = type(optimizer).__name__
        if saved_optimizer.get("name") != expected_name:
            raise ValueError(
                f"checkpoint optimizer is incompatible: expected {expected_name}, "
                f"got {saved_optimizer.get('name')!r}")
        optimizer.load_state_dict(saved_optimizer["state"])
    train_loader.load_state_dict(checkpoint["data_loader"])
    timing = checkpoint["timing"]
    return {
        "completed_updates": completed_updates,
        "training_time": float(timing["training_time"]),
        "current_segment_time": float(timing["current_segment_time"]),
        "last_val_step": int(timing["last_val_step"]),
    }
