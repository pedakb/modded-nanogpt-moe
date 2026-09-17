import random

import numpy as np
import pytest
import torch

from train_gpt_simple import (
    CHECKPOINT_FORMAT_VERSION,
    DistributedDataLoader,
    Muon,
    atomic_save_checkpoint,
    capture_rng_state,
    restore_rng_state,
    validate_checkpoint_config,
)


def _write_shard(path, tokens):
    tokens = np.asarray(tokens, dtype=np.uint16)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    with path.open("wb") as file:
        file.write(header.tobytes())
        file.write(tokens.tobytes())


def _make_relocated_shards(root_a, root_b):
    for root in (root_a, root_b):
        root.mkdir()
        _write_shard(root / "train_000.bin", np.arange(18))
        _write_shard(root / "train_001.bin", np.arange(100, 124))


def test_data_loader_restores_next_batch_across_shard_boundary_and_new_root(tmp_path):
    root_a = tmp_path / "system_a"
    root_b = tmp_path / "system_b"
    _make_relocated_shards(root_a, root_b)

    original = DistributedDataLoader(
        "train_*.bin", batch_size=8, seq_len=2, data_root=root_a, device="cpu")
    next(original)
    next(original)
    state = original.state_dict()
    expected = next(original)

    restored = DistributedDataLoader(
        "train_*.bin", batch_size=8, seq_len=2, data_root=root_b, device="cpu")
    restored.load_state_dict(state)
    actual = next(restored)

    assert state["shard_index"] == 0
    assert state["token_offset"] == 16
    assert restored.shard_index == 1
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_data_loader_rejects_changed_shard_identity(tmp_path):
    root_a = tmp_path / "system_a"
    root_b = tmp_path / "system_b"
    _make_relocated_shards(root_a, root_b)
    loader = DistributedDataLoader(
        "train_*.bin", batch_size=8, seq_len=2, data_root=root_a, device="cpu")
    state = loader.state_dict()
    _write_shard(root_b / "train_001.bin", np.arange(200, 225))
    relocated = DistributedDataLoader(
        "train_*.bin", batch_size=8, seq_len=2, data_root=root_b, device="cpu")
    with pytest.raises(ValueError, match="shards"):
        relocated.load_state_dict(state)


def test_rng_state_round_trip_is_exact():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(4), torch.rand(4))

    random.random()
    np.random.rand(4)
    torch.rand(4)
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(4), torch.rand(4))

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_atomic_checkpoint_rotation_keeps_previous_completed_file(tmp_path):
    first = {"format_version": CHECKPOINT_FORMAT_VERSION, "completed_updates": 1}
    second = {"format_version": CHECKPOINT_FORMAT_VERSION, "completed_updates": 2}
    atomic_save_checkpoint(first, tmp_path)
    atomic_save_checkpoint(second, tmp_path)

    latest = torch.load(tmp_path / "latest.pt", weights_only=False)
    previous = torch.load(tmp_path / "previous.pt", weights_only=False)
    assert latest["completed_updates"] == 2
    assert previous["completed_updates"] == 1
    assert not list(tmp_path.glob(".checkpoint-*.tmp"))


def test_checkpoint_config_requires_exact_schedule_and_model_match():
    config = {"model": {"mlp_type": "dense"}, "training": {"total_steps": 20}}
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "resolved_config": config,
    }
    validate_checkpoint_config(checkpoint, config)
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_config(
            checkpoint,
            {"model": {"mlp_type": "dense"}, "training": {"total_steps": 10}},
        )


def test_muon_state_dict_includes_persistent_momentum():
    parameter = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = Muon([parameter], lr=0.025, weight_decay=0.05)
    momentum = torch.randn_like(parameter)
    optimizer.state[parameter]["momentum"] = momentum.clone()

    restored_parameter = torch.nn.Parameter(torch.randn(4, 4))
    restored = Muon([restored_parameter], lr=1.0, weight_decay=0.0)
    restored.load_state_dict(optimizer.state_dict())

    torch.testing.assert_close(
        restored.state[restored_parameter]["momentum"], momentum, rtol=0, atol=0)
    assert restored.param_groups[0]["lr"] == 0.025
    assert restored.param_groups[0]["weight_decay"] == 0.05
