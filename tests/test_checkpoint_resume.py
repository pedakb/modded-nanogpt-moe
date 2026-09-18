import random

import numpy as np
import pytest
import torch

import modded_nanogpt_moe.optim as optim_module
from modded_nanogpt_moe.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_save_checkpoint,
    capture_rng_state,
    make_repro_diagnostic,
    restore_rng_state,
    save_repro_diagnostic,
    validate_checkpoint_config,
)
from modded_nanogpt_moe.data import DistributedDataLoader
from modded_nanogpt_moe.optim import Muon
from tools.compare_repro_diagnostics import _tensor_difference


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


def test_batched_muon_step_matches_per_parameter_update(monkeypatch):
    torch.manual_seed(2026)
    shapes = [(4, 6), (4, 6), (4, 6), (6, 4), (6, 4)]
    initial_values = [torch.randn(shape) for shape in shapes]
    gradients = [torch.randn(shape) for shape in shapes]
    initial_momentums = [torch.randn(shape) for shape in shapes]
    reference_params = [
        torch.nn.Parameter(value.clone()) for value in initial_values]
    batched_params = [
        torch.nn.Parameter(value.clone()) for value in initial_values]
    reference_momentums = [momentum.clone() for momentum in initial_momentums]
    lr = 0.025
    weight_decay = 0.05
    mu = 0.95

    with torch.no_grad():
        for parameter, gradient, momentum in zip(
                reference_params, gradients, reference_momentums):
            parameter.grad = gradient.clone()
            update = optim_module.muon_update(
                parameter.grad, momentum, mu=mu)
            parameter.mul_(1 - lr * weight_decay)
            parameter.add_(update, alpha=-lr)

    optimizer = Muon(
        batched_params, lr=lr, weight_decay=weight_decay, mu=mu)
    for parameter, gradient, momentum in zip(
            batched_params, gradients, initial_momentums):
        parameter.grad = gradient.clone()
        optimizer.state[parameter]["momentum"] = momentum.clone()

    calls = []
    original_muon_update = optim_module.muon_update

    def record_batched_call(gradient, momentum, **kwargs):
        calls.append(tuple(gradient.shape))
        return original_muon_update(gradient, momentum, **kwargs)

    monkeypatch.setattr(optim_module, "muon_update", record_batched_call)
    monkeypatch.setattr(optim_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(optim_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(optim_module.dist, "all_gather", lambda outputs, value: None)
    optimizer.step()

    assert sorted(calls) == [(2, 6, 4), (3, 4, 6)]
    for reference, actual, reference_momentum in zip(
            reference_params, batched_params, reference_momentums):
        # Batched BF16 matmul can select a different accumulation order than
        # separate GEMMs. The bound is approximately one BF16 update ULP at
        # this learning rate; momentum itself must still match exactly below.
        torch.testing.assert_close(actual, reference, rtol=0, atol=2e-4)
        torch.testing.assert_close(
            optimizer.state[actual]["momentum"],
            reference_momentum,
            rtol=0,
            atol=0,
        )


def test_repro_diagnostic_records_exact_state_without_advancing_rng(tmp_path):
    class Loader:
        def state_dict(self):
            return {"cursor": 7}

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.state[model.weight]["probe"] = torch.arange(6).reshape(2, 3)
    random.seed(44)
    np.random.seed(44)
    torch.manual_seed(44)
    rng_before = capture_rng_state()
    inputs = torch.arange(6).reshape(2, 3)
    payload = make_repro_diagnostic(
        "initialized", model, [optimizer], Loader(), 0, {"seed": 44},
        inputs=inputs)
    expected_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_inputs = inputs.clone()
    expected_optimizer_probe = optimizer.state[model.weight]["probe"].clone()
    with torch.no_grad():
        model.weight.add_(1)
    inputs.zero_()
    optimizer.state[model.weight]["probe"].zero_()
    path = save_repro_diagnostic(payload, tmp_path, "initialized.pt")
    rng_after = capture_rng_state()
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    assert loaded["stage"] == "initialized"
    assert loaded["data_loader"] == {"cursor": 7}
    for name, value in expected_model.items():
        torch.testing.assert_close(loaded["model"][name], value, rtol=0, atol=0)
    torch.testing.assert_close(loaded["inputs"], expected_inputs, rtol=0, atol=0)
    optimizer_state = loaded["optimizers"][0]["state"]["state"]
    torch.testing.assert_close(
        next(iter(optimizer_state.values()))["probe"],
        expected_optimizer_probe,
        rtol=0,
        atol=0,
    )
    assert rng_before["python"] == rng_after["python"]
    np.testing.assert_array_equal(rng_before["numpy"][1], rng_after["numpy"][1])
    torch.testing.assert_close(
        rng_before["torch_cpu"], rng_after["torch_cpu"], rtol=0, atol=0)


def test_repro_tensor_difference_reports_quantitative_error():
    difference = _tensor_difference(
        torch.tensor([3.0, 4.0]), torch.tensor([0.0, 0.0]))
    assert not difference["exact"]
    assert difference["relative_l2"] == 1.0
    assert difference["max_abs"] == 4.0
