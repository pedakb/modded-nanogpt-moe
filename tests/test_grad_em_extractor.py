import os

import pytest
import torch

from modded_nanogpt_moe.model import GPT
from scripts.extract_grad_em_diagnostics import (
    SCHEMA_VERSION,
    atomic_torch_save,
    capture_moe_intermediates,
    checkpoint_identity,
    discover_checkpoints,
    load_or_create_evaluation_batch,
    replay_diagnostics,
    result_matches,
    selected_dot_products,
    tensor_hash,
    validate_result,
)


def _checkpoint(path, step):
    torch.save({"completed_updates": step}, path)


def test_checkpoint_discovery_deduplicates_aliases_and_prefers_numbered(tmp_path):
    numbered = tmp_path / "step_000050.pt"
    _checkpoint(numbered, 50)
    os.link(numbered, tmp_path / "latest.pt")
    _checkpoint(tmp_path / "step_000100.pt", 100)

    discovered = discover_checkpoints(tmp_path)

    assert [(step, path.name) for step, path in discovered] == [
        (50, "step_000050.pt"), (100, "step_000100.pt")]
    assert discover_checkpoints(tmp_path, [100])[0][1].name == "step_000100.pt"
    with pytest.raises(ValueError, match="missing"):
        discover_checkpoints(tmp_path, [150])


def test_checkpoint_discovery_supports_legacy_alias_only_directory(tmp_path):
    _checkpoint(tmp_path / "previous.pt", 50)
    _checkpoint(tmp_path / "latest.pt", 100)
    assert [(step, path.name) for step, path in discover_checkpoints(tmp_path)] == [
        (50, "previous.pt"), (100, "latest.pt")]


def test_eval_batch_hash_is_stable_and_sensitive():
    inputs = torch.tensor([[1, 2]], dtype=torch.int32)
    targets = torch.tensor([[2, 3]], dtype=torch.int64)
    assert tensor_hash(inputs, targets) == tensor_hash(inputs.clone(), targets.clone())
    assert tensor_hash(inputs, targets) != tensor_hash(inputs, targets + 1)


def test_fixed_eval_batch_is_created_once_then_reused(tmp_path):
    shard = tmp_path / "val.bin"
    shard.write_bytes(b"identity only")
    resolved = {
        "training": {
            "sequence_length": 2,
            "microbatch_sequences": 1,
            "validation_shard_pattern": "val.bin",
        },
        "seed_override": 123,
    }
    calls = []

    def create(config, data_root):
        calls.append(True)
        inputs = torch.tensor([[1, 2]], dtype=torch.int32)
        targets = torch.tensor([[2, 3]], dtype=torch.int64)
        metadata = {
            "shape": [1, 2],
            "input_dtype": "torch.int32",
            "target_dtype": "torch.int64",
            "validation_shard_pattern": "val.bin",
            "validation_shard_paths": [str(shard.resolve())],
            "seed": 123,
            "token_hash": tensor_hash(inputs, targets),
        }
        return {"schema_version": SCHEMA_VERSION, "inputs": inputs,
                "targets": targets, "metadata": metadata}

    first = load_or_create_evaluation_batch(tmp_path / "out", resolved, tmp_path, create)
    second = load_or_create_evaluation_batch(tmp_path / "out", resolved, tmp_path, create)
    assert len(calls) == 1
    assert torch.equal(first["inputs"], second["inputs"])
    assert first["metadata"]["token_hash"] == second["metadata"]["token_hash"]


def test_selected_dot_products_matches_explicit_token_slot_reference():
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    selected = torch.tensor([
        [[2.0, 1.0], [1.0, -1.0]],
        [[0.5, 2.0], [-2.0, 1.0]],
    ])
    order = torch.tensor([1, 2, 0, 3])
    sorted_outputs = selected.flatten(0, 1)[order]
    actual = selected_dot_products(gradient, sorted_outputs, order, top_k=2)
    expected = (gradient[:, None, :] * selected).sum(-1)
    torch.testing.assert_close(actual, expected)


def test_selected_logits_and_logsumexp_reconstruct_full_router_probabilities():
    logits = torch.tensor([[1.0, -2.0, 0.5], [3.0, 4.0, -1.0]])
    experts = torch.tensor([[0, 2], [1, 0]])
    selected = logits.gather(1, experts)
    reconstructed = torch.exp(selected - torch.logsumexp(logits, dim=-1)[:, None])
    torch.testing.assert_close(reconstructed, torch.softmax(logits, -1).gather(1, experts))


def test_result_schema_and_shape_range_validation():
    result = {
        "schema_version": SCHEMA_VERSION,
        "layers": {0: {
            "selected_logits": torch.zeros(3, 2),
            "v": torch.ones(3, 2),
            "topk_experts": torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32),
            "router_logsumexp": torch.zeros(3),
        }},
    }
    assert validate_result(result, 1, 4, 2) is result
    result["layers"][0]["topk_experts"][0, 0] = 4
    with pytest.raises(ValueError, match="out of range"):
        validate_result(result, 1, 4, 2)


def test_atomic_result_save_leaves_only_complete_destination(tmp_path):
    destination = tmp_path / "step_000050.pt"
    atomic_torch_save({"completed_updates": 50}, destination)
    assert torch.load(destination, weights_only=False)["completed_updates"] == 50
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_result_save_cleans_partial_file_on_replace_failure(tmp_path, monkeypatch):
    destination = tmp_path / "step_000050.pt"

    def fail_replace(source, target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        atomic_torch_save({"completed_updates": 50}, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_resume_match_requires_schema_step_source_and_batch(tmp_path):
    source = tmp_path / "checkpoint.pt"
    _checkpoint(source, 50)
    identity = checkpoint_identity(source)
    result = tmp_path / "result.pt"
    atomic_torch_save({
        "schema_version": SCHEMA_VERSION,
        "completed_updates": 50,
        "source_checkpoint": identity,
        "eval_batch_hash": "abc",
        "model": {"num_layers": 1, "num_experts": 2, "top_k": 1},
        "layers": {0: {
            "selected_logits": torch.zeros(1, 1),
            "v": torch.zeros(1, 1),
            "topk_experts": torch.zeros(1, 1, dtype=torch.int32),
            "router_logsumexp": torch.zeros(1),
        }},
    }, result)
    assert result_matches(result, 50, identity, "abc")
    assert not result_matches(result, 100, identity, "abc")
    assert not result_matches(result, 50, identity, "different")


def test_single_autograd_replay_captures_true_incoming_gradient_and_v():
    torch.manual_seed(7)
    model = GPT(
        vocab_size=16, num_layers=2, model_dim=128, mlp_type="moe",
        num_experts=3, top_k=2, normalize_topk=True, moe_backend="loop",
        mlp_ratio=1.0, moe_backward="standard")
    inputs = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    targets = torch.tensor([[2, 3, 4]], dtype=torch.int64)

    with capture_moe_intermediates(model) as (captures, expected_layers):
        loss = model(inputs, targets)
        gradients = torch.autograd.grad(
            loss, [captures[index]["combined_output"] for index in sorted(captures)])

    assert expected_layers == len(captures) == 2
    for layer_index, gradient in zip(sorted(captures), gradients):
        capture = captures[layer_index]
        actual = selected_dot_products(
            gradient, capture["expert_outputs"], capture["order"], top_k=2)
        reference = (
            gradient.reshape(-1, 128).float()[:, None, :]
            * capture["expert_outputs"].float()).sum(-1)
        torch.testing.assert_close(actual, reference)
        assert all(parameter.grad is None for parameter in model.parameters())


def test_sequence_chunking_preserves_order_and_raw_diagnostics():
    torch.manual_seed(11)
    model = GPT(
        vocab_size=16, num_layers=2, model_dim=128, mlp_type="moe",
        num_experts=3, top_k=2, normalize_topk=True, moe_backend="loop",
        mlp_ratio=1.0, moe_backward="standard")
    inputs = torch.tensor([
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
    ], dtype=torch.int64)
    targets = torch.tensor([
        [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13],
    ], dtype=torch.int64)

    full, full_layer_count = replay_diagnostics(
        model, inputs, targets, compile_head=False)
    chunked, chunked_layer_count = replay_diagnostics(
        model, inputs, targets, analysis_microbatch_sequences=2,
        compile_head=False)

    assert full_layer_count == chunked_layer_count == 2
    assert full.keys() == chunked.keys()
    for layer_index in full:
        assert full[layer_index].keys() == {
            "selected_logits", "v", "topk_experts", "router_logsumexp"}
        assert torch.equal(
            full[layer_index]["topk_experts"],
            chunked[layer_index]["topk_experts"])
        for name in ("selected_logits", "router_logsumexp", "v"):
            torch.testing.assert_close(
                full[layer_index][name], chunked[layer_index][name],
                rtol=2e-5, atol=2e-5)
        assert full[layer_index]["v"].shape[0] == inputs.numel()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_replay_compiles_the_shared_head_loss_as_fullgraph(monkeypatch):
    torch.manual_seed(13)
    model = GPT(
        vocab_size=16, num_layers=1, model_dim=128, mlp_type="moe",
        num_experts=2, top_k=1, normalize_topk=True, moe_backend="loop",
        mlp_ratio=1.0, moe_backward="standard")
    inputs = torch.tensor([[1, 2]], dtype=torch.int64)
    targets = torch.tensor([[2, 3]], dtype=torch.int64)
    calls = []

    def fake_compile(function, *, fullgraph, dynamic):
        calls.append((fullgraph, dynamic))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    replay_diagnostics(model, inputs, targets, compile_head=True)

    assert calls == [(True, False)]
