import torch

from modded_nanogpt_moe.model import GPT
from scripts.extract_checkpoint_routing import (
    add_router_parameters,
    evaluation_token_metadata,
    expert_gram,
    replay_routing,
    router_input_statistics,
    validate_artifact,
)
from scripts.extract_grad_em_diagnostics import build_replay_model, selected_dot_products


def _model(num_layers=1):
    return GPT(
        vocab_size=16, num_layers=num_layers, model_dim=128, mlp_type="moe",
        num_experts=3, top_k=2, normalize_topk=True, moe_backend="loop",
        mlp_ratio=1.0, moe_backward="standard")


def test_expert_gram_reconstructs_norm_distance_and_centered_geometry():
    outputs = torch.tensor([
        [[1.0, 2.0, -1.0, 0.5], [0.0, -2.0, 3.0, 1.0], [2.0, 1.0, 0.0, -1.0]],
        [[-1.0, 0.0, 2.0, 1.0], [3.0, 1.0, -2.0, 0.0], [0.5, 2.0, 1.0, -3.0]],
    ])
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    gram = expert_gram(outputs, order=None, top_k=3)
    dimension = outputs.shape[-1]

    torch.testing.assert_close(
        gram.diagonal(dim1=-2, dim2=-1), outputs.square().sum(-1) / dimension)
    explicit_distance = (outputs[:, :, None] - outputs[:, None, :]).square().sum(-1) / dimension
    gram_distance = (
        gram.diagonal(dim1=-2, dim2=-1)[:, :, None]
        + gram.diagonal(dim1=-2, dim2=-1)[:, None, :]
        - 2 * gram)
    torch.testing.assert_close(gram_distance, explicit_distance)

    mixture = torch.einsum("tk,tkd->td", weights, outputs)
    centered = outputs - mixture[:, None]
    explicit_centered = torch.bmm(centered, centered.transpose(1, 2)) / dimension
    r = torch.bmm(gram, weights.unsqueeze(-1)).squeeze(-1)
    s = torch.einsum("tk,tkl,tl->t", weights, gram, weights)
    reconstructed = gram - r[:, :, None] - r[:, None, :] + s[:, None, None]
    torch.testing.assert_close(reconstructed, explicit_centered)


def test_grouped_order_gram_and_sensitivity_match_token_slot_reference():
    outputs = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[-1.0, 0.5], [2.0, -3.0]],
    ])
    gradient = torch.tensor([[0.5, -2.0], [4.0, 1.0]])
    order = torch.tensor([1, 2, 0, 3])
    sorted_outputs = outputs.flatten(0, 1)[order]

    gram = expert_gram(sorted_outputs, order, top_k=2)
    sensitivity = selected_dot_products(gradient, sorted_outputs, order, top_k=2)
    torch.testing.assert_close(
        gram, torch.bmm(outputs, outputs.transpose(1, 2)) / outputs.shape[-1])
    torch.testing.assert_close(
        sensitivity, (gradient[:, None] * outputs).sum(-1))


def test_sensitivity_centering_and_output_gradient_norm_share_same_convention():
    generator = torch.Generator().manual_seed(4)
    outputs = torch.randn(3, 2, 5, generator=generator)
    gradient = torch.randn(3, 5, generator=generator)
    weights = torch.softmax(torch.randn(3, 2, generator=generator), dim=-1)
    sensitivity = selected_dot_products(gradient, outputs, None, top_k=2)
    mixture = torch.einsum("tk,tkd->td", weights, outputs)

    centered_sensitivity = sensitivity - (weights * sensitivity).sum(-1, keepdim=True)
    explicit = torch.einsum("td,tkd->tk", gradient, outputs - mixture[:, None])
    torch.testing.assert_close(centered_sensitivity, explicit)
    torch.testing.assert_close(
        torch.linalg.vector_norm(gradient, dim=-1), gradient.square().sum(-1).sqrt())


def test_router_input_sufficient_statistics_reconstruct_covariance():
    values = torch.tensor([
        [1.0, 2.0, -1.0], [2.0, -1.0, 0.0], [-2.0, 3.0, 4.0],
    ])
    norms, total, cross = router_input_statistics(values)
    mean = total / len(values)
    second = cross / len(values)
    covariance = second - mean[:, None] * mean[None, :]

    torch.testing.assert_close(norms, torch.linalg.vector_norm(values, dim=-1))
    torch.testing.assert_close(mean, values.mean(0))
    torch.testing.assert_close(second, values.T @ values / len(values))
    centered = values - values.mean(0)
    torch.testing.assert_close(covariance, centered.T @ centered / len(values))


def test_fixed_batch_token_metadata_makes_flattening_explicit():
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    targets = inputs.to(torch.int64) + 1
    metadata = evaluation_token_metadata({
        "inputs": inputs,
        "targets": targets,
        "metadata": {"token_hash": "fixed"},
    })
    assert metadata["input_ids"].tolist() == [1, 2, 3, 4, 5, 6]
    assert metadata["target_ids"].tolist() == [2, 3, 4, 5, 6, 7]
    assert metadata["sequence_index"].tolist() == [0, 0, 0, 1, 1, 1]
    assert metadata["token_position"].tolist() == [0, 1, 2, 0, 1, 2]


def test_full_and_chunked_replay_match_shapes_dtypes_and_values():
    torch.manual_seed(8)
    model = _model(num_layers=2)
    inputs = torch.tensor([
        [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
    ])
    targets = torch.tensor([
        [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13],
    ])
    full_loss, full_token_loss, full = replay_routing(
        model, inputs, targets, [0, 1], compile_head=False)
    chunked_loss, chunked_token_loss, chunked = replay_routing(
        model, inputs, targets, [0, 1],
        analysis_microbatch_sequences=2, compile_head=False)

    torch.testing.assert_close(full_loss, full_token_loss.sum())
    torch.testing.assert_close(chunked_loss, chunked_token_loss.sum())
    torch.testing.assert_close(full_loss, chunked_loss, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(full_token_loss, chunked_token_loss, rtol=2e-5, atol=2e-5)
    for layer_index in full:
        assert set(full[layer_index]) == {
            "logits", "topk_experts", "topk_weights", "sensitivity",
            "output_grad_norm", "expert_gram", "router_input_norm",
            "router_input_mean", "router_input_second_moment",
        }
        assert full[layer_index]["topk_experts"].dtype == torch.int32
        logits = full[layer_index]["logits"]
        expected_weights, expected_experts = torch.softmax(logits, dim=-1).topk(2, dim=-1)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
        expected_weights = expected_weights.to(torch.bfloat16).float()
        assert torch.equal(full[layer_index]["topk_experts"], expected_experts.to(torch.int32))
        torch.testing.assert_close(full[layer_index]["topk_weights"], expected_weights)
        for name, value in full[layer_index].items():
            if name != "topk_experts":
                assert value.dtype == torch.float32
            if name == "topk_experts":
                assert torch.equal(value, chunked[layer_index][name])
            else:
                torch.testing.assert_close(
                    value, chunked[layer_index][name], rtol=2e-5, atol=2e-5)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_artifact_schema_has_only_requested_base_quantities():
    torch.manual_seed(9)
    model = _model()
    inputs = torch.tensor([[1, 2, 3]])
    targets = torch.tensor([[2, 3, 4]])
    loss, token_loss, layers = replay_routing(
        model, inputs, targets, [0], compile_head=False)
    add_router_parameters(model, layers)
    artifact = {
        "schema_version": 1,
        "metadata": {
            "layers": ["l00"], "layer_indices": [0],
            "E": 3, "K": 2, "d": 128,
        },
        "loss": loss,
        "token_loss": token_loss,
        "layers": {"l00": layers[0]},
    }
    assert validate_artifact(artifact) is artifact
    assert set(artifact["layers"]["l00"]) == {
        "logits", "topk_experts", "topk_weights", "sensitivity",
        "output_grad_norm", "expert_gram", "router_input_norm",
        "router_input_mean", "router_input_second_moment", "router_weight",
        "router_bias",
    }


def test_grad_em_checkpoint_rebuilds_as_standard_task_gradient_replay(monkeypatch):
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "torch")
    config = {
        "vocab_size": 16, "num_layers": 1, "model_dim": 128,
        "mlp_type": "moe", "mlp_ratio": 1.0, "num_experts": 2,
        "top_k": 1, "normalize_topk": True, "moe_backend": "grouped_gemm",
        "moe_parameter_layout": "packed", "moe_backward": "standard",
        "grad_em_eta": 0.2,
    }
    source = GPT(**config)
    checkpoint_config = dict(config, moe_backward="grad_em")
    checkpoint = {
        "resolved_config": {"model": checkpoint_config},
        "model": source.state_dict(),
    }

    replay, trained_backward = build_replay_model(checkpoint, torch.device("cpu"))

    assert trained_backward == "grad_em"
    assert replay.blocks[0].mlp.moe_backward == "standard"
    assert replay.blocks[0].mlp.moe_parameter_layout == "packed"
