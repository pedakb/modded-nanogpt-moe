"""
Tests for MoE: E=1/top_k=1 dense-equivalence, and E>1 top-k routing-weight
semantics (normalize_topk=True/False).

train_gpt_simple.py is a torchrun script (module-level CUDA/NCCL setup is guarded
under `if __name__ == "__main__":`), so importing it here only pulls in the
architecture/optimizer classes with no side effects.
"""
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from train_gpt_simple import MLP, MoE, GPT, add_bias_by_expert_segments, eager_prefix, make_head_loss

GROUPED_GEMM_AVAILABLE = importlib.util.find_spec("grouped_gemm") is not None


def _make_dense_and_moe(dim=32):
    torch.manual_seed(0)
    dense = MLP(dim)
    for p in dense.parameters():
        p.data.normal_()

    moe = MoE(dim, num_experts=1, top_k=1, normalize_topk=True)
    moe.experts[0].load_state_dict(dense.state_dict())
    # Router logits are irrelevant when num_experts=1 (softmax over one logit is
    # always exactly 1), but randomize them anyway to prove the equivalence does
    # not depend on router initialization.
    moe.router.weight.data.normal_()
    moe.router.bias.data.normal_()
    return dense, moe


def test_default_mlp_width_and_checkpoint_keys_remain_compatible():
    default = GPT(vocab_size=37, num_layers=2, model_dim=16)
    explicit = GPT(vocab_size=37, num_layers=2, model_dim=16, mlp_ratio=4)

    assert default.hidden_dim == explicit.hidden_dim == 64
    assert list(default.state_dict()) == list(explicit.state_dict())
    assert [name for name in default.state_dict() if ".mlp." in name] == [
        "blocks.0.mlp.fc.weight", "blocks.0.mlp.fc.bias",
        "blocks.0.mlp.proj.weight", "blocks.0.mlp.proj.bias",
        "blocks.1.mlp.fc.weight", "blocks.1.mlp.fc.bias",
        "blocks.1.mlp.proj.weight", "blocks.1.mlp.proj.bias",
    ]
    for name, default_tensor in default.state_dict().items():
        assert default_tensor.shape == explicit.state_dict()[name].shape, name
    for block in default.blocks:
        assert block.mlp.fc.weight.shape == (64, 16)
        assert block.mlp.fc.bias.shape == (64,)
        assert block.mlp.proj.weight.shape == (16, 64)
        assert block.mlp.proj.bias.shape == (16,)


def test_ratio_two_sets_dense_and_expert_width_independently_of_top_k():
    dense = GPT(vocab_size=37, num_layers=1, model_dim=16, mlp_ratio=2)
    assert dense.hidden_dim == 32
    assert dense.blocks[0].mlp.fc.weight.shape == (32, 16)
    assert dense.blocks[0].mlp.proj.weight.shape == (16, 32)

    for top_k in (1, 2):
        moe = GPT(vocab_size=37, num_layers=1, model_dim=16, mlp_type="moe",
                  mlp_ratio=2, num_experts=8, top_k=top_k)
        assert moe.hidden_dim == 32
        assert len(moe.blocks[0].mlp.experts) == 8
        for expert in moe.blocks[0].mlp.experts:
            assert expert.fc.weight.shape == (32, 16)
            assert expert.fc.bias.shape == (32,)
            assert expert.proj.weight.shape == (16, 32)
            assert expert.proj.bias.shape == (16,)


@pytest.mark.parametrize("mlp_ratio", [0, -1, float("nan"), float("inf"), float("-inf"), 0.1])
def test_invalid_mlp_ratios_are_rejected(mlp_ratio):
    with pytest.raises(ValueError):
        GPT(vocab_size=37, num_layers=1, model_dim=16, mlp_ratio=mlp_ratio)


def test_forward_equivalence():
    dense, moe = _make_dense_and_moe()
    x = torch.randn(2, 5, 32)

    dense_out = dense(x)
    moe_out = moe(x)

    torch.testing.assert_close(moe_out, dense_out)


def test_input_gradient_equivalence():
    dense, moe = _make_dense_and_moe()
    x = torch.randn(2, 5, 32)
    x_dense = x.clone().requires_grad_(True)
    x_moe = x.clone().requires_grad_(True)

    dense(x_dense).sum().backward()
    moe(x_moe).sum().backward()

    torch.testing.assert_close(x_moe.grad, x_dense.grad)


def test_parameter_gradient_equivalence():
    dense, moe = _make_dense_and_moe()
    x = torch.randn(2, 5, 32)

    dense(x).sum().backward()
    moe(x).sum().backward()

    expert0 = moe.experts[0]
    torch.testing.assert_close(expert0.fc.weight.grad, dense.fc.weight.grad)
    torch.testing.assert_close(expert0.fc.bias.grad, dense.fc.bias.grad)
    torch.testing.assert_close(expert0.proj.weight.grad, dense.proj.weight.grad)
    torch.testing.assert_close(expert0.proj.bias.grad, dense.proj.bias.grad)


def test_segmented_expert_bias_matches_indexing_output_and_gradients_with_empty_expert():
    counts = torch.tensor([3, 0, 2, 1], dtype=torch.int64)
    expert_ids = torch.repeat_interleave(torch.arange(4), counts)
    x_base = torch.arange(30, dtype=torch.bfloat16).view(6, 5) / 8
    bias_base = torch.arange(20, dtype=torch.float32).view(4, 5) / 16
    grad_out = (torch.arange(30, dtype=torch.bfloat16).view(6, 5) + 1) / 32

    x_index = x_base.clone().requires_grad_(True)
    bias_index = bias_base.clone().requires_grad_(True)
    out_index = x_index + bias_index.to(torch.bfloat16)[expert_ids]
    out_index.backward(grad_out)

    x_segment = x_base.clone().requires_grad_(True)
    bias_segment = bias_base.clone().requires_grad_(True)
    out_segment = add_bias_by_expert_segments(
        x_segment, bias_segment.to(torch.bfloat16), counts)
    out_segment.backward(grad_out)

    torch.testing.assert_close(out_segment, out_index)
    torch.testing.assert_close(x_segment.grad, x_index.grad)
    torch.testing.assert_close(bias_segment.grad, bias_index.grad)
    assert bias_segment.grad is not None
    assert torch.count_nonzero(bias_segment.grad[1]) == 0


def test_top_k_bounds_are_validated():
    import pytest

    with pytest.raises(AssertionError):
        MoE(dim=8, num_experts=2, top_k=3)


def test_multi_expert_top1_normalize_true_weight_is_exactly_one():
    dim, num_experts = 8, 4
    torch.manual_seed(1)
    moe = MoE(dim, num_experts=num_experts, top_k=1, normalize_topk=True)
    x = torch.randn(6, dim)

    router_logits = moe.router(x)
    expected_expert = router_logits.argmax(dim=-1)

    out = moe(x.unsqueeze(0)).squeeze(0)
    for i in range(x.size(0)):
        expected = moe.experts[expected_expert[i].item()](x[i:i+1]).squeeze(0)
        torch.testing.assert_close(out[i], expected)


def test_multi_expert_top1_normalize_false_uses_raw_softmax_probability():
    # With num_experts > 1, normalize_topk=False must NOT collapse to weight 1:
    # the selected expert's routing weight should equal its raw full-softmax
    # probability, which is generally < 1 once other experts have any mass.
    dim, num_experts = 8, 4
    torch.manual_seed(1)
    moe = MoE(dim, num_experts=num_experts, top_k=1, normalize_topk=False)
    x = torch.randn(6, dim)

    router_logits = moe.router(x)
    probs = F.softmax(router_logits.float(), dim=-1)
    top_prob, top_expert = probs.topk(1, dim=-1)
    assert (top_prob < 1).all(), "test setup should exercise weights strictly below 1"

    out = moe(x.unsqueeze(0)).squeeze(0)
    for i in range(x.size(0)):
        expert_out = moe.experts[top_expert[i, 0].item()](x[i:i+1]).squeeze(0)
        torch.testing.assert_close(out[i], expert_out * top_prob[i, 0].type_as(expert_out))


def test_topk_multi_selection_matches_manual_softmax_weights():
    # General top_k > 1 check, for both normalize_topk settings: the routed
    # output must equal the sum of selected experts' outputs weighted by
    # (optionally renormalized) full-softmax probabilities.
    dim, num_experts, top_k = 8, 5, 3
    for normalize_topk in (True, False):
        torch.manual_seed(2)
        moe = MoE(dim, num_experts=num_experts, top_k=top_k, normalize_topk=normalize_topk)
        x = torch.randn(4, dim)

        router_logits = moe.router(x)
        probs = F.softmax(router_logits.float(), dim=-1)
        top_probs, top_idx = probs.topk(top_k, dim=-1)
        if normalize_topk:
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        out = moe(x.unsqueeze(0)).squeeze(0)
        for i in range(x.size(0)):
            expected = torch.zeros(dim)
            for slot in range(top_k):
                expert_out = moe.experts[top_idx[i, slot].item()](x[i:i+1]).squeeze(0)
                expected = expected + expert_out * top_probs[i, slot].type_as(expert_out)
            torch.testing.assert_close(out[i], expected, atol=1e-5, rtol=1e-4)


def _make_small_gpt(mlp_type="dense", **moe_kwargs):
    torch.manual_seed(0)
    model = GPT(vocab_size=37, num_layers=2, model_dim=16, mlp_type=mlp_type, **moe_kwargs)
    # Explicit nonzero projection weights: the real training init (in
    # train_gpt_simple.py's __main__) zeros proj.weight, which would make an
    # eager-vs-compiled head/loss comparison numerically degenerate (all-zero
    # logits either way) and is not sufficient validation on its own.
    model.proj.weight.data.normal_(std=0.02)
    model.proj.bias.data.normal_(std=0.02)
    return model


_HEAD_LOSS_CONFIGS = [
    ("dense", {}),
    ("moe", dict(num_experts=1, top_k=1, normalize_topk=True)),
]


def test_eager_prefix_and_head_loss_match_monolithic_forward():
    # eager_prefix(model, inputs) -> make_head_loss(model)(x, targets) must be
    # an exact decomposition of GPT.forward, not an approximation of it.
    for mlp_type, kwargs in _HEAD_LOSS_CONFIGS:
        model = _make_small_gpt(mlp_type, **kwargs)
        inputs = torch.randint(0, 37, (2, 6))
        targets = torch.randint(0, 37, (2, 6))

        monolithic = model(inputs, targets)
        x = eager_prefix(model, inputs)
        split = make_head_loss(model)(x, targets)

        torch.testing.assert_close(split, monolithic)


def test_head_loss_eager_vs_compiled_fullgraph_parity():
    # The head/loss region is meant to run under torch.compile(fullgraph=True)
    # in train_gpt_simple.py. Check loss AND gradients (w.r.t. both the
    # block-output activation and the projection weight) match eager, with
    # nonzero projection weights (see _make_small_gpt) so the check isn't
    # vacuously true on all-zero logits.
    for mlp_type, kwargs in _HEAD_LOSS_CONFIGS:
        model = _make_small_gpt(mlp_type, **kwargs)
        head_loss = make_head_loss(model)
        compiled_head_loss = torch.compile(head_loss, fullgraph=True, dynamic=False)

        torch.manual_seed(3)
        x_base = torch.randn(2, 6, 16, dtype=torch.bfloat16)
        targets = torch.randint(0, 37, (2, 6))

        for p in model.parameters():
            p.grad = None
        x_eager = x_base.clone().requires_grad_(True)
        loss_eager = head_loss(x_eager, targets)
        loss_eager.backward()
        grad_proj_eager = model.proj.weight.grad.clone()
        grad_x_eager = x_eager.grad.clone()

        for p in model.parameters():
            p.grad = None
        x_compiled = x_base.clone().requires_grad_(True)
        loss_compiled = compiled_head_loss(x_compiled, targets)
        loss_compiled.backward()
        grad_proj_compiled = model.proj.weight.grad.clone()
        grad_x_compiled = x_compiled.grad.clone()

        assert not torch.allclose(grad_proj_eager, torch.zeros_like(grad_proj_eager)), (
            "grad_proj is all zero -- nonzero-weight setup did not take effect, "
            "this check would otherwise be degenerate")

        torch.testing.assert_close(loss_compiled, loss_eager, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(grad_proj_compiled, grad_proj_eager, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(grad_x_compiled, grad_x_eager, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# moe_backend selector: "loop" (default, existing) vs "grouped_gemm" (new)
# --------------------------------------------------------------------------- #

def test_moe_backend_defaults_to_loop_and_is_backward_compatible():
    # Existing call sites (no moe_backend kwarg) must keep working unchanged.
    moe = MoE(16, num_experts=4, top_k=2)
    assert moe.moe_backend == "loop"


def test_gpt_threads_moe_backend_to_every_block():
    model = GPT(vocab_size=37, num_layers=3, model_dim=16, mlp_type="moe",
                num_experts=4, top_k=2, moe_backend="loop")
    for block in model.blocks:
        assert block.mlp.moe_backend == "loop"


def test_grouped_gemm_backend_rejects_unknown_backend_name():
    with pytest.raises(AssertionError):
        MoE(16, num_experts=2, top_k=1, moe_backend="not_a_real_backend")


@pytest.mark.skipif(GROUPED_GEMM_AVAILABLE, reason="grouped_gemm IS installed; this test "
                     "targets the unavailable-dependency error path specifically")
def test_grouped_gemm_backend_unavailable_raises_clear_error():
    # "fail clearly if unavailable": constructing with moe_backend="grouped_gemm"
    # without the package installed must raise immediately, at construction time
    # (not buried inside a later forward/backward call), naming the package and
    # pointing at install instructions.
    with pytest.raises(ImportError, match="grouped_gemm"):
        MoE(16, num_experts=4, top_k=2, moe_backend="grouped_gemm")


@pytest.mark.skipif(not GROUPED_GEMM_AVAILABLE, reason="requires the grouped_gemm package "
                     "(PyPI nv-grouped-gemm) -- pending LS6, not installed here")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_gemm requires CUDA, none "
                     "available here")
@pytest.mark.parametrize("hidden_dim", [64, 32], ids=["ratio_4", "ratio_2"])
def test_grouped_gemm_backend_matches_loop_backend_output_and_gradients(hidden_dim):
    # Requirement: output/gradient parity using IDENTICAL weights and inputs,
    # plus an explicit checkpoint-key / optimizer-parameter-coverage comparison.
    # Both backends share the exact same nn.ModuleList(MLP) storage, so a
    # state_dict trained under one backend loads directly into the other.
    dim, num_experts, top_k = 16, 8, 2
    torch.manual_seed(6)
    moe_loop = MoE(dim, num_experts=num_experts, top_k=top_k, normalize_topk=True,
                   moe_backend="loop", hidden_dim=hidden_dim).to(
                       device="cuda", dtype=torch.bfloat16)
    moe_gg = MoE(dim, num_experts=num_experts, top_k=top_k, normalize_topk=True,
                 moe_backend="grouped_gemm", hidden_dim=hidden_dim).to(
                     device="cuda", dtype=torch.bfloat16)
    for p in moe_loop.parameters():
        p.data.normal_(std=0.05)
    # Force one expert to receive zero tokens, to exercise the documented
    # None-vs-present-and-zero gradient difference between backends.
    with torch.no_grad():
        moe_loop.router.bias[3] = -1.0e4

    moe_gg.load_state_dict(moe_loop.state_dict())

    # Checkpoint keys / optimizer-parameter coverage: identical by construction
    # (same nn.ModuleList(MLP) storage for both backends), asserted explicitly
    # rather than assumed.
    assert set(moe_loop.state_dict().keys()) == set(moe_gg.state_dict().keys())
    assert set(id(p) for p in moe_loop.parameters()) != set(id(p) for p in moe_gg.parameters()), (
        "sanity check: these must be two distinct parameter sets (loaded via "
        "load_state_dict, not the same objects), or this test would be vacuous")
    loop_named_params = list(moe_loop.named_parameters())
    gg_named_params = list(moe_gg.named_parameters())
    assert [name for name, _ in loop_named_params] == [name for name, _ in gg_named_params]
    for (name, p_loop), (_, p_gg) in zip(loop_named_params, gg_named_params, strict=True):
        assert p_loop.shape == p_gg.shape, name
        assert p_loop.dtype == p_gg.dtype == torch.bfloat16, name
        assert torch.equal(p_loop, p_gg), name

    torch.manual_seed(7)
    x = torch.randn(2, 20, dim, dtype=torch.bfloat16, device="cuda")
    x_loop = x.clone().requires_grad_(True)
    x_gg = x.clone().requires_grad_(True)

    out_loop = moe_loop(x_loop)
    out_loop.float().sum().backward()
    out_gg = moe_gg(x_gg)
    out_gg.float().sum().backward()

    torch.testing.assert_close(out_loop, out_gg, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x_loop.grad, x_gg.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(moe_loop.router.weight.grad, moe_gg.router.weight.grad,
                               atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(moe_loop.router.bias.grad, moe_gg.router.bias.grad,
                               atol=2e-2, rtol=2e-2)

    for i in range(num_experts):
        e_loop, e_gg = moe_loop.experts[i], moe_gg.experts[i]
        loop_expert_params = dict(e_loop.named_parameters())
        gg_expert_params = dict(e_gg.named_parameters())
        assert loop_expert_params.keys() == gg_expert_params.keys()
        if i == 3:
            # Documented difference (see MoE._forward_grouped_gemm docstring):
            # "loop" leaves an unused expert's grad as None; "grouped_gemm"
            # produces a present, exactly-zero grad. Both are checked
            # explicitly, not assumed.
            for name in loop_expert_params:
                assert loop_expert_params[name].grad is None, name
                gg_grad = gg_expert_params[name].grad
                assert gg_grad is not None, name
                assert gg_grad.dtype == gg_expert_params[name].dtype, name
                assert torch.count_nonzero(gg_grad) == 0, name
            continue
        for name in loop_expert_params:
            torch.testing.assert_close(loop_expert_params[name].grad, gg_expert_params[name].grad,
                                       atol=2e-2, rtol=2e-2)
