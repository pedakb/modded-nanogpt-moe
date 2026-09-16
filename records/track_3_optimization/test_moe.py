"""
Tests for MoE: E=1/top_k=1 dense-equivalence, and E>1 top-k routing-weight
semantics (normalize_topk=True/False).

train_gpt_simple.py is a torchrun script (module-level CUDA/NCCL setup is guarded
under `if __name__ == "__main__":`), so importing it here only pulls in the
architecture/optimizer classes with no side effects.
"""
import torch
import torch.nn.functional as F

from train_gpt_simple import MLP, MoE


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
