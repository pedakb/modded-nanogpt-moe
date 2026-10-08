import copy

import pytest
import torch

from modded_nanogpt_moe.model import GPT, eager_prefix
from modded_nanogpt_moe.router_regularization import router_losses, model_router_losses
from modded_nanogpt_moe.config import load_experiment_config, validate_experiment_config
from modded_nanogpt_moe.checkpoint import validate_checkpoint_config, CHECKPOINT_FORMAT_VERSION
from test_shared_expert import cpu_grouped_gemm


def small_model(mode="standard", backend="loop", shared=0, layout="modulelist"):
    return GPT(vocab_size=16, num_layers=2, model_dim=128, mlp_type="moe",
               num_experts=4, top_k=2, mlp_ratio=0.125,
               moe_backend=backend, moe_parameter_layout=layout,
               moe_backward="standard" if mode == "standard" else "grad_em",
               grad_em_mode="global" if mode == "standard" else mode,
               num_shared_experts=shared, shared_expert_ratio=0.125)


def grads(model):
    return {n: torch.zeros_like(p) if p.grad is None else p.grad.clone()
            for n, p in model.named_parameters()}


def test_values_and_balance():
    logits = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
    aux, z = router_losses(logits, 1)
    torch.testing.assert_close(aux, torch.tensor(1.))
    torch.testing.assert_close(z, torch.logsumexp(logits, -1).square().mean())
    imbalanced, _ = router_losses(torch.tensor([[2., 0.], [2., 0.]]), 1)
    assert imbalanced > aux
    (aux + z).backward()
    assert torch.isfinite(logits.grad).all()
    # Explicit detached-count derivative, including nonselected probabilities.
    x = torch.randn(7, 4, requires_grad=True)
    aux, z = router_losses(x, 2)
    p = x.softmax(-1)
    f = torch.nn.functional.one_hot(p.topk(2).indices, 4).float().sum((0, 1)) * (4 / 14)
    expected = (f.detach() * p.mean(0)).sum() + x.logsumexp(-1).square().mean()
    torch.testing.assert_close(torch.autograd.grad(aux + z, x)[0],
                               torch.autograd.grad(expected, x)[0])


@pytest.mark.parametrize("mode,backend,layout", [
    ("standard", "loop", "modulelist"),
    ("standard", "grouped_gemm", "packed"),
    ("global", "grouped_gemm", "packed"),
    ("local_bp", "loop", "modulelist"),
    ("local_bp", "grouped_gemm", "packed"),
])
@pytest.mark.parametrize("shared", [0, 1])
def test_additive_gradients(mode, backend, layout, shared, cpu_grouped_gemm):
    torch.manual_seed(42)
    model = small_model(mode, backend, shared, layout)
    inputs = torch.randint(16, (2, 3))
    # LM-only Grad-EM gradient is captured independently, at the same weights.
    model(inputs, inputs).backward()
    lm = grads(model)
    model.zero_grad(set_to_none=True)
    aux, z = model_router_losses(model, inputs)
    (0.01 * aux + 0.001 * z).backward()
    regularizer = grads(model)
    bp = copy.deepcopy(model)
    for block in bp.blocks:
        block.mlp.moe_backward = "standard"
    bp.zero_grad(set_to_none=True)
    a, b = model_router_losses(bp, inputs)
    (0.01 * a + 0.001 * b).backward()
    for name, gradient in grads(bp).items():
        torch.testing.assert_close(regularizer[name], gradient, rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    model(inputs, inputs).backward()
    a, b = model_router_losses(model, inputs)
    (0.01 * a + 0.001 * b).backward()
    for name, gradient in grads(model).items():
        torch.testing.assert_close(gradient, lm[name] + regularizer[name])
    assert model.blocks[0].mlp.moe_backward == ("standard" if mode == "standard" else "grad_em")
    assert regularizer["blocks.0.mlp.router.weight"].abs().sum() > 0
    # Last-layer expert/shared parameters cannot affect any router objective.
    for name, gradient in regularizer.items():
        if name.startswith("blocks.1.mlp.") and "router" not in name:
            assert gradient.count_nonzero() == 0


def test_accumulation():
    model = small_model()
    inputs = torch.randint(16, (2, 3)).repeat(2, 1)
    # Repeated microbatches have identical counts; their token sums must agree.
    a, z = model_router_losses(model, inputs)
    ((a + z) * inputs.numel()).backward()
    full = grads(model)
    model.zero_grad(set_to_none=True)
    for micro in inputs.chunk(2):
        a, z = model_router_losses(model, micro)
        ((a + z) * micro.numel()).backward()
    for name, gradient in grads(model).items():
        torch.testing.assert_close(gradient, full[name], atol=2e-5, rtol=2e-5)


def test_loop_grouped_compatibility(cpu_grouped_gemm, request):
    # Test mathematical parity with FP32 matmuls, rather than medium-precision
    # CPU bmm versus linear kernels selected by the trainer's global setting.
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    request.addfinalizer(lambda: torch.set_float32_matmul_precision(previous))
    loop = small_model(shared=1)
    grouped = small_model(backend="grouped_gemm", shared=1)
    loop.embed.float()
    grouped.embed.float()
    grouped.load_state_dict(loop.state_dict())
    inputs = torch.randint(16, (2, 3))
    results = []
    for model in (loop, grouped):
        a, z = model_router_losses(model, inputs)
        (a + z).backward()
        results.append((a, z, grads(model)))
    for i in (0, 1):
        torch.testing.assert_close(results[0][i], results[1][i])
    for name in results[0][2]:
        torch.testing.assert_close(results[0][2][name], results[1][2][name],
                                   atol=2e-6, rtol=2e-5)


def test_regularized_checkpoint_resume(tmp_path):
    from modded_nanogpt_moe.checkpoint import make_training_checkpoint, restore_training_checkpoint
    class Loader:
        def state_dict(self):
            return {}
        def load_state_dict(self, state):
            pass
    model = small_model(shared=1)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001)]
    config = load_experiment_config()
    config["model"].update(router_aux_loss_coef=0.01, router_z_loss_coef=0.001)
    config["training"]["total_steps"] = 20
    inputs = torch.randint(16, (2, 3))
    def update(model, optimizers):
        model(inputs, inputs).backward()
        a, z = model_router_losses(model, inputs)
        ((0.01 * a + 0.001 * z) * inputs.numel()).backward()
        optimizers[0].step()
        model.zero_grad(set_to_none=True)
    update(model, optimizers)
    checkpoint = make_training_checkpoint(model, optimizers, 1, 6, config,
                                          Loader(), "regularized", 0, 1., 0.5, 0, {})
    path = tmp_path / "regularized.pt"
    torch.save(checkpoint, path)
    restored = small_model(shared=1)
    restored_opts = [torch.optim.AdamW(restored.parameters(), lr=0.001)]
    restore_training_checkpoint(torch.load(path, weights_only=False), config,
                                restored, restored_opts, Loader(), train_steps=20,
                                batch_size=6)
    update(model, optimizers)
    update(restored, restored_opts)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name], rtol=0, atol=0)


def test_config_and_legacy_resume():
    config = load_experiment_config()
    legacy = copy.deepcopy(config)
    for key in ("router_aux_loss_coef", "router_z_loss_coef"):
        assert config["model"][key] == 0
        del legacy["model"][key]
    checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION, "resolved_config": legacy}
    validate_checkpoint_config(checkpoint, config)
    config["model"]["router_aux_loss_coef"] = 0.01
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_config(checkpoint, config)
    for invalid in (-1, float("nan"), float("inf"), True):
        config["model"]["router_aux_loss_coef"] = invalid
        with pytest.raises(ValueError, match="router_aux_loss_coef"):
            validate_experiment_config(config)


def test_eta_independence_and_disabled_parity(cpu_grouped_gemm):
    model = small_model("global", "grouped_gemm", 1, "packed")
    baseline = copy.deepcopy(model)
    inputs = torch.randint(16, (2, 3))
    # Zero coefficients never call the replay; no model keys/RNG/init change.
    for candidate in (model, baseline):
        eager_prefix(candidate, inputs).sum().backward()
    for name, gradient in grads(model).items():
        torch.testing.assert_close(gradient, grads(baseline)[name], rtol=0, atol=0)
    expected = None
    for eta in (0.001, 1.0):
        model.zero_grad(set_to_none=True)
        for block in model.blocks:
            block.mlp.grad_em_eta = eta
        a, z = model_router_losses(model, inputs)
        (a + z).backward()
        current = grads(model)
        if expected is not None:
            for name in current:
                torch.testing.assert_close(current[name], expected[name], rtol=0, atol=0)
        expected = current


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["standard", "global", "local_bp"])
def test_cuda_additivity(mode, monkeypatch):
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "torch")
    model = small_model(mode, "grouped_gemm", 1, "packed").cuda()
    inputs = torch.randint(16, (2, 3), device="cuda")
    model(inputs, inputs).backward()
    lm = grads(model)
    model.zero_grad(set_to_none=True)
    a, z = model_router_losses(model, inputs)
    (0.01 * a + 0.001 * z).backward()
    regularizer = grads(model)
    model.zero_grad(set_to_none=True)
    model(inputs, inputs).backward()
    a, z = model_router_losses(model, inputs)
    (0.01 * a + 0.001 * z).backward()
    for name, gradient in grads(model).items():
        torch.testing.assert_close(gradient, lm[name] + regularizer[name])
