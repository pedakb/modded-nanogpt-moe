"""Loop oracle vs grouped local BP, plus direct standard-BP input anchoring.

CPU API doubles exercise both production autograd implementations and count
GEMMs. CUDA cases use the real extension/native kernels without guard bypasses.
All models keep FP32 master parameters, including BF16 activation cases.
"""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from modded_nanogpt_moe import _grouped_gemm as gemm
from modded_nanogpt_moe.config import load_experiment_config
from modded_nanogpt_moe.model import MoE


@pytest.fixture(params=["extension", "torch"])
def cpu_backend(request, monkeypatch):
    calls = []

    def raw(a, b, counts, trans_a=False, trans_b=False):
        calls.append("dw" if trans_a else "dx" if trans_b else "forward")
        segments = a.split(counts.tolist())
        if trans_a:
            return torch.stack([x.mT @ y for x, y in zip(segments, b.split(counts.tolist()))])
        return torch.cat([x @ (w.mT if trans_b else w) for x, w in zip(segments, b)])

    def baseline(a, b, counts, trans_b=False):
        assert not trans_b
        return gemm._ProfiledExtensionGemm.apply(a, b, counts, "test")

    def native(a, b, *, offs):
        counts = torch.diff(offs, prepend=offs.new_zeros(1)).long()
        if b.ndim == 2:
            return raw(a.mT, b, counts, trans_a=True)
        trans_b = not b.is_contiguous()
        return raw(a, b.mT if trans_b else b, counts, trans_b=trans_b)

    monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(
        ops=SimpleNamespace(gmm=baseline), backend=SimpleNamespace(gmm=raw)))
    monkeypatch.setattr(gemm, "validate_native_inputs", lambda a, b: None)
    monkeypatch.setattr(gemm.F, "grouped_mm", native)
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", request.param)
    return calls


def make_models(e=8, k=2, d=16, h=32, normalize=True, device="cpu", layout="packed"):
    torch.manual_seed(741)
    loop = MoE(d, e, k, hidden_dim=h, normalize_topk=normalize,
               moe_backward="grad_em", grad_em_mode="local_bp", grad_em_eta=0.4).to(device)
    with torch.no_grad():
        # Imbalanced routing, guaranteed empty final expert whenever k < e.
        loop.router.bias.copy_(torch.linspace(0.7, -0.7, e, device=device))
        if k < e:
            loop.router.bias[-1] = -100
    grouped = MoE(d, e, k, hidden_dim=h, normalize_topk=normalize,
                  moe_backend="grouped_gemm", moe_parameter_layout=layout,
                  moe_backward="grad_em", grad_em_mode="local_bp", grad_em_eta=0.4).to(device)
    if layout == "modulelist":
        grouped.load_state_dict(loop.state_dict())
    else:
        with torch.no_grad():
            grouped.router.load_state_dict(loop.router.state_dict())
            for i, expert in enumerate(loop.experts):
                grouped.fc_weight[i].copy_(expert.fc.weight.mT)
                grouped.fc_bias[i].copy_(expert.fc.bias)
                grouped.proj_weight[i].copy_(expert.proj.weight.mT)
                grouped.proj_bias[i].copy_(expert.proj.bias)
    standard = copy.deepcopy(grouped)
    standard.moe_backward = "standard"
    global_em = copy.deepcopy(grouped)
    global_em.grad_em_mode = "global"
    return loop, grouped, standard, global_em


def canonical_grads(model):
    def grad(p):
        return torch.zeros_like(p) if p.grad is None else p.grad

    result = {f"router.{name}": grad(p) for name, p in model.router.named_parameters()}
    if model.moe_parameter_layout == "modulelist":
        result.update({f"experts.{name}": grad(p) for name, p in model.experts.named_parameters()})
    else:
        for i in range(model.num_experts):
            result[f"experts.{i}.fc.weight"] = grad(model.fc_weight)[i].mT
            result[f"experts.{i}.fc.bias"] = grad(model.fc_bias)[i]
            result[f"experts.{i}.proj.weight"] = grad(model.proj_weight)[i].mT
            result[f"experts.{i}.proj.bias"] = grad(model.proj_bias)[i]
    return result


def run(model, value, upstream, need_x=True):
    x = value.detach().clone().requires_grad_(need_x)
    out = model(x)
    out.backward(upstream)
    return out.detach(), x.grad, canonical_grads(model)


def compare_pair(actual, expected, dtype):
    tolerance = dict(atol=2e-2, rtol=2e-2) if dtype == torch.bfloat16 else dict(atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual, expected, **tolerance)


def check_equivalence(models, dtype, *, variant="normal", report=False):
    loop, grouped, standard, global_em = models
    device = next(loop.parameters()).device
    d = loop.router.in_features
    torch.manual_seed(925)
    value = torch.randn(2, 7, d, device=device, dtype=dtype)
    upstream = torch.randn_like(value) / d**0.5
    if variant == "zero":
        upstream.zero_()
    for model in models:
        for name, p in model.named_parameters():
            if (variant == "frozen_all" or variant == "frozen_router" and name.startswith("router.")
                    or variant == "frozen_experts" and not name.startswith("router.")):
                p.requires_grad_(False)
    # Identical parameters imply identical support; check explicitly before VJPs.
    with torch.no_grad():
        supports = [m.router(value.flatten(0, 1)).float().softmax(-1).topk(m.top_k, -1).indices
                    for m in models]
        for support in supports[1:]:
            torch.testing.assert_close(support, supports[0], atol=0, rtol=0)
    results = [run(m, value, upstream, variant != "no_input_grad") for m in models]
    ref, local, bp, glob = results
    compare_pair(local[0], ref[0], dtype)
    torch.testing.assert_close(local[0], bp[0], atol=0, rtol=0)
    torch.testing.assert_close(local[0], glob[0], atol=0, rtol=0)
    if local[1] is not None:
        # Same backend/rounding: much tighter than loop-vs-grouped tolerance.
        torch.testing.assert_close(local[1], bp[1], atol=0, rtol=0)
        compare_pair(local[1], ref[1], dtype)
    for name in local[2]:
        compare_pair(local[2][name], ref[2][name], dtype)
        torch.testing.assert_close(local[2][name], glob[2][name], atol=0, rtol=0)
    if loop.top_k < loop.num_experts and variant != "frozen_experts":
        for suffix in ("fc.weight", "fc.bias", "proj.weight", "proj.bias"):
            assert torch.count_nonzero(local[2][f"experts.{loop.num_experts-1}.{suffix}"]) == 0
    for model in models:
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    if variant == "zero":
        assert all(torch.count_nonzero(g) == 0 for g in local[2].values())
        assert torch.count_nonzero(local[1]) == 0
    if report:
        maxdiff = lambda a, b: (a.float() - b.float()).abs().max().item()
        print(json.dumps({
            "dtype": str(dtype), "device": str(device),
            "implementation": grouped.gmm_implementation,
            "forward_max_abs": maxdiff(local[0], ref[0]),
            "input_max_abs": maxdiff(local[1], ref[1]),
            "input_vs_grouped_bp_max_abs": maxdiff(local[1], bp[1]),
            "expert_max_abs": max(maxdiff(local[2][n], ref[2][n]) for n in local[2] if n.startswith("experts.")),
            "router_max_abs": max(maxdiff(local[2][n], ref[2][n]) for n in local[2] if n.startswith("router.")),
        }, sort_keys=True))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("e,k", [(1, 1), (8, 1), (8, 2), (8, 8), (64, 8)])
@pytest.mark.parametrize("normalize", [True, False])
def test_loop_grouped_equivalence(cpu_backend, dtype, e, k, normalize):
    check_equivalence(make_models(e, k, normalize=normalize), dtype)


@pytest.mark.parametrize("variant", ["zero", "frozen_router", "frozen_experts", "frozen_all", "no_input_grad"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gradient_edges(cpu_backend, variant, dtype):
    if variant == "frozen_all":
        # The loop reference cannot request grad() with an empty parameter list.
        models = make_models()
        grouped, bp = models[1:3]
        for model in (grouped, bp):
            model.requires_grad_(False)
        value = torch.randn(2, 7, 16, dtype=dtype)
        g = torch.randn_like(value)
        actual, expected = run(grouped, value, g), run(bp, value, g)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        assert all(p.grad is None for p in grouped.parameters())
    else:
        check_equivalence(make_models(), dtype, variant=variant)


@pytest.mark.parametrize("layout", ["packed", "modulelist"])
def test_no_duplicate_wgrad_or_forward(cpu_backend, layout):
    _, grouped, _, _ = make_models(layout=layout)
    value = torch.randn(2, 7, 16)
    g = torch.randn_like(value)
    with torch.no_grad():
        expected = grouped(value)
    cpu_backend.clear()
    out, _, _ = run(grouped, value, g)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert cpu_backend.count("forward") == 2
    assert cpu_backend.count("dx") == 3  # GE FC2, BP FC2, BP FC1
    assert cpu_backend.count("dw") == 2  # GE FC2 and FC1 only


def check_two_layers(dtype, device="cpu"):
    first = make_models(device=device)
    second = make_models(device=device)
    torch.manual_seed(731)
    value = torch.randn(2, 7, 16, device=device, dtype=dtype)
    g = torch.randn_like(value) / 4
    results = []
    for a, b in zip(first, second):
        x = value.clone().requires_grad_()
        intermediate = a(x)
        intermediate.retain_grad()
        out = b(torch.tanh(intermediate))
        out.backward(g)
        results.append((x.grad, intermediate.grad, canonical_grads(a), canonical_grads(b)))
    ref, local, bp, glob = results
    for index in (0, 1):
        torch.testing.assert_close(local[index], bp[index], atol=0, rtol=0)
        compare_pair(local[index], ref[index], dtype)
    assert not torch.allclose(glob[1].float(), bp[1].float())
    for index in (2, 3):
        for name in local[index]:
            compare_pair(local[index][name], ref[index][name], dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_two_moe_input_boundary(cpu_backend, dtype):
    check_two_layers(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_numerical_report(cpu_backend, dtype):
    check_equivalence(make_models(), dtype, report=True)


def test_packed_config_preserves_loop_experiment():
    configs = Path(__file__).resolve().parents[1] / "configs"
    loop = load_experiment_config(configs / "moe_e8k2_r2_gradem_local_bp.toml")
    packed = load_experiment_config(configs / "moe_e8k2_r2_gradem_local_bp_packed.toml")
    assert packed["run_name"] != loop["run_name"]
    loop["run_name"] = packed["run_name"]
    loop["model"].update(moe_backend="grouped_gemm", moe_parameter_layout="packed")
    assert packed == loop


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def require_cuda_backend(monkeypatch, implementation, dtype):
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", implementation)
    if implementation == "torch":
        if torch.cuda.get_device_capability() != (9, 0) or dtype != torch.bfloat16:
            pytest.skip("native production backend requires SM90 and BF16")
    else:
        extension = pytest.importorskip("grouped_gemm")
        if dtype == torch.float32:
            # The installed Vista extension explicitly accepts BF16 only.
            # Probe the actual API and skip only that precise dtype rejection.
            try:
                extension.ops.gmm(torch.zeros(1, 16, device="cuda"),
                                  torch.zeros(1, 16, 16, device="cuda"), torch.tensor([1]))
            except RuntimeError as error:
                if "a.scalar_type() == torch::kBFloat16" not in str(error):
                    raise
                pytest.skip("installed nv-grouped-gemm rejects FP32; FP32 covered by CPU API doubles")


@CUDA
@pytest.mark.parametrize("implementation,dtype", [("extension", torch.float32),
                                                 ("extension", torch.bfloat16),
                                                 ("torch", torch.bfloat16)])
@pytest.mark.parametrize("e,k,h", [(8, 2, 1536), (64, 8, 384)])
def test_cuda_loop_grouped_equivalence(monkeypatch, implementation, dtype, e, k, h):
    require_cuda_backend(monkeypatch, implementation, dtype)
    check_equivalence(make_models(e, k, d=768, h=h, device="cuda"), dtype, report=True)


@CUDA
@pytest.mark.parametrize("implementation,dtype", [("extension", torch.float32),
                                                 ("extension", torch.bfloat16),
                                                 ("torch", torch.bfloat16)])
def test_cuda_two_moe(monkeypatch, implementation, dtype):
    require_cuda_backend(monkeypatch, implementation, dtype)
    check_two_layers(dtype, "cuda")


@CUDA
@pytest.mark.parametrize("implementation", ["extension", "torch"])
@pytest.mark.parametrize("variant", ["zero", "frozen_router", "frozen_experts", "no_input_grad"])
def test_cuda_gradient_edges(monkeypatch, implementation, variant):
    require_cuda_backend(monkeypatch, implementation, torch.bfloat16)
    check_equivalence(make_models(device="cuda"), torch.bfloat16, variant=variant)
