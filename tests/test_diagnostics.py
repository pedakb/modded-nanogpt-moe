import ast
import copy
import inspect
import math
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modded_nanogpt_moe import diagnostics as diag, optim, train
from modded_nanogpt_moe.config import load_experiment_config, validate_experiment_config
from modded_nanogpt_moe.model import MoE


SETTINGS = dict(scalar_interval=10, histogram_interval=0, during_nsys=False)


class Writer:
    def __init__(self):
        self.scalars, self.histograms = {}, {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (value, step)

    def add_histogram(self, tag, values, step):
        self.histograms[tag] = (values.copy(), step)


def test_router_metrics_from_full_softmax_and_actual_assignments():
    p = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]], requires_grad=True)
    stats = diag.RoutingStatistics(3, 2)
    # Feed two accumulation passes. Loads aggregate counts, not average CVs.
    stats.observe(p[:1].log(), p[:1], torch.tensor([[0, 1]]))
    stats.observe(p[1:].log(), p[1:], torch.tensor([[2, 1]]))
    values, q = stats.finish()
    expected_h = -(0.6 * math.log(0.6) + 0.3 * math.log(0.3) + 0.1 * math.log(0.1))
    assert values["entropy"].item() == pytest.approx(expected_h)
    assert values["normalized_entropy"].item() == pytest.approx(expected_h / math.log(3))
    assert values["top1_prob"].item() == pytest.approx(0.6)
    assert values["max_prob"] == values["top1_prob"]
    assert values["top1_top2_gap"].item() == pytest.approx(0.3)
    assert values["topk_margin"].item() == pytest.approx(math.log(3))
    assert values["topk_margin_median"].item() == pytest.approx(math.log(3))
    torch.testing.assert_close(q, torch.tensor([0.25, 0.5, 0.25]))
    assert values["load_cv"].item() == pytest.approx(q.std(correction=0).item() / q.mean().item())
    assert values["max_over_mean_load"].item() == pytest.approx(1.5)
    assert values["zero_experts"] == 0
    assert all(not v.requires_grad and v.grad_fn is None for v in values.values())


def test_margins_thresholds_zero_load_and_zero_probabilities():
    logits = torch.tensor([[0.4, 0.2, 0.2], [0.4, 0.2, 0.18], [0.4, 0.2, 0.12], [0.4, 0.2, 0.]])
    stats = diag.RoutingStatistics(3, 2)
    stats.observe(logits, logits.softmax(-1), torch.tensor([[0, 1]] * 4))
    values, q = stats.finish()
    assert values["topk_margin_median"].item() == pytest.approx(0.05)
    for threshold, fraction in (("0.01", 0.25), ("0.05", 0.5), ("0.1", 0.75)):
        assert values[f"topk_margin_below_{threshold}"].item() == fraction
    torch.testing.assert_close(q, torch.tensor([0.5, 0.5, 0.]))
    assert values["zero_experts"] == 1
    assert values["load_entropy"].item() == pytest.approx(math.log(2))
    assert values["normalized_load_entropy"].item() == pytest.approx(math.log(2) / math.log(3))
    stats = diag.RoutingStatistics(3, 1)
    stats.observe(torch.tensor([[0., -1000., -1000.]]), torch.tensor([[1., 0., 0.]]), torch.tensor([[0]]))
    assert stats.finish()[0]["entropy"] == 0


@pytest.mark.parametrize("e", [1, 3])
def test_k_equals_e_and_uniform_entropy(e):
    stats = diag.RoutingStatistics(e, e)
    stats.observe(torch.zeros(2, e), torch.full((2, e), 1 / e), torch.arange(e).expand(2, e))
    metrics, _ = stats.finish()
    assert "topk_margin" not in metrics
    assert metrics["normalized_entropy"].item() == pytest.approx(1 if e > 1 else 0)
    assert metrics["normalized_load_entropy"].item() == pytest.approx(1 if e > 1 else 0)
    assert metrics["logit_rms"].item() == 0


def test_centered_logit_rms_shift_invariance_and_microbatch_weighting():
    logits = torch.tensor([[1., 3.], [2., 6.], [0., 0.]])
    selected = logits.topk(1, dim=-1).indices
    def collect(z, parts):
        stats = diag.RoutingStatistics(2, 1)
        for start, end in parts:
            stats.observe(z[start:end], z[start:end].softmax(-1), selected[start:end])
        return stats.finish()[0]
    whole = collect(logits, [(0, 3)])
    split = collect(logits, [(0, 1), (1, 3)])
    shifted = collect(logits + torch.tensor([[100.], [-20.], [5.]]), [(0, 1), (1, 3)])
    assert whole["logit_rms"].item() == pytest.approx(math.sqrt(5 / 3))
    for metrics in (split, shifted):
        torch.testing.assert_close(metrics["logit_rms"], whole["logit_rms"], atol=0, rtol=0)
    assert whole["logit_rms"].grad_fn is None


@pytest.mark.parametrize("values", [[1.], [1., 2., 3., 4.], [1., 7., 9.], [0.] * 64,
                                    [float("nan"), 1., 2.]])
def test_percentile_selection_matches_linear_interpolation(values):
    values = torch.tensor(values)
    for q in (0.1, 0.9):
        torch.testing.assert_close(diag.percentile(values, q), values.quantile(q), equal_nan=True)


def test_rms_is_size_normalized_before_expert_aggregation():
    values = [torch.full((2, 2), 3.), torch.full((2, 8), 5.)]
    group = diag.ParameterGroup(values, experts=True)
    norms = group.norms(values)
    torch.testing.assert_close(norms, torch.tensor([6., 20.]))
    torch.testing.assert_close(group.rms(norms), torch.tensor([3., 5.]))
    assert diag.midpoint_median(group.rms(norms)) == 4
    router = diag.ParameterGroup(values)
    torch.testing.assert_close(router.rms(router.norms(values)), torch.tensor([math.sqrt(436 / 20)]))
    packed_values = torch.stack([torch.full((2, 4), 3.), torch.full((2, 4), 5.)])
    packed = diag.ParameterGroup([packed_values], experts=True, packed=True)
    torch.testing.assert_close(packed.rms(packed.norms([packed_values])), torch.tensor([3., 5.]))


def test_expert_summary_and_group_norms():
    values = torch.tensor([1., 2., 3., 4.])
    results = diag.summary(values)
    assert {name: v.item() for name, v in results.items()} == pytest.approx(
        dict(min=1, median=2.5, mean=2.5, max=4, std=math.sqrt(1.25)))
    x = torch.tensor([[[3., 4.]], [[0., 12.]]])
    for group, tensors in ((diag.ParameterGroup([], True, True), [x]),
                           (diag.ParameterGroup([], True), list(x))):
        torch.testing.assert_close(group.norms(tensors), torch.tensor([5., 12.]))
    torch.testing.assert_close(diag.ParameterGroup([]).norms(list(x)), torch.tensor([13.]))
    assert torch.isnan(diag.ratio(torch.tensor(1.), torch.tensor(0.)))


def toy_model(monkeypatch, layout="modulelist"):
    def gmm(a, w, counts, trans_b=False):
        return torch.cat([segment @ matrix for segment, matrix in zip(a.split(counts.tolist()), w)])
    monkeypatch.setitem(sys.modules, "grouped_gemm", SimpleNamespace(ops=SimpleNamespace(gmm=gmm)))
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "extension")
    model = nn.Module()
    block = nn.Module()
    block.attn = nn.Linear(8, 8)
    block.mlp = MoE(8, 4, 2, hidden_dim=8, moe_backend="grouped_gemm", moe_parameter_layout=layout)
    model.blocks = nn.ModuleList([block])
    model.embed = nn.Embedding(8, 8)
    model.proj = nn.Linear(8, 8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.125)
    return model


@pytest.mark.parametrize("layout", ["modulelist", "packed"])
def test_actual_adamw_muon_updates_gradients_and_no_state_changes(monkeypatch, layout):
    model = toy_model(monkeypatch, layout)
    reference = copy.deepcopy(model)
    settings = dict(SETTINGS, histogram_interval=10)
    observer = diag.TrainingDiagnostics(model, settings)
    assert not observer.due(9) and observer.due(10)
    writer = Writer()
    monkeypatch.setattr(optim.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(optim.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(optim.dist, "all_gather", lambda outputs, value: None)
    # The real Muon update including all 12 NS iterations, without compilation.
    monkeypatch.setattr(optim, "muon_update", optim.muon_update._torchdynamo_orig_callable)
    config = load_experiment_config()["optimizers"]
    config["adamw"]["fused"] = False
    actual_opts, expected_opts = optim.build_optimizers(model, config), optim.build_optimizers(reference, config)
    for p, ref in zip(model.parameters(), reference.parameters()):
        p.grad = torch.full_like(p, 0.25)
        ref.grad = p.grad.clone()
    old = {name: p.detach().clone() for name, p in model.named_parameters()}
    saved_rng = torch.get_rng_state().clone()
    with observer.capture_routing():
        with torch.no_grad():
            model.blocks[0].mlp(torch.ones(1, 2, 8))
        observer.observe_loss(torch.tensor(2.))
    observer.before_optimizers()
    # Gradients/snapshots are independent of subsequent optimizer mutation.
    for p, ref in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, ref.grad, atol=0, rtol=0)
    for optimizer in actual_opts + expected_opts:
        optimizer.step()
    observer.after_optimizers(writer, 10, 1)
    assert not observer.before and not observer.metrics
    assert writer.scalars["train/loss"] == (2., 10)
    assert "routing/layer_00/load_fraction" in writer.histograms
    assert writer.scalars["routing/layer_00/zero_experts"] == (2., 10)
    torch.testing.assert_close(torch.get_rng_state(), saved_rng, atol=0, rtol=0)
    for (name, p), ref in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(p, ref, atol=0, rtol=0)
    p = model.embed.weight
    delta = (old["embed.weight"].float() - p.detach().float()).norm().item()
    assert writer.scalars["optimization/embedding/update_norm"] == pytest.approx((delta, 10))
    assert writer.scalars["optimization/embedding/grad_norm"][0] == pytest.approx(2.)
    assert writer.scalars["optimization/embedding/param_rms"][0] == pytest.approx(0.125)
    assert writer.scalars["optimization/embedding/grad_rms"][0] == pytest.approx(0.25)
    assert writer.scalars["optimization/embedding/update_rms"][0] == pytest.approx(delta / 8)
    fc = model.blocks[0].mlp.fc_weight[0] if layout == "packed" else model.blocks[0].mlp.experts[0].fc.weight
    delta = (torch.full_like(fc, 0.125).float() - fc.detach().float()).norm().item()
    assert writer.scalars["optimization/experts/layer_00/fc1/update_norm_median"][0] == pytest.approx(delta)
    for stat in ("p10", "median", "p90"):
        assert writer.scalars[f"optimization/experts/layer_00/fc1/update_rms_{stat}"][0] == pytest.approx(delta / 8)
        assert writer.scalars[f"optimization/experts/layer_00/fc1/param_rms_{stat}"][0] == pytest.approx(0.125)
        assert writer.scalars[f"optimization/experts/layer_00/fc1/grad_rms_{stat}"][0] == pytest.approx(0.25)
        assert writer.scalars[f"optimization/experts/layer_00/fc1/update_ratio_{stat}"][0] == pytest.approx(delta)
    assert writer.scalars["optimization/router/layer_00/param_rms"][0] == pytest.approx(0.125)
    assert writer.scalars["optimization/router/layer_00/grad_rms"][0] == pytest.approx(0.25)
    assert "optimization/experts/layer_00/fc1/grad_rms_min" not in writer.scalars
    assert "optimization/experts/layer_00/fc1/grad_rms" not in writer.histograms
    assert "optimization/experts/layer_00/fc1/update_norm" in writer.histograms
    for actual, expected in zip(actual_opts, expected_opts):
        for p, ref in zip([p for g in actual.param_groups for p in g["params"]],
                          [p for g in expected.param_groups for p in g["params"]]):
            for key in actual.state[p]:
                torch.testing.assert_close(actual.state[p][key], expected.state[ref][key], atol=0, rtol=0)


def test_bf16_actual_delta_uses_stored_values(monkeypatch):
    model = toy_model(monkeypatch).to(torch.bfloat16)
    observer = diag.TrainingDiagnostics(model, SETTINGS)
    with observer.capture_routing():
        pass
    observer.before_optimizers()
    with torch.no_grad():
        model.embed.weight.add_(0.0001)  # Rounds away at 0.125 in BF16.
    writer = Writer()
    observer.after_optimizers(writer, 10, 1)
    assert writer.scalars["optimization/embedding/update_norm"] == (0., 10)
    assert not writer.histograms


def test_profile_explicit_opt_in_and_benchmark_always_wins(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(diag, "TrainingDiagnostics", lambda model, settings: sentinel)
    kwargs = dict(model=object(), writer=object(), settings=dict(SETTINGS, during_nsys=True),
                  rank=0, benchmark=False, nsys_profile=True)
    assert diag.make_diagnostics(**kwargs) is sentinel
    kwargs["benchmark"] = True
    assert diag.make_diagnostics(**kwargs) is None


@pytest.mark.parametrize("layout", ["modulelist", "packed"])
def test_routing_observer_parity_detachment_and_cleanup(monkeypatch, layout):
    model = toy_model(monkeypatch, layout)
    moe = model.blocks[0].mlp
    observer = diag.TrainingDiagnostics(model, SETTINGS)
    x = torch.randn(1, 7, 8, requires_grad=True)
    old = moe(x)
    expected = torch.autograd.grad(old.sum(), (x, *moe.parameters()))
    rng = torch.get_rng_state().clone()
    with observer.capture_routing():
        new = moe(x)
        observer.observe_loss(new.sum())
        actual = torch.autograd.grad(new.sum(), (x, *moe.parameters()))
    torch.testing.assert_close(new, old, atol=0, rtol=0)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert moe._routing_diagnostics is None
    assert not moe.router._forward_hooks
    assert observer.routing["layer_00"].tokens == 7
    assert not observer.losses[0].requires_grad
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    with pytest.raises(RuntimeError), observer.capture_routing():
        output = moe(x)
        raise RuntimeError("forward failed")
    assert moe._routing_diagnostics is None
    assert not moe.router._forward_hooks
    assert not observer.routing["layer_00"].gradient_handles


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sampled_logit_gradient_rms_no_retention_and_no_normal_hooks(monkeypatch, dtype):
    model = toy_model(monkeypatch).to(dtype)
    moe = model.blocks[0].mlp
    observer = diag.TrainingDiagnostics(model, SETTINGS)
    # A normal update has no module or tensor gradient hooks installed.
    normal = moe.router(torch.ones(1, 8, dtype=dtype))
    assert not moe.router._forward_hooks and not normal._backward_hooks
    refs = []
    expected_squares = 0.
    with observer.capture_routing():
        stats = observer.routing["layer_00"]
        for tokens, magnitude in ((1, 2.), (3, 4.)):
            logits = moe.router(torch.ones(tokens, 8, dtype=dtype))
            refs.append(weakref.ref(logits))
            p = logits.float().softmax(-1)
            stats.observe(logits, p, p.topk(2, dim=-1).indices)
            (logits * magnitude).sum().backward()
            expected_squares += tokens * 4 * magnitude**2
            del logits, p
            assert refs[-1]() is None  # Observer doesn't keep logits/graph alive.
        assert stats.logit_grad_elements == 16
        assert stats.logit_grad_squares.grad_fn is None
        assert stats.logit_grad_squares.numel() == 1
    value = stats.finish()[0]["dL_dlogits_rms"]
    assert value.item() == pytest.approx(math.sqrt(expected_squares / 16))
    assert not moe.router._forward_hooks and not stats.gradient_handles
    observer.before_optimizers()
    writer = Writer()
    observer.after_optimizers(writer, 10, 4)
    assert writer.scalars["routing/layer_00/dL_dlogits_rms"] == pytest.approx((value.item(), 10))
    # Detached/no-grad router output cannot carry a gradient hook.
    with observer.capture_routing(), torch.no_grad():
        moe(torch.ones(1, 2, 8, dtype=dtype))
        assert not observer.routing["layer_00"].gradient_handles
    assert "dL_dlogits_rms" not in observer.routing["layer_00"].finish()[0]
    observer.before_optimizers()
    writer = Writer()
    observer.after_optimizers(writer, 10, 1)
    assert "routing/layer_00/logit_rms" in writer.scalars


def test_expert_rms_percentile_series_are_per_expert(monkeypatch):
    model = toy_model(monkeypatch)
    observer = diag.TrainingDiagnostics(model, SETTINGS)
    with torch.no_grad():
        for index, expert in enumerate(model.blocks[0].mlp.experts):
            expert.fc.weight.fill_(index + 1.)
            expert.fc.weight.grad = torch.full_like(expert.fc.weight, 2 * (index + 1.))
    with observer.capture_routing():
        pass
    observer.before_optimizers()
    with torch.no_grad():
        for index, expert in enumerate(model.blocks[0].mlp.experts):
            expert.fc.weight.add_(0.25 * (index + 1.))
    writer = Writer()
    observer.after_optimizers(writer, 10, 1)
    for suffix, quantile in (("p10", 1.3), ("median", 2.5), ("p90", 3.7)):
        for metric, value in (("param_rms", quantile), ("grad_rms", 2 * quantile),
                              ("update_rms", 0.25 * quantile), ("update_ratio", 0.25)):
            tag = f"optimization/experts/layer_00/fc1/{metric}_{suffix}"
            assert writer.scalars[tag] == pytest.approx((value, 10))


@pytest.mark.parametrize("override", [dict(writer=None), dict(rank=1), dict(benchmark=True),
                                     dict(nsys_profile=True), dict(settings=dict(SETTINGS, scalar_interval=0))])
def test_disabled_does_not_construct_or_inspect_model(monkeypatch, override):
    def forbidden(*args):
        pytest.fail("disabled diagnostics constructed")
    monkeypatch.setattr(diag, "TrainingDiagnostics", forbidden)
    kwargs = dict(model=object(), writer=object(), settings=SETTINGS, rank=0, benchmark=False, nsys_profile=False)
    kwargs.update(override)
    assert diag.make_diagnostics(**kwargs) is None


def test_trainer_benchmark_guard_and_optimizer_boundaries(monkeypatch):
    tree = ast.parse(inspect.getsource(train.main))
    setup = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "tb_diagnostics" for t in node.targets))
    def forbidden(*args):
        pytest.fail("benchmark instantiated diagnostics")
    monkeypatch.setattr(diag, "TrainingDiagnostics", forbidden)
    namespace = dict(make_diagnostics=diag.make_diagnostics, model=object(), writer=object(),
                     experiment_config=dict(diagnostics=SETTINGS), benchmark=dict(enabled=True),
                     nsys_profile=False, dist=SimpleNamespace(get_rank=lambda: 0))
    exec(compile(ast.Module(body=[setup], type_ignores=[]), train.__file__, "exec"), namespace)
    assert namespace["tb_diagnostics"] is None
    calls = {ast.unparse(node.func): node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert calls["dist.all_reduce"] < calls["tb_diagnostics.before_optimizers"] < calls["opt.step"]
    assert calls["opt.step"] < calls["tb_diagnostics.after_optimizers"] < calls["model.zero_grad"]


@pytest.mark.parametrize("key,value", [("scalar_interval", -1), ("scalar_interval", True),
                                       ("histogram_interval", 11), ("during_nsys", "yes")])
def test_config_validation(key, value):
    config = load_experiment_config()
    config["diagnostics"][key] = value
    with pytest.raises(ValueError, match="diagnostics"):
        validate_experiment_config(config)
