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
from modded_nanogpt_moe.grad_em import GradEMCombine
from modded_nanogpt_moe.model import MoE


SETTINGS = dict(scalar_interval=10, histogram_interval=0, during_nsys=False)


class Writer:
    def __init__(self):
        self.scalars, self.histograms = {}, {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (value, step)

    def add_histogram(self, tag, values, step):
        self.histograms[tag] = (values.copy(), step)


def toy_model(monkeypatch, layout="modulelist", layers=1, moe_backward="standard"):
    def gmm(a, w, counts, trans_b=False):
        return torch.cat([
            segment @ matrix for segment, matrix in zip(a.split(counts.tolist()), w)])

    monkeypatch.setitem(
        sys.modules, "grouped_gemm", SimpleNamespace(ops=SimpleNamespace(gmm=gmm)))
    monkeypatch.setenv("MOE_GMM_IMPLEMENTATION", "extension")
    model = nn.Module()
    blocks = []
    for _ in range(layers):
        block = nn.Module()
        block.attn = nn.Linear(8, 8)
        block.mlp = MoE(
            8, 4, 2, hidden_dim=8, moe_backend="grouped_gemm",
            moe_parameter_layout=layout, moe_backward=moe_backward)
        blocks.append(block)
    model.blocks = nn.ModuleList(blocks)
    model.embed = nn.Embedding(8, 8)
    model.proj = nn.Linear(8, 8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.125)
    return model


def expert_weights(moe, expert):
    if moe.moe_parameter_layout == "packed":
        return moe.fc_weight[expert], moe.proj_weight[expert]
    return moe.experts[expert].fc.weight, moe.experts[expert].proj.weight


def test_router_statistics_compute_only_retained_behavior_metrics():
    p = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])
    stats = diag.RoutingStatistics(3, 2)
    selected = torch.tensor([[0, 1]])
    stats.observe(p[:1].log(), p[:1], selected, torch.tensor([[2 / 3, 1 / 3]]))
    stats.observe(p[1:].log(), p[1:], torch.tensor([[2, 1]]),
                  torch.tensor([[2 / 3, 1 / 3]]))
    values = stats.finish()
    expected_pre = -sum(value * math.log(value) for value in (0.6, 0.3, 0.1))
    expected_post = -sum(value * math.log(value) for value in (2 / 3, 1 / 3))
    q = torch.tensor([0.25, 0.5, 0.25])
    assert set(values) == {
        "normalized_entropy", "entropy_post_norm", "logit_range_median",
        "topk_margin_median", "load_cv", "zero_experts"}
    assert values["normalized_entropy"].item() == pytest.approx(
        expected_pre / math.log(3))
    assert values["entropy_post_norm"].item() == pytest.approx(
        expected_post / math.log(2))
    assert values["logit_range_median"].item() == pytest.approx(math.log(6))
    assert values["topk_margin_median"].item() == pytest.approx(math.log(3))
    assert values["load_cv"].item() == pytest.approx(
        q.std(correction=0).item() / q.mean().item())
    assert values["zero_experts"] == 0
    assert all(not value.requires_grad and value.grad_fn is None
               for value in values.values())


@pytest.mark.parametrize("experts", [1, 3])
def test_k_equals_e_omits_boundary_margin(experts):
    stats = diag.RoutingStatistics(experts, experts)
    stats.observe(
        torch.zeros(2, experts), torch.full((2, experts), 1 / experts),
        torch.arange(experts).expand(2, experts),
        torch.full((2, experts), 1 / experts))
    metrics = stats.finish()
    assert "topk_margin_median" not in metrics
    assert metrics["normalized_entropy"].item() == pytest.approx(
        1 if experts > 1 else 0)
    assert metrics["entropy_post_norm"].item() == pytest.approx(
        1 if experts > 1 else 0)
    assert metrics["logit_range_median"] == 0


def test_margin_median_and_empty_experts_preserve_definitions():
    logits = torch.tensor([
        [0.4, 0.2, 0.2], [0.4, 0.2, 0.18],
        [0.4, 0.2, 0.12], [0.4, 0.2, 0.]])
    stats = diag.RoutingStatistics(3, 2)
    probabilities = logits.softmax(-1)
    selected = torch.tensor([[0, 1]] * 4)
    selected_weights = probabilities.gather(1, selected)
    selected_weights /= selected_weights.sum(dim=-1, keepdim=True)
    stats.observe(logits, probabilities, selected, selected_weights)
    values = stats.finish()
    assert values["topk_margin_median"].item() == pytest.approx(0.05)
    assert values["zero_experts"] == 1


def test_midpoint_median_selection():
    assert diag.midpoint_median(torch.tensor([1., 4., 2., 3.])) == 2.5
    assert diag.midpoint_median(torch.tensor([1., 9., 7.])) == 7
    assert torch.isnan(diag.midpoint_median(torch.tensor([1., float("nan")])))


def test_standard_and_grad_em_sensitivity_use_existing_backward_values():
    standard = diag.RoutingStatistics(4, 2)
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]], requires_grad=True)
    sensitivity = torch.tensor([[3., -1.], [2., 8.]])
    standard.attach_sensitivity_gradient(weights)
    (weights * sensitivity).sum().backward()
    assert standard.finish() == {}  # Routing state has not been observed.
    assert diag.midpoint_median(torch.cat(standard.sensitivity_ranges)) == 5

    grad_em = diag.RoutingStatistics(4, 2)
    logits = torch.tensor([[2., 1., 0., -1.], [0., 1., 2., 3.]], requires_grad=True)
    indices = torch.tensor([[0, 1], [3, 2]])
    selected_logits = logits.gather(1, indices)
    topk_weights = selected_logits.softmax(-1)
    selected_outputs = torch.tensor([
        [[1., 2.], [3., 4.]], [[-1., 2.], [2., -2.]]], requires_grad=True)
    incoming = torch.tensor([[2., -1.], [3., 4.]])
    order = torch.arange(4)
    output = GradEMCombine.apply(
        selected_outputs.flatten(0, 1), logits, topk_weights, indices, order,
        0.1, grad_em.observe_sensitivity)
    output.backward(incoming)
    expected_v = (incoming[:, None, :] * selected_outputs.detach()).sum(-1)
    expected_range = expected_v.amax(-1) - expected_v.amin(-1)
    torch.testing.assert_close(
        torch.cat(grad_em.sensitivity_ranges), expected_range, atol=0, rtol=0)


@pytest.mark.parametrize("layout", ["modulelist", "packed"])
def test_expert_fc1_fc2_are_combined_before_median(monkeypatch, layout):
    model = toy_model(monkeypatch, layout)
    moe = model.blocks[0].mlp
    observer = diag.TrainingDiagnostics(model, SETTINGS)
    fc_values = torch.tensor([1., 2., 3., 4.])
    proj_values = torch.tensor([7., 6., 5., 4.])
    fc_grads = torch.tensor([2., 4., 6., 8.])
    proj_grads = torch.tensor([1., 3., 5., 7.])
    fc_updates = torch.tensor([0.1, 0.2, 0.3, 0.4])
    proj_updates = torch.tensor([0.8, 0.6, 0.4, 0.2])
    with torch.no_grad():
        if layout == "packed":
            moe.fc_weight.grad = torch.empty_like(moe.fc_weight)
            moe.proj_weight.grad = torch.empty_like(moe.proj_weight)
        for expert in range(4):
            fc, proj = expert_weights(moe, expert)
            fc.fill_(fc_values[expert])
            proj.fill_(proj_values[expert])
            if layout == "packed":
                moe.fc_weight.grad[expert].fill_(fc_grads[expert])
                moe.proj_weight.grad[expert].fill_(proj_grads[expert])
            else:
                fc.grad = torch.full_like(fc, fc_grads[expert])
                proj.grad = torch.full_like(proj, proj_grads[expert])
    with observer.capture_routing():
        pass
    observer.before_optimizers()
    with torch.no_grad():
        for expert in range(4):
            fc, proj = expert_weights(moe, expert)
            fc.add_(fc_updates[expert])
            proj.add_(proj_updates[expert])
    writer = Writer()
    observer.after_optimizers(writer, 10)

    param_rms = ((fc_values.square() + proj_values.square()) / 2).sqrt()
    grad_rms = ((fc_grads.square() + proj_grads.square()) / 2).sqrt()
    update_rms = ((fc_updates.square() + proj_updates.square()) / 2).sqrt()
    update_ratio = ((fc_updates.square() + proj_updates.square()) /
                    (fc_values.square() + proj_values.square())).sqrt()
    for metric, values in (
            ("param_rms", param_rms), ("grad_rms", grad_rms),
            ("update_rms", update_rms), ("update_ratio", update_ratio)):
        expected = diag.midpoint_median(values).item()
        assert writer.scalars[f"opt/expert/l00/{metric}_med"] == pytest.approx(
            (expected, 10))
    # This differs from combining independently summarized FC1 and FC2 values.
    separate_then_combine = math.sqrt(
        (diag.midpoint_median(fc_values).square()
         + diag.midpoint_median(proj_values).square()).item() / 2)
    assert writer.scalars["opt/expert/l00/param_rms_med"][0] != pytest.approx(
        separate_then_combine)
    assert not writer.histograms


def test_global_metrics_are_true_parameter_count_weighted_values(monkeypatch):
    model = toy_model(monkeypatch)
    observer = diag.TrainingDiagnostics(model, SETTINGS)
    old_values, grad_values, delta_values = [], [], []
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) / 10)
            parameter.grad = torch.full_like(parameter, (index + 2) / 20)
            old_values.append(parameter.detach().float().flatten().clone())
            grad_values.append(parameter.grad.detach().float().flatten().clone())
            delta_values.append(torch.full_like(old_values[-1], (index + 1) / 1000))
    with observer.capture_routing():
        pass
    observer.before_optimizers()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.add_((index + 1) / 1000)
    writer = Writer()
    observer.after_optimizers(writer, 10)
    old = torch.cat(old_values)
    gradients = torch.cat(grad_values)
    deltas = torch.cat(delta_values)
    expected = {
        "param_rms": old.square().mean().sqrt(),
        "grad_rms": gradients.square().mean().sqrt(),
        "update_rms": deltas.square().mean().sqrt(),
        "update_ratio": deltas.norm() / old.norm(),
    }
    for metric, value in expected.items():
        assert writer.scalars[f"opt/global/{metric}"] == pytest.approx(
            (value.item(), 10), rel=1e-5)


@pytest.mark.parametrize("layout", ["modulelist", "packed"])
def test_actual_optimizer_steps_and_states_are_unchanged(monkeypatch, layout):
    model = toy_model(monkeypatch, layout)
    reference = copy.deepcopy(model)
    observer = diag.TrainingDiagnostics(model, SETTINGS)
    monkeypatch.setattr(optim.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(optim.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(optim.dist, "all_gather", lambda outputs, value: None)
    monkeypatch.setattr(
        optim, "muon_update", optim.muon_update._torchdynamo_orig_callable)
    config = load_experiment_config()["optimizers"]
    config["adamw"]["fused"] = False
    actual_opts = optim.build_optimizers(model, config)
    expected_opts = optim.build_optimizers(reference, config)
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        parameter.grad = torch.full_like(parameter, 0.25)
        expected.grad = parameter.grad.clone()
    with observer.capture_routing():
        with torch.no_grad():
            model.blocks[0].mlp(torch.ones(1, 2, 8))
    observer.before_optimizers()
    for optimizer in actual_opts + expected_opts:
        optimizer.step()
    writer = Writer()
    observer.after_optimizers(
        writer, 10, extra_scalars={"metric/loss/train": torch.tensor(2.)})
    assert writer.scalars["metric/loss/train"] == (2., 10)
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
    for actual, expected in zip(actual_opts, expected_opts):
        actual_parameters = [p for group in actual.param_groups for p in group["params"]]
        expected_parameters = [p for group in expected.param_groups for p in group["params"]]
        for parameter, expected_parameter in zip(actual_parameters, expected_parameters):
            for key in actual.state[parameter]:
                torch.testing.assert_close(
                    actual.state[parameter][key], expected.state[expected_parameter][key],
                    atol=0, rtol=0)


def test_bf16_actual_delta_uses_stored_values(monkeypatch):
    model = toy_model(monkeypatch).to(torch.bfloat16)
    observer = diag.TrainingDiagnostics(model, SETTINGS)
    with observer.capture_routing():
        pass
    observer.before_optimizers()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.0001)  # Rounds away at 0.125 in BF16.
    writer = Writer()
    observer.after_optimizers(writer, 10)
    assert writer.scalars["opt/global/update_rms"] == (0., 10)
    assert writer.scalars["opt/global/update_ratio"] == (0., 10)


@pytest.mark.parametrize("layout", ["modulelist", "packed"])
@pytest.mark.parametrize("moe_backward", ["standard", "grad_em"])
def test_routing_observer_preserves_outputs_gradients_rng_and_cleanup(
        monkeypatch, layout, moe_backward):
    model = toy_model(monkeypatch, layout, moe_backward=moe_backward)
    moe = model.blocks[0].mlp
    observer = diag.TrainingDiagnostics(model, SETTINGS)
    x = torch.randn(1, 7, 8, requires_grad=True)
    old = moe(x)
    expected = torch.autograd.grad(old.sum(), (x, *moe.parameters()))
    rng = torch.get_rng_state().clone()
    with observer.capture_routing():
        new = moe(x)
        actual = torch.autograd.grad(new.sum(), (x, *moe.parameters()))
    torch.testing.assert_close(new, old, atol=0, rtol=0)
    for value, expected_value in zip(actual, expected):
        torch.testing.assert_close(value, expected_value, atol=0, rtol=0)
    assert observer.routing["l00"].tokens == 7
    assert "sensitivity_range_median" in observer.routing["l00"].finish()
    assert moe._routing_diagnostics is None
    assert not moe.router._forward_hooks
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    with pytest.raises(RuntimeError), observer.capture_routing():
        moe(x)
        raise RuntimeError("forward failed")
    assert moe._routing_diagnostics is None
    assert not moe.router._forward_hooks
    assert not observer.routing["l00"].gradient_handles


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_logit_gradient_rms_retains_no_graph_and_has_no_normal_hooks(
        monkeypatch, dtype):
    model = toy_model(monkeypatch).to(dtype)
    moe = model.blocks[0].mlp
    observer = diag.TrainingDiagnostics(model, SETTINGS)
    normal = moe.router(torch.ones(1, 8, dtype=dtype))
    assert not moe.router._forward_hooks and not normal._backward_hooks
    refs = []
    expected_squares = 0.
    with observer.capture_routing():
        stats = observer.routing["l00"]
        for tokens, magnitude in ((1, 2.), (3, 4.)):
            logits = moe.router(torch.ones(tokens, 8, dtype=dtype))
            refs.append(weakref.ref(logits))
            probabilities = logits.float().softmax(-1)
            weights, indices = probabilities.topk(2, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            stats.observe(logits, probabilities, indices, weights)
            (logits * magnitude).sum().backward()
            expected_squares += tokens * 4 * magnitude**2
            del logits, probabilities
            assert refs[-1]() is None
    value = stats.finish()["dL_dlogits_rms"]
    assert value.item() == pytest.approx(math.sqrt(expected_squares / 16))
    assert not moe.router._forward_hooks and not stats.gradient_handles
    observer.before_optimizers()
    writer = Writer()
    observer.after_optimizers(writer, 10)
    assert writer.scalars["opt/router/l00/dlogit_rms"] == pytest.approx(
        (value.item(), 10))


@pytest.mark.parametrize("layout", ["modulelist", "packed"])
def test_only_representative_layers_emit_compact_heavy_surface(monkeypatch, layout):
    model = toy_model(monkeypatch, layout, layers=12)
    observer = diag.TrainingDiagnostics(model, dict(SETTINGS, histogram_interval=10))
    assert set(observer.layers) == {"l00", "l05", "l11"}
    with observer.capture_routing():
        for index, block in enumerate(model.blocks):
            assert (block.mlp._routing_diagnostics is not None) == (index in (0, 5, 11))
        for index in (0, 5, 11):
            model.blocks[index].mlp(torch.ones(1, 3, 8)).sum().backward()
    observer.before_optimizers()
    writer = Writer()
    observer.after_optimizers(
        writer, 10, extra_scalars={"metric/loss/train": torch.tensor(2.)})

    expected = {"metric/loss/train"}
    expected.update(f"opt/global/{metric}" for metric in (
        "param_rms", "grad_rms", "update_rms", "update_ratio"))
    for layer in ("l00", "l05", "l11"):
        expected.update(f"opt/router/{layer}/{metric}" for metric in (
            "param_rms", "grad_rms", "update_rms", "update_ratio", "dlogit_rms",
            "sens_range_med"))
        expected.update(f"opt/expert/{layer}/{metric}_med" for metric in (
            "param_rms", "grad_rms", "update_rms", "update_ratio"))
        expected.update({
            f"router/{layer}/entropy_norm", f"router/{layer}/entropy_post_norm",
            f"router/{layer}/logit_range_med", f"router/{layer}/topk_margin_med",
            f"router/{layer}/load/cv", f"router/{layer}/load/zero",
        })
    assert set(writer.scalars) == expected
    assert len(writer.scalars) == 53
    assert not writer.histograms
    discarded = (
        "/fc1/", "/fc2/", "param_norm", "grad_norm", "update_norm",
        "grad_ratio", "_min", "_max", "_mean", "_std", "_p10", "_p90",
        "opt/compare/", "opt/attn/", "opt/embed/", "opt/head/",
        "/logit_rms", "/top1_prob", "/top1_top2_gap",
        "/margin/", "/load/entropy", "/load/min",
        "/load/max", "/load/mean", "/load/std", "/load/max_mean",
    )
    for tag in writer.scalars:
        assert not any(fragment in tag for fragment in discarded), tag
        assert not any(f"l{index:02d}" in tag for index in range(12)
                       if index not in (0, 5, 11)), tag


def test_expected_production_scalar_surface_has_exactly_60_series():
    tags = {
        "metric/loss/train", "metric/loss/val", "perf/step_ms", "perf/tok_s",
        "opt/lr/adamw/g0", "opt/lr/adamw/g1", "opt/lr/adamw/g2", "opt/lr/muon",
    }
    tags.update(f"opt/global/{metric}" for metric in (
        "param_rms", "grad_rms", "update_rms", "update_ratio"))
    for layer in ("l00", "l05", "l11"):
        tags.update(f"opt/router/{layer}/{metric}" for metric in (
            "param_rms", "grad_rms", "update_rms", "update_ratio", "dlogit_rms",
            "sens_range_med"))
        tags.update(f"opt/expert/{layer}/{metric}_med" for metric in (
            "param_rms", "grad_rms", "update_rms", "update_ratio"))
        tags.update({
            f"router/{layer}/entropy_norm", f"router/{layer}/entropy_post_norm",
            f"router/{layer}/logit_range_med", f"router/{layer}/topk_margin_med",
            f"router/{layer}/load/cv", f"router/{layer}/load/zero",
        })
    assert len(tags) == 60


def test_sampling_cadence_uses_completed_updates(monkeypatch):
    observer = diag.TrainingDiagnostics(
        toy_model(monkeypatch), dict(SETTINGS, scalar_interval=25))
    assert not observer.due(1)
    assert not observer.due(24)
    assert observer.due(25)
    assert not observer.due(49)
    assert observer.due(50)


@pytest.mark.parametrize("adamw_groups,muon_groups", [(3, 1), (1, 1), (4, 2)])
def test_trainer_learning_rate_tags_generalize_group_counts(adamw_groups, muon_groups):
    adamw = torch.optim.AdamW([
        dict(params=[nn.Parameter(torch.ones(2, 2))], lr=0.01 * (index + 1))
        for index in range(adamw_groups)])
    muon = optim.Muon([nn.Parameter(torch.ones(2, 2))], lr=0.025)
    for index in range(1, muon_groups):
        muon.add_param_group(dict(
            params=[nn.Parameter(torch.ones(2, 2))], lr=0.025 * (index + 1)))
    tree = ast.parse(inspect.getsource(train.main))
    lr_loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                   and "learning_rate_tag(opt, grp_idx)" in ast.unparse(node)
                   and ast.unparse(node.target) == "(opt_idx, opt)")
    writer = Writer()
    namespace = dict(
        writer=writer, optimizers=[adamw, muon], step=7,
        learning_rate_tag=train.learning_rate_tag)
    exec(compile(ast.Module(body=[lr_loop], type_ignores=[]), train.__file__, "exec"),
         namespace)
    expected = {
        f"opt/lr/adamw/g{index}": (0.01 * (index + 1), 7)
        for index in range(adamw_groups)}
    expected.update({
        "opt/lr/muon" if muon_groups == 1 else f"opt/lr/muon/g{index}":
            (0.025 * (index + 1), 7)
        for index in range(muon_groups)})
    assert writer.scalars == expected


def test_trainer_has_only_canonical_loss_and_performance_tags():
    tree = ast.parse(inspect.getsource(train.main))
    constant_tags = [
        node.args[0].value for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "writer.add_scalar"
        and isinstance(node.args[0], ast.Constant)]
    assert "perf/train_s" not in constant_tags
    assert set(constant_tags) == {
        "metric/loss/train", "metric/loss/val", "perf/step_ms", "perf/tok_s"}
    assert constant_tags.count("metric/loss/train") == 1
    source = inspect.getsource(train.main)
    assert 'extra_scalars={"metric/loss/train": train_loss}' in source
    assert 'elif writer is not None:\n                    writer.add_scalar(' in source


@pytest.mark.parametrize(
    "override",
    [dict(writer=None), dict(rank=1), dict(benchmark=True), dict(nsys_profile=True),
     dict(settings=dict(SETTINGS, scalar_interval=0))])
def test_disabled_does_not_construct_or_inspect_model(monkeypatch, override):
    def forbidden(*args):
        pytest.fail("disabled diagnostics constructed")

    monkeypatch.setattr(diag, "TrainingDiagnostics", forbidden)
    kwargs = dict(
        model=object(), writer=object(), settings=SETTINGS, rank=0,
        benchmark=False, nsys_profile=False)
    kwargs.update(override)
    assert diag.make_diagnostics(**kwargs) is None


def test_profile_explicit_opt_in_and_benchmark_always_wins(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(diag, "TrainingDiagnostics", lambda model, settings: sentinel)
    kwargs = dict(
        model=object(), writer=object(), settings=dict(SETTINGS, during_nsys=True),
        rank=0, benchmark=False, nsys_profile=True)
    assert diag.make_diagnostics(**kwargs) is sentinel
    kwargs["benchmark"] = True
    assert diag.make_diagnostics(**kwargs) is None


def test_trainer_benchmark_guard_and_optimizer_boundaries(monkeypatch):
    tree = ast.parse(inspect.getsource(train.main))
    setup = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "tb_diagnostics"
                         for target in node.targets))

    def forbidden(*args):
        pytest.fail("benchmark instantiated diagnostics")

    monkeypatch.setattr(diag, "TrainingDiagnostics", forbidden)
    namespace = dict(
        make_diagnostics=diag.make_diagnostics, model=object(), writer=object(),
        experiment_config=dict(diagnostics=SETTINGS), benchmark=dict(enabled=True),
        nsys_profile=False, dist=SimpleNamespace(get_rank=lambda: 0))
    exec(compile(ast.Module(body=[setup], type_ignores=[]), train.__file__, "exec"),
         namespace)
    assert namespace["tb_diagnostics"] is None
    calls = {
        ast.unparse(node.func): node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)}
    assert calls["dist.all_reduce"] < calls["tb_diagnostics.before_optimizers"] < calls["opt.step"]
    assert calls["opt.step"] < calls["tb_diagnostics.after_optimizers"] < calls["model.zero_grad"]


@pytest.mark.parametrize(
    "key,value",
    [("scalar_interval", -1), ("scalar_interval", True),
     ("histogram_interval", 11), ("during_nsys", "yes")])
def test_config_validation(key, value):
    config = load_experiment_config()
    config["diagnostics"][key] = value
    with pytest.raises(ValueError, match="diagnostics"):
        validate_experiment_config(config)
