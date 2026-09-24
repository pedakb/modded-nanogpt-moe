from pathlib import Path

import numpy as np
import pytest
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from analysis.grad_em_metrics import (
    load_cv,
    routing_distributions,
    routing_gradients,
    summarize_layer,
    token_metrics,
)
from analysis.tb_utils import (
    align_scalar_series,
    list_scalar_tags,
    load_all_scalars,
    load_scalar_tag,
)


def _write_scalars(directory: Path, values):
    writer = EventFileWriter(str(directory))
    try:
        for tag, step, value, wall_time in values:
            writer.add_event(event_pb2.Event(
                wall_time=wall_time,
                step=step,
                summary=summary_pb2.Summary(value=[
                    summary_pb2.Summary.Value(tag=tag, simple_value=value),
                ]),
            ))
        writer.flush()
    finally:
        writer.close()


def test_tensorboard_scalar_loading_filtering_and_missing_tag(tmp_path):
    trial = tmp_path / "run/trial_0"
    _write_scalars(trial, [
        ("metric/loss/train", 1, 2.5, 10.0),
        ("metric/loss/train", 2, 2.0, 11.0),
        ("router/l00/load/cv", 25, 0.3, 12.0),
    ])

    assert list_scalar_tags(tmp_path / "run", substring="loss") == ["metric/loss/train"]
    assert list_scalar_tags(tmp_path / "run", regex=r"^router/.*/cv$") == [
        "router/l00/load/cv"]
    records = load_scalar_tag(tmp_path / "run", "metric/loss/train")
    assert [(record["step"], record["value"]) for record in records] == [
        (1, 2.5), (2, 2.0)]
    assert load_scalar_tag(tmp_path / "run", "missing") == []
    with pytest.raises(KeyError, match="scalar tag not found"):
        load_scalar_tag(tmp_path / "run", "missing", missing="raise")
    assert {record["tag"] for record in load_all_scalars(
        tmp_path / "run", regex=r"^(metric|router)/")} == {
            "metric/loss/train", "router/l00/load/cv"}


def test_exact_step_alignment_is_sparse_and_uses_latest_duplicate():
    left = [
        {"step": 0, "value": 1.0, "wall_time": 1.0},
        {"step": 2, "value": 2.0, "wall_time": 2.0},
        {"step": 2, "value": 2.5, "wall_time": 3.0},
    ]
    right = [
        {"step": 1, "value": 10.0, "wall_time": 1.0},
        {"step": 2, "value": 20.0, "wall_time": 1.0},
    ]
    assert align_scalar_series({"left": left, "right": right}) == [
        {"step": 2, "left": 2.5, "right": 20.0}]
    assert align_scalar_series({"left": left, "right": right}, how="outer") == [
        {"step": 0, "left": 1.0, "right": None},
        {"step": 1, "left": None, "right": 10.0},
        {"step": 2, "left": 2.5, "right": 20.0},
    ]


def test_metric_formulas_and_balanced_load():
    logits = np.zeros((2, 2))
    values = np.array([[0.0, 2.0], [2.0, 0.0]])
    experts = np.array([[0, 1], [0, 1]])
    eta = 0.5
    metrics = token_metrics(logits, values, eta)
    a, q = routing_distributions(logits, values, eta)
    g_bp, g_ge = routing_gradients(a, q, values, eta)

    np.testing.assert_allclose(a, 0.5)
    np.testing.assert_allclose(metrics["q"], q)
    np.testing.assert_allclose(metrics["g_bp"], g_bp)
    np.testing.assert_allclose(metrics["g_ge"], g_ge)
    np.testing.assert_allclose(metrics["entropy_norm"], 1.0)
    np.testing.assert_allclose(metrics["selected_logit_std"], 0.0)
    np.testing.assert_allclose(metrics["delta_v"], 2.0)
    np.testing.assert_allclose(metrics["d"], 1.0)
    np.testing.assert_allclose(metrics["sigma_v"], 1.0)
    assert load_cv(experts, 2) == (0.0, 0)

    summary = summarize_layer(
        logits, values, experts, eta=eta, num_experts=2, chunk_size=1)
    assert summary["entropy_norm_mean"] == pytest.approx(1.0)
    assert summary["eta_dv_p99"] == pytest.approx(1.0)
    assert summary["load_cv"] == 0.0
    assert summary["r_valid_count"] == 2


def test_metrics_are_invariant_to_common_v_translation():
    logits = np.array([[0.2, -0.3, 1.1], [0.7, 0.0, -0.2]])
    values = np.array([[1.0, -2.0, 0.5], [4.0, 3.0, -1.0]])
    baseline = token_metrics(logits, values, eta=0.03)
    shifted = token_metrics(logits, values + 1e5, eta=0.03)
    for name in (
        "a", "q", "g_bp", "g_ge", "delta_v", "eta_delta_v", "tv",
        "c_bp", "c_ge", "amplification_ratio", "d", "sigma_v",
        "min_v_probability", "current_top_v_gap",
    ):
        np.testing.assert_allclose(shifted[name], baseline[name], rtol=2e-10, atol=2e-10)
    np.testing.assert_array_equal(
        shifted["argmax_a_ne_argmin_v"], baseline["argmax_a_ne_argmin_v"])


def test_small_eta_grad_em_direction_approaches_bp():
    logits = np.array([[0.1, 0.3, -0.4], [1.0, -0.2, 0.5]])
    values = np.array([[2.0, -1.0, 0.2], [-0.5, 0.7, 1.3]])
    errors = []
    for eta in (1e-2, 1e-4):
        a, q = routing_distributions(logits, values, eta)
        g_bp, g_ge = routing_gradients(a, q, values, eta)
        errors.append(np.linalg.norm(g_ge - g_bp))
    assert errors[1] < errors[0] / 50
    assert errors[1] < 1e-4


def test_near_zero_cbp_produces_nan_ratio_not_infinity():
    metrics = token_metrics(
        np.array([[0.0, 1.0, -1.0]]),
        np.array([[3.0, 3.0, 3.0]]),
        eta=0.1,
    )
    assert metrics["c_bp"][0] == pytest.approx(0.0, abs=1e-30)
    assert np.isnan(metrics["amplification_ratio"][0])
    assert not np.isinf(metrics["amplification_ratio"]).any()
