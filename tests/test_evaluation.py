import copy
import tomllib
from pathlib import Path

import pytest

from modded_nanogpt_moe.config import (
    load_experiment_config,
    validate_experiment_config,
)
from modded_nanogpt_moe.train import should_evaluate


PRODUCTION_EVALUATION = {
    "tokens": 10485760,
    "interval": 125,
    "final_fraction": 0.10,
    "final_interval": 25,
}


def evaluation_steps(total_steps, evaluation=PRODUCTION_EVALUATION, start=0):
    return [
        step for step in range(start, total_steps + 1)
        if should_evaluate(step, total_steps, evaluation)
    ]


@pytest.mark.parametrize(
    "filename",
    ["dense_baseline.toml", "moe_e8k2_r2.toml", "moe_e64k8_r0.5.toml"],
)
def test_production_configs_parse_evaluation_section(filename):
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / filename
    raw = tomllib.loads(path.read_text())
    config = load_experiment_config(path)

    assert raw["evaluation"] == PRODUCTION_EVALUATION
    assert config["evaluation"] == PRODUCTION_EVALUATION
    assert "validation_tokens" not in raw["training"]
    assert "validation_tokens" not in config["training"]


def test_legacy_training_validation_tokens_is_not_a_second_source_of_truth(tmp_path):
    path = tmp_path / "legacy-validation-tokens.toml"
    path.write_text(
        'run_name = "legacy-validation-tokens"\n'
        '[training]\n'
        'validation_tokens = 10485760\n'
    )

    with pytest.raises(
            ValueError,
            match=r"unknown configuration field\(s\): training\.validation_tokens"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("tokens", 0, "evaluation.tokens must be a positive integer"),
        ("interval", 0, "evaluation.interval must be a positive integer"),
        ("final_interval", -1,
         "evaluation.final_interval must be a positive integer"),
        ("final_fraction", -0.01,
         "evaluation.final_fraction must be in [0, 1]"),
        ("final_fraction", 1.01,
         "evaluation.final_fraction must be in [0, 1]"),
    ],
)
def test_invalid_evaluation_config_is_rejected(key, value, message):
    config = copy.deepcopy(load_experiment_config())
    config["evaluation"][key] = value

    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        validate_experiment_config(config)


def test_production_schedule_regular_and_final_phase_boundaries():
    assert should_evaluate(0, 3250, PRODUCTION_EVALUATION)
    assert should_evaluate(125, 3250, PRODUCTION_EVALUATION)
    assert should_evaluate(2875, 3250, PRODUCTION_EVALUATION)
    assert not should_evaluate(2900, 3250, PRODUCTION_EVALUATION)
    assert should_evaluate(2925, 3250, PRODUCTION_EVALUATION)
    assert should_evaluate(2950, 3250, PRODUCTION_EVALUATION)
    assert should_evaluate(3250, 3250, PRODUCTION_EVALUATION)


def test_final_step_is_forced_when_not_on_either_cadence():
    evaluation = dict(PRODUCTION_EVALUATION, interval=128, final_interval=32)
    assert should_evaluate(3250, 3250, evaluation)


def test_3250_step_schedule_matches_existing_behavior_without_duplicates():
    actual = evaluation_steps(3250)
    expected = list(range(0, 2876, 125)) + list(range(2925, 3251, 25))

    assert actual == expected
    assert len(actual) == len(set(actual))


def test_resume_uses_restored_global_step_without_extra_evaluation():
    complete_schedule = evaluation_steps(3250)

    assert not should_evaluate(2930, 3250, PRODUCTION_EVALUATION)
    assert evaluation_steps(3250, start=2930) == [
        step for step in complete_schedule if step >= 2930]
    assert evaluation_steps(3250, start=2925)[0] == 2925


def test_diagnostics_interval_does_not_affect_evaluation_schedule():
    config_a = load_experiment_config()
    config_b = copy.deepcopy(config_a)
    config_a["diagnostics"]["scalar_interval"] = 1
    config_b["diagnostics"]["scalar_interval"] = 1000

    assert evaluation_steps(3250, config_a["evaluation"]) == evaluation_steps(
        3250, config_b["evaluation"])
