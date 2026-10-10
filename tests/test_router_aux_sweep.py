"""Scientific settings for the isolated E256/K6 shared BP auxiliary sweep."""
import copy
from pathlib import Path

import pytest

from modded_nanogpt_moe.config import load_experiment_config
from modded_nanogpt_moe.divergence_guard import DEFAULTS


ROOT = Path(__file__).parents[1]
COEFFICIENTS = ("0.001", "0.003", "0.01", "0.03", "0.1")


@pytest.mark.parametrize("coefficient", COEFFICIENTS)
def test_auxiliary_sweep_preserves_bp_baseline(coefficient):
    baseline = load_experiment_config(ROOT / "configs/moe_architectures/moe_e256k6_shared.toml")
    config = load_experiment_config(
        ROOT / f"configs/router_regularizers/moe_e256k6_shared_bp_aux{coefficient}_z0.toml")
    assert config["model"]["moe_backward"] == "standard"
    assert config["model"]["router_aux_loss_coef"] == float(coefficient)
    assert config["model"]["router_z_loss_coef"] == 0
    assert config["divergence_guard"] == dict(DEFAULTS, enabled=True)
    assert config["model"]["model_dim"] * config["model"]["mlp_ratio"] == 432
    assert config["model"]["model_dim"] * config["model"]["shared_expert_ratio"] == 432
    assert config["training"]["total_steps"] == 3250
    assert config["training"]["global_batch_tokens"] // (
        config["training"]["microbatch_sequences"] * config["training"]["sequence_length"]) == 8
    normalized = copy.deepcopy(config)
    normalized["run_name"] = baseline["run_name"]
    normalized["model"]["router_aux_loss_coef"] = 0
    normalized["divergence_guard"] = baseline["divergence_guard"]
    assert normalized == baseline


def test_all_run_names_are_unique():
    configs = [load_experiment_config(path) for path in (ROOT / "configs").rglob("*.toml")]
    names = [config["run_name"] for config in configs]
    assert len(names) == len(set(names))
