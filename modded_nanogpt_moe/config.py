"""Minimal TOML experiment configuration for the active trainer."""

import argparse
import copy
import math
import os
import tomllib
from pathlib import Path


DEFAULT_CONFIG = {
    "run_name": None,
    "num_trials": 1,
    "seed": None,
    "diagnostics": {
        "scalar_interval": 10,
        "histogram_interval": 0,
        "during_nsys": False,
    },
    "model": {
        "vocab_size": 50304,
        "num_layers": 12,
        "model_dim": 768,
        "mlp_type": "dense",
        "mlp_ratio": 4.0,
        "num_experts": 1,
        "top_k": 1,
        "normalize_topk": True,
        "moe_backend": "loop",
        "moe_parameter_layout": "modulelist",
    },
    "training": {
        "sequence_length": 1024,
        "global_batch_tokens": 524288,
        "microbatch_sequences": 64,
        "validation_tokens": 10485760,
        "total_steps": 3250,
        "cooldown_fraction": 0.7,
        "training_shard_pattern": "data/fineweb10B/fineweb_train_*.bin",
        "validation_shard_pattern": "data/fineweb10B/fineweb_val_*.bin",
    },
    "optimizers": {
        "adamw": {
            "group_lrs": [0.7, 0.004, 0.015],
            "betas": [0.8, 0.95],
            "eps": 1e-10,
            "weight_decay": 0.001,
            "fused": True,
        },
        "muon": {
            "lr": 0.025,
            "weight_decay": 0.05,
            "mu": 0.95,
        },
    },
}


def _merge_known(defaults, supplied, location=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{location or 'config'} must be a TOML table")
    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        prefix = f"{location}." if location else ""
        raise ValueError(
            "unknown configuration field(s): "
            + ", ".join(prefix + key for key in unknown)
        )
    merged = copy.deepcopy(defaults)
    for key, value in supplied.items():
        child_location = f"{location}.{key}" if location else key
        if isinstance(defaults[key], dict):
            merged[key] = _merge_known(defaults[key], value, child_location)
        else:
            merged[key] = value
    return merged


def _positive_int(config, section, key):
    value = config[section][key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{section}.{key} must be a positive integer")


def validate_experiment_config(config, require_run_name=False):
    diagnostics = config["diagnostics"]
    for key in ("scalar_interval", "histogram_interval"):
        value = diagnostics[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"diagnostics.{key} must be a nonnegative integer")
    if not isinstance(diagnostics["during_nsys"], bool):
        raise ValueError("diagnostics.during_nsys must be a boolean")
    if diagnostics["histogram_interval"] and (
            not diagnostics["scalar_interval"]
            or diagnostics["histogram_interval"] % diagnostics["scalar_interval"]):
        raise ValueError("diagnostics.histogram_interval must be a multiple of scalar_interval")
    run_name = config["run_name"]
    if require_run_name and not run_name:
        raise ValueError("run_name is required in an experiment config")
    if run_name is not None:
        if not isinstance(run_name, str) or not run_name.strip():
            raise ValueError("run_name must be a nonempty string")
        if (Path(run_name).name != run_name or run_name in (".", "..")
                or "/" in run_name or "\\" in run_name):
            raise ValueError("run_name must be a single directory-safe name")
    if (isinstance(config["num_trials"], bool)
            or not isinstance(config["num_trials"], int)
            or config["num_trials"] <= 0):
        raise ValueError("num_trials must be a positive integer")
    if config["seed"] is not None and (
            isinstance(config["seed"], bool) or not isinstance(config["seed"], int)):
        raise ValueError("seed must be an integer")

    for key in ("vocab_size", "num_layers", "model_dim", "num_experts", "top_k"):
        _positive_int(config, "model", key)
    for key in (
        "sequence_length", "global_batch_tokens", "microbatch_sequences",
        "validation_tokens", "total_steps",
    ):
        _positive_int(config, "training", key)
    if config["model"]["mlp_type"] not in ("dense", "moe"):
        raise ValueError("model.mlp_type must be 'dense' or 'moe'")
    if config["model"]["moe_backend"] not in ("loop", "grouped_gemm"):
        raise ValueError("model.moe_backend must be 'loop' or 'grouped_gemm'")
    layout = config["model"]["moe_parameter_layout"]
    if layout not in ("modulelist", "packed"):
        raise ValueError("model.moe_parameter_layout must be 'modulelist' or 'packed'")
    if layout == "packed" and (
            config["model"]["mlp_type"] != "moe"
            or config["model"]["moe_backend"] != "grouped_gemm"):
        raise ValueError("packed parameters require grouped_gemm MoE")
    if not isinstance(config["model"]["normalize_topk"], bool):
        raise ValueError("model.normalize_topk must be a boolean")
    ratio = config["model"]["mlp_ratio"]
    hidden_dim = config["model"]["model_dim"] * ratio
    if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio) or ratio <= 0
            or not math.isfinite(hidden_dim) or not float(hidden_dim).is_integer()):
        raise ValueError(
            "model.mlp_ratio must be positive and produce an integer hidden width")
    if config["model"]["mlp_type"] == "moe" and (
            config["model"]["top_k"] > config["model"]["num_experts"]):
        raise ValueError("model.top_k cannot exceed model.num_experts")
    cooldown = config["training"]["cooldown_fraction"]
    if (isinstance(cooldown, bool) or not isinstance(cooldown, (int, float))
            or not math.isfinite(cooldown) or not 0 < cooldown <= 1):
        raise ValueError("training.cooldown_fraction must be in (0, 1]")
    for key in ("training_shard_pattern", "validation_shard_pattern"):
        if not isinstance(config["training"][key], str) or not config["training"][key]:
            raise ValueError(f"training.{key} must be a nonempty string")

    adamw = config["optimizers"]["adamw"]
    muon = config["optimizers"]["muon"]
    if len(adamw["group_lrs"]) != 3:
        raise ValueError("optimizers.adamw.group_lrs must contain exactly three values")
    if len(adamw["betas"]) != 2:
        raise ValueError("optimizers.adamw.betas must contain exactly two values")
    if not isinstance(adamw["fused"], bool):
        raise ValueError("optimizers.adamw.fused must be a boolean")
    numeric_optimizer_values = (
        *(('optimizers.adamw.group_lrs', value) for value in adamw["group_lrs"]),
        *(('optimizers.adamw.betas', value) for value in adamw["betas"]),
        ("optimizers.adamw.eps", adamw["eps"]),
        ("optimizers.adamw.weight_decay", adamw["weight_decay"]),
        ("optimizers.muon.lr", muon["lr"]),
        ("optimizers.muon.weight_decay", muon["weight_decay"]),
        ("optimizers.muon.mu", muon["mu"]),
    )
    for name, value in numeric_optimizer_values:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"{name} values must be finite numbers")
    return config


def load_experiment_config(path=None):
    if path is None:
        return validate_experiment_config(copy.deepcopy(DEFAULT_CONFIG))
    config_path = Path(path)
    with config_path.open("rb") as file:
        supplied = tomllib.load(file)
    return validate_experiment_config(
        _merge_known(DEFAULT_CONFIG, supplied), require_run_name=True)


def parse_train_args(argv):
    parser = argparse.ArgumentParser(description="Train the active dense/MoE model")
    parser.add_argument("num_trials", nargs="?", type=int)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    if args.num_trials is not None:
        if args.num_trials <= 0:
            parser.error("num_trials must be positive")
        config["num_trials"] = args.num_trials
    return apply_environment_overrides(config), args.config


def apply_environment_overrides(config):
    config = copy.deepcopy(config)
    overrides = (
        ("SEED_OVERRIDE", None, "seed", int),
        ("MBS_OVERRIDE", "training", "microbatch_sequences", int),
        ("TRAIN_STEPS_OVERRIDE", "training", "total_steps", int),
        ("MLP_TYPE_OVERRIDE", "model", "mlp_type", str),
        ("MLP_RATIO_OVERRIDE", "model", "mlp_ratio", float),
        ("NUM_EXPERTS_OVERRIDE", "model", "num_experts", int),
        ("TOP_K_OVERRIDE", "model", "top_k", int),
        ("MOE_BACKEND_OVERRIDE", "model", "moe_backend", str),
    )
    for environment_name, section, key, convert in overrides:
        value = os.environ.get(environment_name, "")
        if value:
            if section is None:
                config[key] = convert(value)
            else:
                config[section][key] = convert(value)
    return validate_experiment_config(config, require_run_name=config["run_name"] is not None)
