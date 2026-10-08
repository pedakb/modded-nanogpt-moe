"""Inventory and invariants for the organized production experiment configs."""

import copy
import os
from pathlib import Path
import subprocess

import pytest

from modded_nanogpt_moe.config import load_experiment_config
from modded_nanogpt_moe.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    validate_checkpoint_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
MATRIX = {
    "baselines/dense.toml": ("dense", None, None, 3072, 0, 4.0, 56_669_184),
    "baselines/moe_e64k8.toml": ("moe", 64, 8, 384, 0, 4.0, 453_869_568),
    "moe_architectures/moe_e64k8_shared.toml": (
        "moe", 64, 8, 336, 1, 3.9375, 403_416_000),
    "moe_architectures/moe_e128k8.toml": (
        "moe", 128, 8, 384, 0, 4.0, 907_739_136),
    "moe_architectures/moe_e128k8_shared.toml": (
        "moe", 128, 8, 336, 1, 3.9375, 800_625_600),
    "moe_architectures/moe_e256k6.toml": (
        "moe", 256, 6, 512, 0, 4.0, 2_419_851_264),
    "moe_architectures/moe_e256k6_shared.toml": (
        "moe", 256, 6, 432, 1, 3.9375, 2_050_095_168),
    "moe_architectures/moe_e512k10.toml": (
        "moe", 512, 10, 304, 0, 3.9583333333333335, 2_875_490_304),
    "moe_architectures/moe_e512k10_shared.toml": (
        "moe", 512, 10, 288, 1, 4.125, 2_729_718_144),
}


def test_every_repository_config_loads_and_run_names_are_unique():
    paths = sorted(CONFIG_ROOT.rglob("*.toml"))
    assert len(paths) == 47
    run_names = {}
    for path in paths:
        config = load_experiment_config(path)
        assert config["run_name"] not in run_names, (
            f"duplicate run_name {config['run_name']!r}: "
            f"{run_names.get(config['run_name'])} and {path}")
        run_names[config["run_name"]] = path


def test_config_root_and_archive_layout_are_complete():
    assert sorted(path.name for path in CONFIG_ROOT.iterdir() if path.is_file()) == [
        "README.md"]
    expected_counts = {
        "legacy": 5,
        "grad_em": 29,
        "router_ablations": 2,
        "prototypes": 2,
    }
    archive = CONFIG_ROOT / "archive"
    assert {
        directory.name: len(list(directory.glob("*.toml")))
        for directory in archive.iterdir() if directory.is_dir()
    } == expected_counts


def test_archived_configs_retain_resolved_checkpoint_identity():
    for path in sorted((CONFIG_ROOT / "archive").rglob("*.toml")):
        config = load_experiment_config(path)
        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "resolved_config": copy.deepcopy(config),
        }
        validate_checkpoint_config(checkpoint, config)


@pytest.mark.parametrize("relative_path,expected", MATRIX.items())
def test_bp_matrix_geometry_and_shared_expert_contract(relative_path, expected):
    mlp_type, experts, top_k, hidden_dim, shared_count, expansion, parameter_count = expected
    config = load_experiment_config(CONFIG_ROOT / relative_path)
    model = config["model"]
    assert model["model_dim"] == 768
    assert model["num_layers"] == 12
    assert model["mlp_type"] == mlp_type
    assert model["moe_backward"] == "standard"
    resolved_hidden = int(model["model_dim"] * model["mlp_ratio"])
    assert resolved_hidden == hidden_dim

    if mlp_type == "dense":
        assert model["num_shared_experts"] == 0
        active_expansion = model["mlp_ratio"]
        counted_experts = 1
    else:
        assert model["num_experts"] == experts
        assert model["top_k"] == top_k
        assert model["num_shared_experts"] == shared_count
        assert model["moe_backend"] == "grouped_gemm"
        assert model["moe_parameter_layout"] == "packed"
        assert model["normalize_topk"] is True
        active_expansion = (
            top_k * model["mlp_ratio"]
            + shared_count * model["shared_expert_ratio"])
        counted_experts = experts + shared_count
        if shared_count:
            assert int(model["model_dim"] * model["shared_expert_ratio"]) == hidden_dim

    assert active_expansion == pytest.approx(expansion)
    per_expert = 2 * 768 * hidden_dim + hidden_dim + 768
    assert counted_experts * per_expert * 12 == parameter_count


def test_bp_matrix_shares_scientific_training_policy():
    reference = load_experiment_config(CONFIG_ROOT / "baselines" / "dense.toml")
    scheduled_tokens = 3250 * 524288
    assert scheduled_tokens == 1_703_936_000
    run_names = set()
    for relative_path in MATRIX:
        config = load_experiment_config(CONFIG_ROOT / relative_path)
        assert config["run_name"] not in run_names
        run_names.add(config["run_name"])
        assert config["num_trials"] == reference["num_trials"] == 1
        assert config["seed"] == reference["seed"] == 1234
        assert config["training"] == reference["training"]
        assert config["evaluation"] == reference["evaluation"]
        assert config["optimizers"] == reference["optimizers"]
        assert config["diagnostics"] == reference["diagnostics"]
        assert config["checkpoint"] == reference["checkpoint"] == {"interval": 250}


@pytest.mark.parametrize("relative_path", MATRIX)
def test_moved_configs_retain_resolved_checkpoint_identity(relative_path):
    config = load_experiment_config(CONFIG_ROOT / relative_path)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "resolved_config": copy.deepcopy(config),
    }
    validate_checkpoint_config(checkpoint, config)


def test_archive_preserves_superseded_prototype_identities():
    archive = CONFIG_ROOT / "archive" / "prototypes"
    unshared = load_experiment_config(archive / "moe_e256k6_r0.5.toml")
    shared = load_experiment_config(archive / "moe_e256k6_r0.5_shared_r0.5.toml")
    assert unshared["run_name"] == "moe-e256k6-r0.5"
    assert shared["run_name"] == "moe-e256k6-r0.5-shared-r0.5"
    assert unshared["model"]["top_k"] * unshared["model"]["mlp_ratio"] == 3.0
    assert ((shared["model"]["top_k"] + 1)
            * shared["model"]["mlp_ratio"] == 3.5)


def test_vista_submit_accepts_nested_config_paths(tmp_path):
    fake_sbatch = tmp_path / "sbatch"
    capture = tmp_path / "submission.txt"
    fake_sbatch.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE"\nexit 23\n')
    fake_sbatch.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        CAPTURE=str(capture),
    )
    paths = [
        "configs/baselines/dense.toml",
        "configs/baselines/moe_e64k8.toml",
    ]
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "vista" / "train.sh"), "--submit", *paths],
        cwd=ROOT, env=environment, capture_output=True, text=True)
    assert result.returncode == 23, result.stderr
    submitted = capture.read_text().splitlines()
    assert submitted[-3] == "--"
    assert submitted[-2:] == [str(ROOT / path) for path in paths]


def test_vista_default_is_established_moe_baseline():
    launcher = (ROOT / "scripts" / "vista" / "train.sh").read_text()
    assert "config_arguments=(configs/baselines/moe_e64k8.toml)" in launcher
