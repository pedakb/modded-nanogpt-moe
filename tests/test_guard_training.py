"""Real trainer update/cleanup and Vista sequencing, with tiny CPU model/data."""
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import torch

from modded_nanogpt_moe import train
from modded_nanogpt_moe.config import load_experiment_config


def run_tiny_trainer(root, *, failure=False, enabled=True, loss=float("nan"),
                     checkpoint_enabled=True, tensorboard=True, policy_disabled=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config()
    config["run_name"] = root.name
    config["training"].update(total_steps=4, global_batch_tokens=4,
                              sequence_length=2, microbatch_sequences=1)
    config["evaluation"].update(tokens=4, interval=3)
    config["checkpoint"]["interval"] = 3 if checkpoint_enabled else None
    config["divergence_guard"].update(enabled=enabled, grace_updates=1, window=1,
                                        moving_avg_patience=1, raw_loss_patience=1)
    updates, saved, events, closed, destroyed = [], [], [], [], []

    class TinyModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.model_dim = kwargs["model_dim"]
            self.mlp_ratio = kwargs["mlp_ratio"]
            self.hidden_dim = int(self.model_dim * self.mlp_ratio)
            self.moe_parameter_layout = kwargs["moe_parameter_layout"]

        def cuda(self):
            return self

        def compile(self, **kwargs):
            pass

        def forward(self, inputs, targets):
            if failure:
                raise RuntimeError("injected genuine failure")
            value = loss if len(updates) >= 1 else 4.
            return (self.weight * 0 + value) * inputs.numel()

    class Loader:
        shard_identities = []

        def __next__(self):
            return torch.zeros(2, 2), torch.zeros(2, 2)

    class Writer:
        def __init__(self, **kwargs):
            pass

        def add_scalar(self, tag, value, step):
            events.append((tag, float(value), step))

        def add_text(self, tag, value, step):
            events.append((tag, value, step))

        def flush(self):
            events.append(("flush",))

        def close(self):
            closed.append(True)

    def optimizers(model, settings):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.param_groups[0]["initial_lr"] = 0.1
        optimizer.register_step_post_hook(lambda opt, args, kwargs: updates.append(opt.param_groups[0]["lr"]))
        return [optimizer]

    def checkpoint(**kwargs):
        assert all(p.grad is None for p in kwargs["model"].parameters())
        return {"completed_updates": kwargs["completed_updates"]}

    def save(payload, directory):
        saved.append(payload)
        return Path(directory) / "synthetic.pt"

    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        for key in ("RESUME_CHECKPOINT", "STOP_AFTER_COMPLETED_UPDATES", "CHECKPOINT_DIR",
                    "NSYS_PROFILE", "REPRO_DIAGNOSTICS_DIR", "TRAINING_BENCHMARK"):
            patch.delenv(key, raising=False)
        for key, value in dict(LOCAL_RANK="0", TB_ROOT=str(root / "tb") if tensorboard else "",
                               CHECKPOINT_ROOT=str(root / "checkpoints"),
                               CHECKPOINT_POLICY_DISABLED="1" if policy_disabled else "0").items():
            patch.setenv(key, value)
        patch.setattr(train, "parse_train_args", lambda argv: (config, None))
        patch.setattr(train, "read_source_snapshot", lambda: "synthetic")
        patch.setattr(train, "collect_environment_metadata", lambda: {})
        for name in ("set_device", "memory_allocated", "memory_reserved",
                     "max_memory_allocated", "max_memory_reserved"):
            patch.setattr(torch.cuda, name, lambda *args, **kwargs: 0)
        patch.setattr(torch.cuda, "get_device_name", lambda *args: "CPU test")
        for name in ("init_process_group", "barrier", "broadcast", "all_reduce"):
            patch.setattr(train.dist, name, lambda *args, **kwargs: None)
        patch.setattr(train.dist, "destroy_process_group", lambda: destroyed.append(True))
        patch.setattr(train.dist, "get_world_size", lambda: 1)
        patch.setattr(train.dist, "get_rank", lambda: 0)
        patch.setattr(train, "GPT", TinyModel)
        patch.setattr(train, "initialize_model_parameters", lambda model: None)
        patch.setattr(train, "build_optimizers", optimizers)
        patch.setattr(train, "distributed_data_generator", lambda *args, **kwargs: Loader())
        patch.setattr(train, "make_training_checkpoint", checkpoint)
        patch.setattr(train, "atomic_save_checkpoint", save)
        patch.setattr(train, "make_diagnostics", lambda *args, **kwargs: None)
        from torch.utils import tensorboard as tensorboard_module
        patch.setattr(tensorboard_module, "SummaryWriter", Writer)
        train.main(["train"])
    return updates, saved, events, closed, destroyed


@pytest.mark.parametrize("loss,reason", [(float("nan"), "nonfinite_loss"),
                                          (float("inf"), "nonfinite_loss"),
                                          (7., "moving_average_deterioration")])
@pytest.mark.parametrize("checkpoint_enabled,tensorboard,policy_disabled", [
    (True, True, False), (False, False, False), (True, False, True)])
def test_guard_real_trainer_clean_stop(tmp_path, capsys, loss, reason,
                                         checkpoint_enabled, tensorboard, policy_disabled):
    updates, saved, events, closed, destroyed = run_tiny_trainer(
        tmp_path, loss=loss, checkpoint_enabled=checkpoint_enabled, tensorboard=tensorboard,
        policy_disabled=policy_disabled)
    assert len(updates) == 2
    assert updates[1] == pytest.approx(0.1)  # Four-step schedule is still in its plateau.
    assert destroyed == [True]
    assert closed == ([True] if tensorboard else [])
    assert [p["completed_updates"] for p in saved] == (
        [2] if checkpoint_enabled and not policy_disabled else [])
    if saved:
        assert saved[0]["divergence_stop"]["reason"] == reason
        assert saved[0]["divergence_stop"]["update"] == 2
        assert "divergence_guard_state" in saved[0]
    assert not any(event[0].startswith("guard/") for event in events)
    if tensorboard:
        assert [event[2] for event in events if event[0] == "metric/loss/train"] == [1, 2]
        assert any(event[0] == "metric/loss/val" for event in events)
        assert events[-1] == ("flush",)
    assert f"[DIVERGENCE_STOP] update=2 reason={reason}" in capsys.readouterr().out


def test_disabled_guard_keeps_normal_completion(tmp_path):
    updates, saved, _, _, destroyed = run_tiny_trainer(tmp_path, enabled=False, loss=7.)
    assert len(updates) == 4
    assert [p["completed_updates"] for p in saved] == [3, 4]
    assert all("divergence_stop" not in p for p in saved)
    assert destroyed == [True]


def test_stable_guard_preserves_update_schedule_and_checkpoint_cadence(tmp_path):
    ordinary = run_tiny_trainer(tmp_path / "ordinary", enabled=False, loss=3.)
    monitored = run_tiny_trainer(tmp_path / "monitored", enabled=True, loss=3.)
    assert ordinary[0] == monitored[0]
    assert [p["completed_updates"] for p in monitored[1]] == [3, 4]
    assert all("divergence_stop" not in p for p in monitored[1])


@pytest.mark.parametrize("failure", [False, True])
def test_actual_vista_sequential_loop(tmp_path, failure):
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts/vista/train.sh").read_text()
    # Exercise the actual shared current-node/Slurm worker loop; omit machine
    # setup and preflight. The uv shim runs real main() with tiny CPU fixtures.
    loop = source[source.index('for index in "${!config_paths[@]}"; do',
                               source.index('unset MOE_GMM_IMPLEMENTATION')):]
    helper = tmp_path / "runner.py"
    helper.write_text(
        "import sys\nfrom pathlib import Path\n"
        "from test_guard_training import run_tiny_trainer\n"
        "name=sys.argv[-1]\n"
        f"run_tiny_trainer(Path({str(tmp_path)!r})/name, "
        f"failure=({failure!r} and name=='A'), enabled=(name=='A'), loss=7.)\n")
    uv = tmp_path / "uv"
    uv.write_text(f"#!/bin/bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(helper))} \"$@\"\n")
    uv.chmod(0o755)
    script = ('set -euo pipefail\nconfig_paths=(A B)\nrun_names=(A B)\n'
              'steps=""\nsmoke=0\nbenchmark_worker=0\ncheckpoint_interval=""\nresume_checkpoint=""\n'
              'repo_root="$PWD"\n' + loop)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               PYTHONPATH=f"{repo / 'tests'}:{repo}", STOCKYARD=str(tmp_path))
    result = subprocess.run(["bash", "-c", script], env=env, text=True,
                            capture_output=True, timeout=120)
    if failure:
        assert result.returncode != 0
        assert "injected genuine failure" in result.stderr
        assert "run_name=B" not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert "[DIVERGENCE_STOP] update=2" in result.stdout
        assert "run_name=B" in result.stdout
        assert "Suite complete:" in result.stdout
