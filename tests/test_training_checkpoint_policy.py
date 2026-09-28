"""Exercise the real trainer's save/stop decisions with a tiny CPU test model."""
from pathlib import Path
import json

import pytest
import torch

from modded_nanogpt_moe import train
from modded_nanogpt_moe.config import load_experiment_config


@pytest.mark.parametrize("disabled,stop,explicit", [
    (True, None, False), (True, 2, False), (True, 3, False),
    (True, None, True), (True, 2, True),
    (False, None, False), (False, 2, False),
])
def test_trainer_checkpoint_policy_covers_periodic_final_and_early_stop(
        tmp_path, monkeypatch, capsys, disabled, stop, explicit):
    config = load_experiment_config()
    config["run_name"] = "checkpoint-policy"
    config["training"].update(total_steps=3, global_batch_tokens=4,
                              sequence_length=2, microbatch_sequences=1)
    config["evaluation"].update(tokens=4, interval=1)
    config["checkpoint"]["interval"] = 1  # Every completed update would save.
    monkeypatch.setattr(train, "parse_train_args", lambda argv: (config, None))
    monkeypatch.setattr(train, "read_source_snapshot", lambda: "test source")
    monkeypatch.setattr(train, "collect_environment_metadata", lambda: {})
    monkeypatch.chdir(tmp_path)
    for name in ("RESUME_CHECKPOINT", "STOP_AFTER_COMPLETED_UPDATES", "CHECKPOINT_DIR",
                 "NSYS_PROFILE", "REPRO_DIAGNOSTICS_DIR", "TRAINING_BENCHMARK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TB_ROOT", "")
    monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("CHECKPOINT_POLICY_DISABLED", "1" if disabled else "0")
    if stop is not None:
        monkeypatch.setenv("STOP_AFTER_COMPLETED_UPDATES", str(stop))
    if explicit:
        monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "explicit"))

    # Replace hardware/data/model work, but execute main() and its update loop,
    # runtime metadata, checkpoint policy, and early-stop control unchanged.
    for name in ("set_device", "memory_allocated", "memory_reserved",
                 "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "test CPU")
    for name in ("init_process_group", "barrier", "broadcast", "all_reduce", "destroy_process_group"):
        monkeypatch.setattr(train.dist, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(train.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(train.dist, "get_rank", lambda: 0)

    updates = []

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
            return self.weight.square() * inputs.numel()

    class Loader:
        shard_identities = []

        def __next__(self):
            return torch.zeros(2, 2), torch.zeros(2, 2)

    def optimizers(model, settings):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.param_groups[0]["initial_lr"] = 0.1
        optimizer.register_step_post_hook(lambda opt, args, kwargs: updates.append(opt.param_groups[0]["lr"]))
        return [optimizer]

    monkeypatch.setattr(train, "GPT", TinyModel)
    monkeypatch.setattr(train, "initialize_model_parameters", lambda model: None)
    monkeypatch.setattr(train, "build_optimizers", optimizers)
    monkeypatch.setattr(train, "distributed_data_generator", lambda *args, **kwargs: Loader())
    saved = []

    def checkpoint(**kwargs):
        assert not disabled, "trainer attempted to construct a disabled checkpoint"
        return {"completed_updates": kwargs["completed_updates"]}

    def save(payload, directory):
        assert not disabled, "trainer attempted a disabled checkpoint write"
        saved.append(payload["completed_updates"])
        path = Path(directory) / f"step_{saved[-1]:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test checkpoint")
        return path

    monkeypatch.setattr(train, "make_training_checkpoint", checkpoint)
    monkeypatch.setattr(train, "atomic_save_checkpoint", save)
    train.main(["train"])

    completed = stop or 3
    assert len(updates) == completed
    # The second update still uses the three-update horizon, even at early stop.
    assert updates[1] == pytest.approx(0.1 * (1 - 1/3) / 0.7)
    assert config["training"]["total_steps"] == 3
    assert saved == ([] if disabled else list(range(1, completed + 1)))
    assert (tmp_path / "checkpoints").exists() == (not disabled)
    assert not (tmp_path / "explicit").exists()
    output = capsys.readouterr().out.split("Resolved runtime settings:\n", 1)[1]
    runtime = json.JSONDecoder().raw_decode(output)[0]["runtime"]
    assert runtime["checkpoint"]["enabled"] == (not disabled)
    assert runtime["tensorboard"]["enabled"] is False
    if disabled:
        assert runtime["checkpoint"]["directory"] is None
        assert runtime["checkpoint"]["interval"] is None
