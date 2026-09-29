import copy
import math

import pytest

from modded_nanogpt_moe.divergence_guard import DEFAULTS, DivergenceGuard
from modded_nanogpt_moe.config import load_experiment_config, validate_experiment_config
from modded_nanogpt_moe.checkpoint import CHECKPOINT_FORMAT_VERSION, validate_checkpoint_config


def guard(**kwargs):
    return DivergenceGuard({**DEFAULTS, "enabled": True, **kwargs})


def replay(detector, losses, start=1):
    for update, loss in enumerate(losses, start):
        stop = detector.observe(update, loss)
        if stop is not None:
            return stop
    return None


def test_disabled_never_triggers():
    assert replay(guard(enabled=False), [math.nan, math.inf, -math.inf] + [100.] * 200) is None


@pytest.mark.parametrize("loss", [math.nan, math.inf, -math.inf])
def test_nonfinite_immediate(loss):
    stop = guard().observe(1, loss)
    assert stop["reason"] == "nonfinite_loss"
    assert stop["update"] == 1


def test_grace_and_full_twenty_update_window():
    detector = guard()
    assert replay(detector, [1.] * 79 + [float(i) for i in range(1, 21)]) is None
    assert detector.moving_avg == 10.5
    assert detector.best_moving_avg is None
    assert detector.observe(100, 21.) is None
    assert detector.moving_avg == 11.5
    assert detector.best_moving_avg == 11.5  # Earlier average 1 is excluded.


def test_window_larger_than_grace_waits_for_full_window():
    detector = guard(grace_updates=0)
    assert replay(detector, [1.] * 19) is None
    assert detector.best_moving_avg is None
    detector.observe(20, 1.)
    assert detector.best_moving_avg == 1.


def test_moving_average_patience_and_reason():
    detector = guard(raw_loss_delta=100.)
    assert replay(detector, [4.] * 100) is None
    # Window rises by .05/update; strict > .5 first occurs at update 111.
    assert replay(detector, [5.] * 19, 101) is None
    assert detector.moving_avg_count == 9
    stop = detector.observe(120, 5.)
    assert stop == dict(update=120, reason="moving_average_deterioration",
                        current_loss=5., moving_avg=5., best_moving_avg=4.)


def test_moving_average_recovery_resets_patience():
    detector = guard(window=1, grace_updates=1, raw_loss_delta=100.)
    detector.observe(1, 4.)
    assert replay(detector, [5.] * 9, 2) is None
    detector.observe(11, 4.)
    assert detector.moving_avg_count == 0
    assert replay(detector, [5.] * 9, 12) is None
    assert detector.observe(21, 5.)["update"] == 21


def test_raw_loss_patience_and_recovery():
    detector = guard(moving_avg_delta=100.)
    replay(detector, [4.] * 100)
    assert replay(detector, [7.] * 2, 101) is None
    assert detector.raw_loss_count == 2
    detector.observe(103, 4.)
    assert detector.raw_loss_count == 0
    assert replay(detector, [7.] * 2, 104) is None
    stop = detector.observe(106, 7.)
    assert stop["reason"] == "raw_loss_spike"
    assert stop["update"] == 106


def test_strict_threshold_and_slow_improvement():
    detector = guard(moving_avg_delta=100.)
    replay(detector, [4.] * 100)
    assert replay(detector, [6.] * 20, 101) is None
    assert replay(guard(), [10. - i / 1000 for i in range(3250)]) is None


def test_resume_preserves_history_and_patience():
    original = guard()
    replay(original, [4.] * 100 + [7.] * 2)
    restored = guard()
    restored.load_state_dict(original.state_dict())
    assert restored.observe(103, 7.) == original.observe(103, 7.)


@pytest.mark.parametrize("key,values", [
    ("enabled", [0, 1, "true"]),
    ("grace_updates", [-1, True, 1.5]),
    ("window", [0, -1, True, 1.5]),
    ("moving_avg_patience", [0, -1, True, 1.5]),
    ("raw_loss_patience", [0, -1, True, 1.5]),
    ("moving_avg_delta", [-1, math.nan, math.inf, True, "0.5", 10**400]),
    ("raw_loss_delta", [-1, math.nan, math.inf, True, "2", 10**400]),
])
def test_config_validation(key, values):
    for value in values:
        config = load_experiment_config()
        config["divergence_guard"][key] = value
        with pytest.raises(ValueError, match="divergence_guard"):
            validate_experiment_config(config)


def test_toml_defaults_and_legacy_checkpoint(tmp_path):
    defaults = load_experiment_config()
    assert defaults["divergence_guard"] == DEFAULTS
    legacy = copy.deepcopy(defaults)
    del legacy["divergence_guard"]
    checkpoint = dict(format_version=CHECKPOINT_FORMAT_VERSION, resolved_config=legacy)
    validate_checkpoint_config(checkpoint, defaults)
    assert "divergence_guard" not in legacy
    path = tmp_path / "guard.toml"
    path.write_text('run_name = "guard"\n[divergence_guard]\nenabled = true\n')
    config = load_experiment_config(path)
    assert config["divergence_guard"] == {**DEFAULTS, "enabled": True}
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_config(checkpoint, config)
