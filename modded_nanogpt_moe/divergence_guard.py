"""Causal completed-update loss guard; no model-selection logic."""

from collections import deque
import math


DEFAULTS = {
    "enabled": False,
    "grace_updates": 100,
    "window": 20,
    "moving_avg_delta": 0.5,
    "moving_avg_patience": 10,
    "raw_loss_delta": 2.0,
    "raw_loss_patience": 3,
}


def validate_guard_config(settings):
    if not isinstance(settings["enabled"], bool):
        raise ValueError("divergence_guard.enabled must be a boolean")
    for key in ("grace_updates", "window", "moving_avg_patience", "raw_loss_patience"):
        value = settings[key]
        minimum = 0 if key == "grace_updates" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"divergence_guard.{key} must be an integer >= {minimum}")
    for key in ("moving_avg_delta", "raw_loss_delta"):
        value = settings[key]
        try:
            valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                     and value >= 0 and math.isfinite(value))
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"divergence_guard.{key} must be finite and nonnegative")


class DivergenceGuard:
    def __init__(self, settings):
        validate_guard_config(settings)
        self.settings = dict(settings)
        self.enabled = settings["enabled"]
        self.losses = deque(maxlen=settings["window"])
        self.best_moving_avg = None
        self.moving_avg = None
        self.moving_avg_count = 0
        self.raw_loss_count = 0

    def observe(self, update, loss):
        """Return a stop record, or None. Call once per completed update."""
        if not self.enabled:
            return None
        reason = None
        if not math.isfinite(loss):
            reason = "nonfinite_loss"
        else:
            self.losses.append(loss)
            if len(self.losses) < self.losses.maxlen:
                return None
            self.moving_avg = math.fsum(self.losses) / len(self.losses)
            if update < self.settings["grace_updates"]:
                return None
            self.best_moving_avg = (self.moving_avg if self.best_moving_avg is None
                                    else min(self.best_moving_avg, self.moving_avg))
            self.moving_avg_count = (self.moving_avg_count + 1
                if self.moving_avg - self.best_moving_avg > self.settings["moving_avg_delta"] else 0)
            self.raw_loss_count = (self.raw_loss_count + 1
                if loss - self.best_moving_avg > self.settings["raw_loss_delta"] else 0)
            if self.moving_avg_count >= self.settings["moving_avg_patience"]:
                reason = "moving_average_deterioration"
            elif self.raw_loss_count >= self.settings["raw_loss_patience"]:
                reason = "raw_loss_spike"
        if reason is None:
            return None
        return dict(update=update, reason=reason, current_loss=loss,
                    moving_avg=self.moving_avg, best_moving_avg=self.best_moving_avg)

    def state_dict(self):
        return dict(losses=list(self.losses), best_moving_avg=self.best_moving_avg,
                    moving_avg=self.moving_avg, moving_avg_count=self.moving_avg_count,
                    raw_loss_count=self.raw_loss_count)

    def load_state_dict(self, state):
        self.losses.extend(state["losses"])
        for key in ("best_moving_avg", "moving_avg", "moving_avg_count", "raw_loss_count"):
            setattr(self, key, state[key])
