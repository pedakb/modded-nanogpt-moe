"""Read and align TensorBoard scalar data without importing PyTorch.

Records are plain dictionaries with ``tag``, ``step``, ``value``, and
``wall_time`` fields. Keeping the interchange format small avoids a required
pandas dependency.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


_EVENT_PREFIX = "events.out.tfevents."


def find_event_directory(run_directory: str | Path) -> Path:
    """Find the unique event directory beneath a run or trial directory.

    Multiple event files in one directory are supported for restarted/merged
    streams. Multiple directories are rejected to avoid silently mixing trials.
    """
    run_directory = Path(run_directory).expanduser()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"TensorBoard run directory not found: {run_directory}")
    candidates = {
        path.parent for path in run_directory.rglob(f"{_EVENT_PREFIX}*")
        if path.is_file()
    }
    if not candidates:
        raise FileNotFoundError(f"no TensorBoard event files found under: {run_directory}")
    if len(candidates) != 1:
        choices = ", ".join(str(path) for path in sorted(candidates))
        raise ValueError(
            f"multiple TensorBoard event directories under {run_directory}; "
            f"select one explicitly: {choices}")
    return candidates.pop()


def _accumulator(run_directory: str | Path) -> EventAccumulator:
    return EventAccumulator(
        str(find_event_directory(run_directory)),
        size_guidance={"scalars": 0},
    ).Reload()


def _filter_tags(tags: Iterable[str], substring: str | None,
                 regex: str | re.Pattern[str] | None) -> list[str]:
    pattern = re.compile(regex) if isinstance(regex, str) else regex
    return sorted(
        tag for tag in tags
        if (substring is None or substring in tag)
        and (pattern is None or pattern.search(tag)))


def list_scalar_tags(run_directory: str | Path, *, substring: str | None = None,
                     regex: str | re.Pattern[str] | None = None) -> list[str]:
    """List scalar tags, optionally filtered by substring and/or regex."""
    accumulator = _accumulator(run_directory)
    return _filter_tags(accumulator.Tags().get("scalars", ()), substring, regex)


def _records(accumulator: EventAccumulator, tag: str) -> list[dict]:
    return sorted(
        ({"tag": tag, "step": int(event.step), "value": float(event.value),
          "wall_time": float(event.wall_time)}
         for event in accumulator.Scalars(tag)),
        key=lambda record: (record["step"], record["wall_time"]),
    )


def load_scalar_tag(run_directory: str | Path, tag: str, *,
                    missing: str = "empty") -> list[dict]:
    """Load one scalar tag, returning [] or raising for an absent tag."""
    if missing not in ("empty", "raise"):
        raise ValueError("missing must be 'empty' or 'raise'")
    accumulator = _accumulator(run_directory)
    if tag not in accumulator.Tags().get("scalars", ()):
        if missing == "empty":
            return []
        raise KeyError(f"scalar tag not found: {tag}")
    return _records(accumulator, tag)


def load_all_scalars(run_directory: str | Path, *,
                     substring: str | None = None,
                     regex: str | re.Pattern[str] | None = None) -> list[dict]:
    """Load all matching scalars without assuming tags share steps."""
    accumulator = _accumulator(run_directory)
    tags = _filter_tags(accumulator.Tags().get("scalars", ()), substring, regex)
    return [record for tag in tags for record in _records(accumulator, tag)]


def latest_by_step(records: Iterable[Mapping]) -> dict[int, Mapping]:
    """Select the latest-wall-time value for each duplicate global step."""
    chosen: dict[int, Mapping] = {}
    for record in records:
        step = int(record["step"])
        previous = chosen.get(step)
        if (previous is None
                or float(record.get("wall_time", float("-inf")))
                >= float(previous.get("wall_time", float("-inf")))):
            chosen[step] = record
    return chosen


def align_scalar_series(series: Mapping[str, Iterable[Mapping]], *,
                        how: str = "inner") -> list[dict]:
    """Join named scalar series on exact step, without interpolation."""
    if how not in ("inner", "outer"):
        raise ValueError("how must be 'inner' or 'outer'")
    indexed = {name: latest_by_step(records) for name, records in series.items()}
    if not indexed:
        return []
    step_sets = [set(records) for records in indexed.values()]
    steps = (set.intersection(*step_sets) if how == "inner"
             else set.union(*step_sets))
    return [
        {"step": step, **{
            name: (float(records[step]["value"]) if step in records else None)
            for name, records in indexed.items()
        }}
        for step in sorted(steps)
    ]
