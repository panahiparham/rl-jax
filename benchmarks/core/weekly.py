"""
Weekly benchmark: dispatch bench_core to Vulcan on a schedule, publish
results as a PR.

Wraps simple-experiments' experiment.weekly (a stateless, crash-safe
scheduler: check due, dispatch, poll, publish) with the concrete
Dispatcher/Reporter/Publisher/TransientStore/HistoryStore/Lock this project
needs. `tick()` is the whole surface a cron entry calls:

    uv run benchmarks/core/weekly.py tick

Runtime scheduling state lives under .cluster/ (gitignored, so a dispatch
that requires a clean tree is never blocked by it); durable completion
history is benchmarks/state.json, committed by the same PR that publishes
each week's results.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

from experiment.weekly import DurableHistory, Phase, TransientState


def _state_to_json(state: TransientState) -> dict:
    data = dataclasses.asdict(state)
    data["phase"] = state.phase.value
    data["next_wake_at"] = state.next_wake_at.isoformat()
    return data


def _state_from_json(data: dict) -> TransientState:
    return TransientState(
        phase=Phase(data["phase"]),
        next_wake_at=datetime.fromisoformat(data["next_wake_at"]),
        dispatch_sha=data["dispatch_sha"],
        dispatch_token=data["dispatch_token"],
        last_publish_id=data["last_publish_id"],
        attempt_count=data["attempt_count"],
        last_error=data["last_error"],
    )


class _FileTransientStore:
    """Runtime scheduling state, outside git tracking."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> TransientState | None:
        if not self._path.is_file():
            return None
        return _state_from_json(json.loads(self._path.read_text()))

    def save(self, state: TransientState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(_state_to_json(state), indent=2) + "\n")


class _StateJsonHistoryStore:
    """Durable completion history: benchmarks/state.json, git-tracked."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> DurableHistory | None:
        if not self._path.is_file():
            return None
        data = json.loads(self._path.read_text())
        return DurableHistory(
            last_completed_sha=data["last_sha"],
            last_completed_at=datetime.fromisoformat(data["last_run"]),
        )

    def save(self, history: DurableHistory) -> None:
        if history.last_completed_sha is None or history.last_completed_at is None:
            raise ValueError("a completed run always has a sha and a timestamp")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_sha": history.last_completed_sha,
            "last_run": history.last_completed_at.isoformat(),
        }
        self._path.write_text(json.dumps(data, indent=2) + "\n")
