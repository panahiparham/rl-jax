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
import fcntl
import json
import os
import shlex
import shutil
import subprocess
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from pathlib import Path

from experiment.slurm import dispatch as _slurm_dispatch
from experiment.slurm import is_queued, load_config, repo_root
from experiment.slurm import wipe as _slurm_wipe
from experiment.weekly import DurableHistory, JobStatus, Phase, TransientState


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


@contextmanager
def _held(handle):
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


class _FlockLock:
    """Coordinates against overlapping ticks with an flock on a marker file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def try_acquire(self) -> AbstractContextManager[None] | None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return None
        return _held(handle)


def _reraise_as_runtime(fn, /, *args, **kwargs):
    """Run an experiment.slurm call, translating its SystemExit.

    tick() only wraps a Dispatcher/Publisher call's own exception type in its
    crash recovery - SystemExit (slurm's error-signalling convention) is not
    an Exception subclass, so left alone it would escape tick() uncaught
    instead of settling the suite in FAILED.
    """
    try:
        return fn(*args, **kwargs)
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from exc


def _remote_or_local(cfg, command: str) -> subprocess.CompletedProcess:
    """Run a shell command on the login node, or locally under EXPERIMENT_LOCAL_MODE.

    Mirrors experiment.slurm's own local-mode switch (used by its test
    sandbox) so this module's cluster calls are testable the same way.
    """
    argv = (
        ["bash", "-c", command]
        if os.environ.get("EXPERIMENT_LOCAL_MODE") == "1"
        else ["ssh", cfg.host, command]
    )
    return subprocess.run(argv, capture_output=True, text=True, check=False)


class _SlurmDispatcher:
    """Dispatches bench_core to Vulcan and polls it, deduping by commit sha."""

    def __init__(
        self, *, label: str, run_py: Path, results_dir: Path, num_workers: int
    ) -> None:
        self._label = label
        self._run_py = run_py
        self._results_dir = results_dir
        self._num_workers = num_workers
        self._state_path = repo_root() / ".cluster" / f"{label}.json"

    def _dispatched_sha(self) -> str | None:
        if not self._state_path.is_file():
            return None
        return json.loads(self._state_path.read_text()).get("sha")

    def submit(self, sha: str) -> str:
        """Dispatch a fresh sweep for ``sha``, unless one is already running.

        A dispatch already recorded for this sha means either a resubmit
        after the token was lost to a crash, or a redundant Submit for a
        dispatch already in flight - either way, dispatching again would
        duplicate the sweep, so this is a no-op.
        """
        if self._dispatched_sha() == sha:
            return sha
        shutil.rmtree(self._results_dir, ignore_errors=True)
        _reraise_as_runtime(_slurm_wipe, label=self._label)
        _reraise_as_runtime(
            _slurm_dispatch,
            label=self._label,
            run_py=self._run_py,
            mode="sweep",
            argv=["--num-workers", str(self._num_workers)],
        )
        return sha

    def poll(self, token: str) -> JobStatus:
        """Whether the dispatch is still running, succeeded, or failed.

        Never raises: tick() polls outside its try/except, so an
        unreachable Vulcan (an expired ssh ControlMaster, say) has to read
        as "can't tell yet" rather than crash the tick.
        """
        del token  # the label's own .cluster state is authoritative
        try:
            queued = is_queued(label=self._label)
        except SystemExit:
            return JobStatus.RUNNING
        return JobStatus.RUNNING if queued else self._final_status()

    def _final_status(self) -> JobStatus:
        state = json.loads(self._state_path.read_text())
        ids = ",".join(v for v in state.get("jobs", {}).values() if v)
        cfg = load_config()
        proc = _remote_or_local(
            cfg, f"sacct -j {shlex.quote(ids)} -X --format=State --noheader"
        )
        states = proc.stdout.split()
        if proc.returncode != 0 or not states:
            return JobStatus.RUNNING
        if all(s.startswith("COMPLETED") for s in states):
            return JobStatus.SUCCEEDED
        return JobStatus.FAILED
