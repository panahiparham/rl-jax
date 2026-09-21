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

import argparse
import dataclasses
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, Any

from experiment.design import Experiment
from experiment.slurm import ClusterConfig
from experiment.slurm import dispatch as _slurm_dispatch
from experiment.slurm import is_queued, load_config, repo_root
from experiment.slurm import wipe as _slurm_wipe
from experiment.weekly import (
    DurableHistory,
    JobStatus,
    Phase,
    TransientState,
    WeeklyBenchmarkConfig,
    tick,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ENVIRONMENTS, EXPERIMENT, PLOTS_DIR
from report import render_plots

LABEL = EXPERIMENT.name
_RUN_PY = Path(__file__).resolve().parent / "run.py"
_NUM_WORKERS = 1


def _state_to_json(state: TransientState) -> dict[str, Any]:
    data: dict[str, Any] = dataclasses.asdict(state)
    data["phase"] = state.phase.value
    data["next_wake_at"] = state.next_wake_at.isoformat()
    return data


def _state_from_json(data: dict[str, Any]) -> TransientState:
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
def _held(handle: IO[str]) -> Iterator[None]:
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


def _reraise_as_runtime[T](fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
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


def _remote_or_local(
    cfg: ClusterConfig, command: str
) -> subprocess.CompletedProcess[str]:
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
        sha = json.loads(self._state_path.read_text()).get("sha")
        return sha if isinstance(sha, str) else None

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


class _GhPublisher:
    """Commits the rendered plots + history, and opens (or reuses) a PR."""

    def __init__(self, *, repo_root: Path, plots_dir: Path) -> None:
        self._repo_root = repo_root
        self._plots_dir = plots_dir

    def publish(self, sha: str, artifacts: Sequence[Path]) -> str:
        del artifacts  # already rendered to self._plots_dir by the Reporter
        del sha  # the branch name is what's deduped on, not the commit
        run_date = datetime.now(UTC).date().isoformat()
        branch = f"chore/weekly-benchmark-{run_date}"

        existing = self._existing_pr(branch)
        if existing is not None:
            return existing

        # -B off a freshly-fetched origin/main: idempotent regardless of what's
        # currently checked out, or of a same-day PR that didn't get this far.
        self._git("fetch", "origin", "main")
        self._git("checkout", "-B", branch, "origin/main")
        self._git(
            "add", "benchmarks/state.json",
            str(self._plots_dir.relative_to(self._repo_root)),
        )
        self._git("commit", "-m", f"data: weekly benchmark run {run_date}")
        self._git("push", "--force", "-u", "origin", branch)
        return self._create_pr(branch, run_date)

    def _existing_pr(self, branch: str) -> str | None:
        proc = subprocess.run(
            ["gh", "pr", "view", branch, "--json", "url", "-q", ".url"],
            cwd=self._repo_root, capture_output=True, text=True, check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    def _git(self, *args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=self._repo_root, check=True,
            capture_output=True, text=True,
        )

    def _repo_slug(self) -> str:
        proc = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            cwd=self._repo_root, capture_output=True, text=True, check=True,
        )
        return proc.stdout.strip()

    def _pr_body(self, branch: str) -> str:
        slug = self._repo_slug()
        lines = ["Automated run of bench_core, one learning curve per environment.", ""]
        for png in sorted(self._plots_dir.glob("*.png")):
            rel = png.relative_to(self._repo_root).as_posix()
            url = f"https://github.com/{slug}/blob/{branch}/{rel}?raw=true"
            lines += [f"### {png.stem}", "", f"![{png.stem}]({url})", ""]
        return "\n".join(lines)

    def _create_pr(self, branch: str, run_date: str) -> str:
        proc = subprocess.run(
            [
                "gh", "pr", "create",
                "--title", f"Weekly benchmark: {run_date}",
                "--body", self._pr_body(branch),
            ],
            cwd=self._repo_root, capture_output=True, text=True, check=True,
        )
        return proc.stdout.strip()


def _reporter(sha: str, experiment: Experiment, out_dir: Path) -> Sequence[Path]:
    """Sync the cluster's results home, then render this week's plots."""
    del sha  # render_plots renders whatever is now stored locally
    synced = subprocess.run(
        [sys.executable, str(_RUN_PY), "sync"], cwd=_REPO_ROOT, check=False
    )
    if synced.returncode != 0:
        raise RuntimeError(f"[{LABEL}] sync failed")
    return render_plots(experiment, ENVIRONMENTS, out_dir)


def _remote_sha(history_store: _StateJsonHistoryStore) -> str:
    """origin/main's current sha, falling back to the last completed one.

    tick() calls this outside its try/except, so it must never raise: a
    transient git failure has to read as "nothing new" rather than crash
    the tick.
    """
    proc = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/main"],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    line = proc.stdout.strip()
    if proc.returncode == 0 and line:
        return line.split()[0]
    history = history_store.load()
    return history.last_completed_sha if history else ""


def _next_scheduled_wake(now: datetime) -> datetime:
    return now + timedelta(days=7)


def _poll_interval(now: datetime) -> datetime:
    return now + timedelta(minutes=20)


def _config() -> WeeklyBenchmarkConfig:
    history_store = _StateJsonHistoryStore(_REPO_ROOT / "benchmarks" / "state.json")
    return WeeklyBenchmarkConfig(
        label=LABEL,
        experiment=EXPERIMENT,
        remote_sha=lambda: _remote_sha(history_store),
        next_scheduled_wake=_next_scheduled_wake,
        poll_interval=_poll_interval,
        dispatcher=_SlurmDispatcher(
            label=LABEL, run_py=_RUN_PY, results_dir=EXPERIMENT.results_dir,
            num_workers=_NUM_WORKERS,
        ),
        reporter=_reporter,
        publisher=_GhPublisher(repo_root=_REPO_ROOT, plots_dir=PLOTS_DIR),
        transient_store=_FileTransientStore(
            _REPO_ROOT / ".cluster" / f"{LABEL}-weekly.json"
        ),
        history_store=history_store,
        lock=_FlockLock(_REPO_ROOT / ".cluster" / f"{LABEL}-weekly.lock"),
        out_dir=PLOTS_DIR,
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="weekly.py")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("tick")
    parser.parse_args()

    state = tick(_config(), datetime.now(UTC))
    detail = f": {state.last_error}" if state.last_error else ""
    print(f"[{LABEL}] {state.phase.value}{detail}")
    return 1 if state.phase is Phase.FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
