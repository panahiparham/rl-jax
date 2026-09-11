"""Weekly-benchmark orchestration: check/dispatch/finish, no chat mediation.

Three questions, checked in this order:

1. Is a dispatch already in flight (``.cluster/<label>.json`` exists)? If so,
   are its jobs still queued, or done?
2. If nothing is in flight, is a fresh run due (:func:`benchmarking.schedule.is_due`)?
3. Otherwise, idle.

:func:`check_status` answers all three with one :class:`Status`. The actions
a human explicitly triggers from a benchmark's ``weekly.py`` CLI - the Vulcan
MFA step happens before :func:`dispatch`, a PR review happens after - live
here too, reusing :mod:`benchmarking.schedule`, :mod:`benchmarking.report` and
:mod:`experiment.slurm`.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from experiment.design import Experiment
from experiment.slurm import dispatch as _slurm_dispatch
from experiment.slurm import is_queued
from experiment.slurm import wipe as _slurm_wipe

from benchmarking.report import Environment, render_plots
from benchmarking.schedule import (
    BenchmarkState,
    is_due,
    read_state,
    remote_sha,
    write_state,
)

__all__ = [
    "STATUS_DUE",
    "STATUS_IDLE",
    "STATUS_QUEUED",
    "STATUS_READY_TO_FINISH",
    "STATUS_UNREACHABLE",
    "AuthRefusedError",
    "Status",
    "check_status",
    "dispatch",
    "finish",
    "open_pr",
]

_AUTH_REFUSAL_MARKER = "MFA needs a terminal"

STATUS_IDLE = "idle"
STATUS_DUE = "due"
STATUS_QUEUED = "queued"
STATUS_READY_TO_FINISH = "ready_to_finish"
STATUS_UNREACHABLE = "unreachable"


class AuthRefusedError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Status:
    """The answer to "what, if anything, should happen next?"."""

    status: str  # one of the STATUS_* constants
    detail: str  # a human-readable reason, for a CLI to print


def check_status(
    *,
    label: str,
    state_path: str | Path,
    repo_root: str | Path,
    config_path: str | Path | None = None,
) -> Status:
    """Whether a dispatch is in flight, or a fresh run is due.

    An in-flight dispatch that cannot reach Vulcan (e.g. an expired ssh
    ControlMaster) reports STATUS_UNREACHABLE rather than STATUS_QUEUED: a
    real connection failure is worth surfacing, not reading as "nothing to
    do" alongside a legitimately still-running sweep.
    """
    repo_root = Path(repo_root)
    if (repo_root / ".cluster" / f"{label}.json").is_file():
        try:
            queued = is_queued(label=label, config_path=config_path)
        except SystemExit as exc:
            return Status(STATUS_UNREACHABLE, f"can't reach Vulcan: {exc}")
        if queued:
            return Status(STATUS_QUEUED, "jobs still in squeue")
        return Status(STATUS_READY_TO_FINISH, "jobs no longer queued - run finish")

    state = read_state(state_path)
    sha = remote_sha(cwd=repo_root)
    if is_due(state, sha, datetime.now(UTC)):
        last = f"{state.last_sha[:7]} on {state.last_run}" if state else "never"
        return Status(STATUS_DUE, f"origin/main is {sha[:7]}, last benchmarked {last}")
    return Status(STATUS_IDLE, "nothing to do")


def dispatch(
    *,
    label: str,
    run_py: str | Path,
    results_dir: str | Path,
    num_workers: int,
    config_path: str | Path | None = None,
) -> None:
    """Wipe prior results and dispatch a fresh sweep.

    The caller's job to have an authenticated Vulcan session open - this
    raises :class:`AuthRefusedError` if the ssh session has expired, and the
    same ``SystemExit`` ``slurm.dispatch`` always does if the tree is dirty.

    ``results_dir`` is wiped first because dedup is by ``run_id`` regardless
    of commit, so a rerun against unwiped results would skip everything.
    """
    shutil.rmtree(results_dir, ignore_errors=True)
    try:
        _slurm_wipe(label=label, config_path=config_path)
        _slurm_dispatch(
            label=label,
            run_py=Path(run_py),
            mode="sweep",
            argv=["--num-workers", str(num_workers)],
            config_path=config_path,
        )
    except SystemExit as exc:
        if _AUTH_REFUSAL_MARKER in str(exc):
            raise AuthRefusedError(str(exc)) from exc
        raise


def finish(
    *,
    experiment: Experiment,
    environments: Sequence[Environment],
    run_py: str | Path,
    plots_dir: str | Path,
    state_path: str | Path,
    repo_root: str | Path = Path("."),
) -> str:
    """Sync results, render the plots, and record the new state.

    Syncing runs the benchmark's own ``run.py sync``, so the experiment brings
    its results home and merges them exactly as it would by hand.
    """
    label = experiment.name
    synced = subprocess.run(
        [sys.executable, str(run_py), "sync"], cwd=str(repo_root), check=False
    )
    if synced.returncode != 0:
        raise SystemExit(f"[{label}] sync failed")

    sha = remote_sha(cwd=repo_root)
    render_plots(experiment, environments, Path(plots_dir))

    write_state(
        state_path,
        BenchmarkState(last_sha=sha, last_run=datetime.now(UTC).isoformat()),
    )

    # The dispatch this run processed is done - without this, check_status()
    # would find .cluster/<label>.json still there and report ready_to_finish
    # forever, re-running finish every tick indefinitely.
    dispatch_state = Path(repo_root) / ".cluster" / f"{label}.json"
    dispatch_state.unlink(missing_ok=True)

    return sha


def _git(repo_root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
    )


def _repo_slug(repo_root: Path) -> str:
    proc = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _pr_body(label: str, plots_dir: Path, repo_root: Path, branch: str) -> str:
    slug = _repo_slug(repo_root)
    lines = [f"Automated run of {label}, one learning curve per environment.", ""]
    for png in sorted(plots_dir.glob("*.png")):
        rel = png.relative_to(repo_root).as_posix()
        url = f"https://github.com/{slug}/blob/{branch}/{rel}?raw=true"
        lines += [f"### {png.stem}", "", f"![{png.stem}]({url})", ""]
    return "\n".join(lines)


def open_pr(
    *,
    label: str,
    plots_path: str | Path,
    repo_root: str | Path,
    run_date: str | None = None,
) -> str:
    """Branch, commit the rendered plots + state, push, and open a PR."""
    repo_root = Path(repo_root)
    plots_dir = repo_root / plots_path
    run_date = run_date or datetime.now(UTC).date().isoformat()
    branch = f"chore/weekly-benchmark-{run_date}"

    # -B (not -b) off a freshly-fetched origin/main: idempotent regardless of
    # what's currently checked out or whether this branch already exists
    # locally (e.g. a prior run of this same day that didn't get this far).
    _git(repo_root, "fetch", "origin", "main")
    _git(repo_root, "checkout", "-B", branch, "origin/main")
    _git(repo_root, "add", "benchmarks/state.json", str(plots_path))
    _git(repo_root, "commit", "-m", f"data: weekly benchmark run {run_date}")
    _git(repo_root, "push", "--force", "-u", "origin", branch)

    proc = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--title",
            f"Weekly benchmark: {run_date}",
            "--body",
            _pr_body(label, plots_dir, repo_root, branch),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()
