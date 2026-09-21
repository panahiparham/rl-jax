"""Tests for benchmarks/core/weekly.py's scheduler adapters.

Exercises each adapter against monkeypatched seams (experiment.slurm's
dispatch/wipe/is_queued/load_config, and subprocess.run for git/gh) rather
than a live Vulcan session or GitHub - decide()/tick() themselves are
already covered by simple-experiments' own test suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "benchmarks" / "core"))

import weekly
from experiment.weekly import DurableHistory, JobStatus, TransientState

# --- stores ------------------------------------------------------------


def test_transient_store_round_trips_a_state(tmp_path):
    state = TransientState.initial(datetime(2026, 1, 1, tzinfo=UTC))
    store = weekly._FileTransientStore(tmp_path / "state.json")

    store.save(state)

    assert store.load() == state


def test_transient_store_returns_none_when_nothing_is_saved(tmp_path):
    store = weekly._FileTransientStore(tmp_path / "state.json")

    assert store.load() is None


def test_history_store_round_trips_a_completed_run(tmp_path):
    history = DurableHistory(
        last_completed_sha="abc123", last_completed_at=datetime(2026, 1, 1, tzinfo=UTC)
    )
    store = weekly._StateJsonHistoryStore(tmp_path / "state.json")

    store.save(history)

    assert store.load() == history


def test_history_store_rejects_an_incomplete_history(tmp_path):
    store = weekly._StateJsonHistoryStore(tmp_path / "state.json")

    with pytest.raises(ValueError, match="sha and a timestamp"):
        store.save(DurableHistory(last_completed_sha=None, last_completed_at=None))


# --- lock ----------------------------------------------------------------


def test_lock_refuses_a_second_acquire_while_held(tmp_path):
    lock = weekly._FlockLock(tmp_path / "lock")

    with lock.try_acquire():
        assert lock.try_acquire() is None

    assert lock.try_acquire() is not None


# --- dispatcher --------------------------------------------------------


def _dispatcher(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly, "repo_root", lambda: tmp_path)
    return weekly._SlurmDispatcher(
        label="bench_core", run_py=tmp_path / "run.py",
        results_dir=tmp_path / "results", num_workers=1,
    )


def _write_dispatch_state(tmp_path, sha, jobs=None):
    path = tmp_path / ".cluster" / "bench_core.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sha": sha, "jobs": jobs or {"merge": "1"}}))


def test_submit_dispatches_when_nothing_is_in_flight(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(weekly, "_slurm_wipe", lambda **kw: calls.append("wipe"))
    monkeypatch.setattr(weekly, "_slurm_dispatch", lambda **kw: calls.append("dispatch"))
    dispatcher = _dispatcher(tmp_path, monkeypatch)

    returned = dispatcher.submit("sha1")

    assert returned == "sha1"
    assert calls == ["wipe", "dispatch"]


def test_submit_is_a_no_op_for_a_sha_already_dispatched(tmp_path, monkeypatch):
    _write_dispatch_state(tmp_path, "sha1")
    calls = []
    monkeypatch.setattr(weekly, "_slurm_wipe", lambda **kw: calls.append("wipe"))
    monkeypatch.setattr(weekly, "_slurm_dispatch", lambda **kw: calls.append("dispatch"))
    dispatcher = _dispatcher(tmp_path, monkeypatch)

    returned = dispatcher.submit("sha1")

    assert returned == "sha1"
    assert calls == []


def test_submit_redispatches_for_a_different_sha(tmp_path, monkeypatch):
    _write_dispatch_state(tmp_path, "sha1")
    calls = []
    monkeypatch.setattr(weekly, "_slurm_wipe", lambda **kw: calls.append("wipe"))
    monkeypatch.setattr(weekly, "_slurm_dispatch", lambda **kw: calls.append("dispatch"))
    dispatcher = _dispatcher(tmp_path, monkeypatch)

    dispatcher.submit("sha2")

    assert calls == ["wipe", "dispatch"]


def test_submit_translates_a_systemexit_into_a_catchable_error(tmp_path, monkeypatch):
    def _refused(**kw):
        raise SystemExit("MFA needs a terminal")

    monkeypatch.setattr(weekly, "_slurm_wipe", _refused)
    dispatcher = _dispatcher(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="MFA needs a terminal"):
        dispatcher.submit("sha1")


def test_poll_reports_running_while_queued(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly, "is_queued", lambda **kw: True)
    dispatcher = _dispatcher(tmp_path, monkeypatch)

    assert dispatcher.poll("sha1") is JobStatus.RUNNING


def test_poll_reports_running_when_vulcan_is_unreachable(tmp_path, monkeypatch):
    def _unreachable(**kw):
        raise SystemExit("can't reach vulcan")

    monkeypatch.setattr(weekly, "is_queued", _unreachable)
    dispatcher = _dispatcher(tmp_path, monkeypatch)

    assert dispatcher.poll("sha1") is JobStatus.RUNNING


def _sacct(tmp_path, monkeypatch, output: str):
    _write_dispatch_state(tmp_path, "sha1", jobs={"merge": "42"})
    monkeypatch.setattr(weekly, "is_queued", lambda **kw: False)
    monkeypatch.setattr(weekly, "load_config", lambda: object())
    monkeypatch.setattr(
        weekly, "_remote_or_local",
        lambda cfg, cmd: subprocess.CompletedProcess(cmd, 0, output, ""),
    )
    return _dispatcher(tmp_path, monkeypatch)


def test_poll_reports_succeeded_once_every_job_completed(tmp_path, monkeypatch):
    dispatcher = _sacct(tmp_path, monkeypatch, "COMPLETED\n")

    assert dispatcher.poll("sha1") is JobStatus.SUCCEEDED


def test_poll_reports_failed_when_a_job_did_not_complete(tmp_path, monkeypatch):
    dispatcher = _sacct(tmp_path, monkeypatch, "FAILED\n")

    assert dispatcher.poll("sha1") is JobStatus.FAILED


def test_poll_reports_running_when_sacct_has_no_data_yet(tmp_path, monkeypatch):
    dispatcher = _sacct(tmp_path, monkeypatch, "")

    assert dispatcher.poll("sha1") is JobStatus.RUNNING


# --- publisher -----------------------------------------------------------


def _fake_run(responses):
    def run(argv, **kwargs):
        del kwargs
        for pattern, result in responses:
            if tuple(argv[: len(pattern)]) == pattern:
                return result
        raise AssertionError(f"unexpected call: {argv}")

    return run


def test_publish_reuses_an_existing_pr_for_the_branch(tmp_path, monkeypatch):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    responses = [
        (("gh", "pr", "view"),
         subprocess.CompletedProcess([], 0, "https://github.com/x/y/pull/1", "")),
    ]
    monkeypatch.setattr(subprocess, "run", _fake_run(responses))
    publisher = weekly._GhPublisher(repo_root=tmp_path, plots_dir=plots_dir)

    url = publisher.publish("sha1", [])

    assert url == "https://github.com/x/y/pull/1"


def test_publish_creates_a_pr_when_none_exists_for_the_branch(tmp_path, monkeypatch):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "catch.png").write_bytes(b"x")
    responses = [
        (("gh", "pr", "view"), subprocess.CompletedProcess([], 1, "", "no pr")),
        (("git", "fetch"), subprocess.CompletedProcess([], 0, "", "")),
        (("git", "checkout"), subprocess.CompletedProcess([], 0, "", "")),
        (("git", "add"), subprocess.CompletedProcess([], 0, "", "")),
        (("git", "commit"), subprocess.CompletedProcess([], 0, "", "")),
        (("git", "push"), subprocess.CompletedProcess([], 0, "", "")),
        (("gh", "repo", "view"), subprocess.CompletedProcess([], 0, "me/repo\n", "")),
        (("gh", "pr", "create"),
         subprocess.CompletedProcess([], 0, "https://github.com/me/repo/pull/2\n", "")),
    ]
    monkeypatch.setattr(subprocess, "run", _fake_run(responses))
    publisher = weekly._GhPublisher(repo_root=tmp_path, plots_dir=plots_dir)

    url = publisher.publish("sha1", [])

    assert url == "https://github.com/me/repo/pull/2"
