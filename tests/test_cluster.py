"""End-to-end tests for the SLURM workflow - without a cluster.

See ``conftest.py`` for the ``sandbox`` fixture this relies on. What is under
test here is the wiring: which commit gets snapshotted, which venv and
PYTHONPATH the jobs are told to use, what sbatch is actually asked for, when a
venv is rebuilt, and that synced databases merge into local ones instead of
replacing them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import (
    _REMOTE_DIR,
    CLUSTER_TOML,
    QUIET_STUB,
    SBATCH_STUB,
    TOY_RUN_PY,
    UV_STUB,
    Sandbox,
    _commit,
    _git,
    _run_ids,
    _seed_cluster_results,
    _seed_db,
    run_py,
    setup_cluster,
)
from experiment import slurm

_REPO = Path(__file__).resolve().parents[1]


def _build_sandbox_at(tmp_path: Path) -> Sandbox:
    home = tmp_path / "home"
    (home / "projects" / "def-test").mkdir(parents=True)
    (home / "projects" / "def-sponsor00").mkdir()
    (home / "scratch").mkdir()

    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copytree(_REPO / "src", repo / "src")
    shutil.copytree(_REPO / "scripts", repo / "scripts")
    shutil.copy(_REPO / ".gitignore", repo / ".gitignore")
    shutil.copy(_REPO / "uv.lock", repo / "uv.lock")
    shutil.copy(_REPO / "pyproject.toml", repo / "pyproject.toml")
    shutil.copy(_REPO / "README.md", repo / "README.md")
    (repo / "cluster.toml").write_text(CLUSTER_TOML)
    exp = repo / "experiments" / "toy"
    exp.mkdir(parents=True)
    (exp / "run.py").write_text(TOY_RUN_PY)
    (repo / "marker.txt").write_text("v1")

    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _commit(repo, "initial")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in [
        ("sbatch", SBATCH_STUB),
        ("uv", UV_STUB),
        ("squeue", QUIET_STUB),
        ("sacct", QUIET_STUB),
    ]:
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)

    calls = tmp_path / "calls"
    calls.mkdir()

    env = {
        "HOME": str(home),
        "EXPERIMENT_LOCAL_MODE": "1",
        "EXPERIMENT_REMOTE_DIR": str(_REMOTE_DIR),
        "GSP_TEST_CALLS": str(calls),
        "GSP_TEST_PYTHON": sys.executable,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHONPATH": str(repo / "src"),
    }
    return Sandbox(repo=repo, home=home, calls=calls, env=env)


@pytest.fixture(scope="module")
def bare_repo_and_venvs(tmp_path_factory: pytest.TempPathFactory) -> Sandbox:
    sandbox = _build_sandbox_at(tmp_path_factory.mktemp("setup"))
    setup_cluster(sandbox)
    return sandbox


@pytest.fixture(scope="module")
def dispatched_sweep(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    sandbox = _build_sandbox_at(tmp_path_factory.mktemp("dispatch"))
    setup_cluster(sandbox)
    sha = _git(sandbox.repo, "rev-parse", "HEAD")

    run_py(sandbox, "sweep", "--num-workers", "3", "--slurm")

    (rundir,) = sandbox.run_dirs
    plan, array, merge = sandbox.sbatch_calls()
    return SimpleNamespace(
        sandbox=sandbox, sha=sha, rundir=rundir, plan=plan, array=array, merge=merge
    )


# --- setup (requirement 1) ---------------------------------------------------------


def test_setup_creates_the_bare_repo_wired_as_the_cluster_remote(
    bare_repo_and_venvs: Sandbox,
):
    sandbox = bare_repo_and_venvs
    assert (sandbox.root / "online-gsp.git" / "HEAD").exists(), "no bare repo"
    url = _git(sandbox.repo, "remote", "get-url", "cluster-testcluster")
    assert url == str(sandbox.root / "online-gsp.git")


def test_setup_builds_a_venv_per_name_with_a_lock_hash(bare_repo_and_venvs: Sandbox):
    sandbox = bare_repo_and_venvs
    assert sorted(p.name for p in (sandbox.root / "envs").iterdir() if p.is_dir()) == [
        "cpu",
        "gpu",
    ]
    for name in ("cpu", "gpu"):
        venv = sandbox.root / "envs" / name / ".venv" / "bin" / "python"
        assert venv.exists(), f"{name} venv was not built"
        assert (sandbox.root / "envs" / name / "lock.sha256").read_text().strip()


def test_setup_syncs_each_venv_once_deps_only_with_cuda_for_gpu(
    bare_repo_and_venvs: Sandbox,
):
    syncs = bare_repo_and_venvs.uv_syncs()
    assert len(syncs) == 2
    assert any("--extra cuda" in s for s in syncs)
    assert all("--no-install-workspace" in s for s in syncs), (
        "the venvs must stay deps-only"
    )


def test_setup_cleans_up_the_scratch_snapshot(bare_repo_and_venvs: Sandbox):
    assert not (bare_repo_and_venvs.root / "runs" / ".setup").exists()


def test_setup_is_idempotent_and_does_not_resync(sandbox: Sandbox):
    setup_cluster(sandbox)
    setup_cluster(sandbox)
    assert len(sandbox.uv_syncs()) == 2, "an unchanged lock must not rebuild the venvs"


def test_setup_rejects_a_dirty_tree(sandbox: Sandbox):
    (sandbox.repo / "marker.txt").write_text("uncommitted")
    proc = setup_cluster(sandbox, expect_ok=False)
    assert proc.returncode != 0
    assert "working tree is dirty" in proc.stderr


# --- dispatch (requirement 2) ------------------------------------------------------


def test_dispatch_snapshots_the_commit_into_the_run_dir(
    dispatched_sweep: SimpleNamespace,
):
    sha, rundir, sandbox = (
        dispatched_sweep.sha,
        dispatched_sweep.rundir,
        dispatched_sweep.sandbox,
    )
    assert rundir.name.startswith("toy_") and rundir.name.endswith(sha[:7])
    assert (rundir / "marker.txt").read_text() == "v1", "snapshot is not the commit"

    # Results are redirected to the experiment's shared cluster directory, which is what
    # makes sync one rsync and lets runs accumulate across commits.
    link = rundir / "experiments" / "toy" / "results"
    assert link.is_symlink()
    assert link.resolve() == (sandbox.root / "results" / "toy").resolve()


def test_dispatch_requests_the_configured_sbatch_resources(
    dispatched_sweep: SimpleNamespace,
):
    array = dispatched_sweep.array
    assert "--array=0-2" in array
    assert "--account=def-test" in array
    assert "--time=01:00:00" in array
    assert "--mem-per-cpu=4G" in array
    assert not any(a.startswith("--gpus") for a in array), (
        "gpus = 0 must not ask for GPUs"
    )


def test_dispatch_chains_plan_array_and_merge_as_dependencies(
    dispatched_sweep: SimpleNamespace,
):
    plan, array, merge = (
        dispatched_sweep.plan,
        dispatched_sweep.array,
        dispatched_sweep.merge,
    )
    # The array waits on the plan, and the merge waits on the array.
    assert not any(a.startswith("--dependency") for a in plan)
    assert "--dependency=afterok:1001" in array
    assert "--dependency=afterok:1002" in merge


def test_dispatch_wrap_commands_run_the_snapshot_with_its_own_pythonpath(
    dispatched_sweep: SimpleNamespace,
):
    sandbox, rundir = dispatched_sweep.sandbox, dispatched_sweep.rundir
    plan, array, merge = (
        dispatched_sweep.plan,
        dispatched_sweep.array,
        dispatched_sweep.merge,
    )

    wrap = sandbox.wrap_of(array)
    assert f"cd {rundir}" in wrap
    assert f"PYTHONPATH={rundir}/src" in wrap, "the snapshot's own src must win"
    assert f"{sandbox.root}/envs/cpu/.venv/bin/python" in wrap
    # A task is told only which share is its own; the plan holds the rest.
    assert f"run.py sweep --plan {rundir}/plan.pickle" in wrap
    # Must reach the compute node unexpanded, for sbatch to substitute per task.
    assert "--worker-index $SLURM_ARRAY_TASK_ID" in wrap

    assert f"--write-plan {rundir}/plan.pickle" in sandbox.wrap_of(plan)
    assert "--num-workers 3" in sandbox.wrap_of(plan)
    assert "run.py sweep --merge-only" in sandbox.wrap_of(merge)


def test_dispatch_writes_the_job_state_file(dispatched_sweep: SimpleNamespace):
    state = dispatched_sweep.sandbox.state()
    assert state["sha"] == dispatched_sweep.sha
    assert state["venv"] == "cpu"
    assert state["jobs"] == {"plan": "1001", "array": "1002", "merge": "1003"}


def test_dispatch_rejects_a_dirty_tree(sandbox: Sandbox):
    setup_cluster(sandbox)
    (sandbox.repo / "marker.txt").write_text("uncommitted")

    proc = run_py(sandbox, "sweep", "--num-workers", "2", "--slurm", expect_ok=False)

    assert proc.returncode != 0
    assert "working tree is dirty" in proc.stderr
    assert sandbox.run_dirs == []
    assert sandbox.sbatch_calls() == []


def test_sweep_requires_num_workers(sandbox: Sandbox):
    setup_cluster(sandbox)
    proc = run_py(sandbox, "sweep", "--slurm", expect_ok=False)
    assert "--num-workers" in proc.stderr


def test_passthrough_args_reach_the_plan_job(sandbox: Sandbox):
    """Filters and overrides shape the plan, so only the plan job needs them."""
    setup_cluster(sandbox)
    run_py(
        sandbox,
        "sweep",
        "--num-workers",
        "2",
        "--component",
        "comp",
        "--shard-size",
        "1",
        "--slurm",
    )

    plan, array, merge = sandbox.sbatch_calls()
    plan_wrap = sandbox.wrap_of(plan)
    assert "--component comp" in plan_wrap
    assert "--shard-size 1" in plan_wrap
    # A task reads the plan, so repeating the filter to it would be redundant -
    # and a chance for the two to disagree.
    assert "--component" not in sandbox.wrap_of(array)
    assert "--component" not in sandbox.wrap_of(merge)


def test_single_dispatches_one_job(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "single", "--seed", "0", "--slurm")

    (call,) = sandbox.sbatch_calls()
    assert "--job-name=single" in call
    assert not any(a.startswith("--array") for a in call)
    assert "run.py single --seed 0" in sandbox.wrap_of(call)
    assert sandbox.state()["jobs"] == {"single": "1001"}


def test_gpu_experiment_uses_the_gpu_venv_and_asks_for_gpus(sandbox: Sandbox):
    """Per-experiment overrides in cluster.toml select resources and the venv."""
    setup_cluster(sandbox)
    # Relabel the toy experiment so [experiments.gpu_toy] applies to it.
    run_py_path = sandbox.repo / "experiments" / "toy" / "run.py"
    run_py_path.write_text(
        run_py_path.read_text().replace('name="toy"', 'name="gpu_toy"')
    )
    _commit(sandbox.repo, "gpu label")

    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")

    plan, array, merge = sandbox.sbatch_calls()
    assert "--gpus-per-node=2" in array
    assert "--time=12:00:00" in array, "per-experiment override must beat the default"
    assert f"{sandbox.root}/envs/gpu/.venv/bin/python" in sandbox.wrap_of(array)

    # Planning and merging use no GPU, so neither may queue for one - the array
    # would then wait behind a job holding hardware it never needed.
    for cheap in (plan, merge):
        assert not any(a.startswith("--gpus") for a in cheap)
        assert "--time=12:00:00" in cheap, "other resources are still inherited"


def test_per_experiment_account_override_beats_the_default(sandbox: Sandbox):
    """[experiments.<label>] account overrides [cluster] account for that label only."""
    setup_cluster(sandbox)
    run_py_path = sandbox.repo / "experiments" / "toy" / "run.py"
    run_py_path.write_text(
        run_py_path.read_text().replace('name="toy"', 'name="acct_toy"')
    )
    _commit(sandbox.repo, "acct label")

    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")

    for call in sandbox.sbatch_calls():
        assert "--account=acct-test" in call
        assert "--account=def-test" not in call


def test_dry_run_submits_nothing_and_leaves_no_run_dir(sandbox: Sandbox):
    setup_cluster(sandbox)
    proc = run_py(sandbox, "sweep", "--num-workers", "3", "--slurm-dry-run")

    assert sandbox.sbatch_calls() == []
    assert sandbox.run_dirs == []
    assert not (sandbox.repo / ".cluster" / "toy.json").exists()
    assert "--array=0-2" in proc.stdout, "the sbatch commands should still be shown"
    assert "--dependency=afterok:" in proc.stdout


def test_status_is_local_and_refuses_slurm(sandbox: Sandbox):
    """status answers from the results dir; queue is the cluster question."""
    setup_cluster(sandbox)
    before = sandbox.sbatch_calls()

    assert "2 run(s)" in run_py(sandbox, "status").stdout

    proc = run_py(sandbox, "status", "--slurm", expect_ok=False)
    assert "runs here, not on the cluster" in proc.stderr
    assert sandbox.sbatch_calls() == before, "status must not submit anything"
    assert sandbox.run_dirs == [], "status must not snapshot anything"


def test_venv_is_rebuilt_only_when_the_lock_changes(sandbox: Sandbox):
    setup_cluster(sandbox)
    assert len(sandbox.uv_syncs()) == 2

    (sandbox.repo / "marker.txt").write_text("v2")
    _commit(sandbox.repo, "code only")
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    assert len(sandbox.uv_syncs()) == 2, "a code change must not rebuild a venv"

    (sandbox.repo / "uv.lock").write_text("version = 1\nchanged = true\n")
    _commit(sandbox.repo, "bump deps")
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    assert len(sandbox.uv_syncs()) == 3, "a lock change must re-sync before running"

    # Each snapshot still carries its own source.
    assert {(d / "marker.txt").read_text() for d in sandbox.run_dirs} == {"v2"}


# --- sync (requirement 3) ----------------------------------------------------------


def test_sync_merges_cluster_results_without_overwriting_local_ones(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["cluster-1", "cluster-2"])
    _seed_db(sandbox.results_dir / "toy.db", ["local-0"])

    run_py(sandbox, "sync")

    assert _run_ids(sandbox.results_dir / "toy.db") == {
        "local-0",
        "cluster-1",
        "cluster-2",
    }
    assert not list((sandbox.results_dir / "comp.parts").glob("*.db")), (
        "consolidate removes the parts it merged"
    )


def test_sync_creates_the_local_db_when_there_is_none(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["cluster-1"])

    run_py(sandbox, "sync")

    assert _run_ids(sandbox.results_dir / "toy.db") == {"cluster-1"}


def test_sync_is_idempotent(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["cluster-1", "cluster-2"])

    run_py(sandbox, "sync")
    run_py(sandbox, "sync")

    assert _run_ids(sandbox.results_dir / "toy.db") == {"cluster-1", "cluster-2"}


def test_sync_needs_no_run_id_and_spans_commits(sandbox: Sandbox):
    """Results are per-experiment on the cluster, so one sync covers every commit."""
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["from-commit-1"])

    (sandbox.repo / "marker.txt").write_text("v2")
    _commit(sandbox.repo, "second commit")
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    # The second snapshot writes into the same shared dir.
    _seed_db(sandbox.root / "results" / "toy" / "toy.db", ["from-commit-2"])

    run_py(sandbox, "sync")

    assert _run_ids(sandbox.results_dir / "toy.db") == {
        "from-commit-1",
        "from-commit-2",
    }


def test_sync_warns_when_a_sweep_is_still_in_flight(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["cluster-1"])
    (sandbox.root / "results" / "toy" / "toy.parts").mkdir()

    proc = run_py(sandbox, "sync")

    assert "still in flight" in proc.stderr
    assert _run_ids(sandbox.results_dir / "toy.db") == {"cluster-1"}


def test_sync_errors_before_anything_has_run(sandbox: Sandbox):
    setup_cluster(sandbox)
    proc = run_py(sandbox, "sync", expect_ok=False)
    assert "nothing on the cluster" in proc.stderr


# --- wipe --------------------------------------------------------------------------


def test_wipe_removes_the_remote_results_dir(sandbox: Sandbox, monkeypatch):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    _seed_cluster_results(sandbox, ["cluster-1"])
    monkeypatch.setenv("HOME", str(sandbox.home))
    monkeypatch.setenv("EXPERIMENT_LOCAL_MODE", "1")
    slurm._ROOT_CACHE.clear()  # keyed on the literal "$HOME/..." string, stale otherwise

    slurm.wipe(label="toy", config_path=sandbox.repo / "cluster.toml")

    assert not (sandbox.root / "results" / "toy").exists()


def test_wipe_is_a_no_op_when_nothing_is_there(sandbox: Sandbox, monkeypatch):
    setup_cluster(sandbox)
    monkeypatch.setenv("HOME", str(sandbox.home))
    monkeypatch.setenv("EXPERIMENT_LOCAL_MODE", "1")
    slurm._ROOT_CACHE.clear()

    slurm.wipe(label="toy", config_path=sandbox.repo / "cluster.toml")  # must not raise


# --- is_queued -----------------------------------------------------------------------


def _is_queued_proc(sandbox: Sandbox, env: dict | None = None):
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiment import slurm; print(slurm.is_queued(label='toy'))",
        ],
        cwd=str(sandbox.repo),
        env={**os.environ, **(env or sandbox.env)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_is_queued_is_false_once_nothing_is_in_squeue(sandbox: Sandbox):
    """QUIET_STUB's squeue always reports nothing queued or running."""
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")

    proc = _is_queued_proc(sandbox)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_is_queued_is_true_while_squeue_reports_a_job(sandbox: Sandbox, tmp_path: Path):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    busy_bin = tmp_path / "busybin"
    busy_bin.mkdir()
    (busy_bin / "squeue").write_text(
        "#!/usr/bin/env bash\necho '1001 sweep RUNNING 0:01 node1'\n"
    )
    (busy_bin / "squeue").chmod(0o755)
    env = {**sandbox.env, "PATH": f"{busy_bin}:{sandbox.env['PATH']}"}

    proc = _is_queued_proc(sandbox, env)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True"


def test_is_queued_requires_a_dispatch_first(sandbox: Sandbox):
    setup_cluster(sandbox)

    proc = _is_queued_proc(sandbox)

    assert proc.returncode != 0
    assert "nothing dispatched yet" in proc.stderr


# --- status and logs ---------------------------------------------------------------


def test_queue_reports_the_recorded_jobs(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")

    proc = run_py(sandbox, "queue")

    assert "1001,1002,1003" in proc.stdout
    assert "squeue" in proc.stdout and "sacct" in proc.stdout


def test_logs_fetches_the_run_directory_output(sandbox: Sandbox):
    setup_cluster(sandbox)
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")
    (rundir,) = sandbox.run_dirs
    (rundir / "logs" / "sweep-1001_0.out").write_text("task 0 output\n")

    listing = run_py(sandbox, "logs")
    assert "sweep-1001_0.out" in listing.stdout

    shown = run_py(sandbox, "logs", "0")
    assert "task 0 output" in shown.stdout


def test_an_expired_ssh_session_says_so(sandbox: Sandbox, monkeypatch, tmp_path: Path):
    """Alliance MFA cannot be answered by a subprocess, so an expired ControlMaster
    surfaces on whatever remote call runs first. That must read as "log in again", not
    as a failure of that particular command."""
    ssh = tmp_path / "authbin" / "ssh"
    ssh.parent.mkdir()
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'parham1@vulcan.alliancecan.ca: Permission denied '"
        "'(keyboard-interactive).' >&2\n"
        "exit 255\n"
    )
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{ssh.parent}:{os.environ['PATH']}")
    # exercise the real ssh path
    monkeypatch.delenv("EXPERIMENT_LOCAL_MODE", raising=False)

    cfg = slurm.load_config(sandbox.repo / "cluster.toml")
    with pytest.raises(SystemExit) as excinfo:
        slurm._ssh(cfg, "printf hello")

    message = str(excinfo.value)
    assert "ssh to testcluster was refused" in message
    assert "ssh testcluster true" in message, "must name the fix, not just the symptom"
    assert "printf hello" not in message, (
        "must not blame the command that happened to run"
    )


def test_cluster_modes_require_a_dispatch_first(sandbox: Sandbox):
    setup_cluster(sandbox)
    for mode in ("queue", "logs"):
        proc = run_py(sandbox, mode, expect_ok=False)
        assert "nothing dispatched yet" in proc.stderr


# --- post-sync hook ------------------------------------------------------------------

POST_SYNC_HOOK = """\
#!/usr/bin/env bash
echo "$EXPERIMENT_VENV" >> "$GSP_TEST_CALLS/post_sync.venvs"
echo "$EXPERIMENT_EXTRAS" > "$GSP_TEST_CALLS/post_sync.$(basename $(dirname \
    "$EXPERIMENT_VENV"))"
"""


HOOK_PATH = "scripts/hook.sh"


def _name_post_sync(sandbox: Sandbox, path: str) -> None:
    config = sandbox.repo / "cluster.toml"
    config.write_text(f'[project]\npost_sync = "{path}"\n\n' + config.read_text())
    _commit(sandbox.repo, "add a post-sync hook")


def _with_post_sync(sandbox: Sandbox) -> None:
    (sandbox.repo / HOOK_PATH).write_text(POST_SYNC_HOOK)
    _name_post_sync(sandbox, HOOK_PATH)


def test_the_post_sync_hook_is_told_the_extras_its_venv_was_built_with(
    sandbox: Sandbox,
):
    _with_post_sync(sandbox)

    setup_cluster(sandbox)

    assert (sandbox.calls / "post_sync.cpu").read_text().strip() == "", (
        "the cpu venv is built with no extras"
    )
    assert (sandbox.calls / "post_sync.gpu").read_text().strip() == "cuda", (
        "the gpu venv is built with the cuda extra"
    )


def test_the_post_sync_hook_is_told_the_venv_it_must_install_into(sandbox: Sandbox):
    _with_post_sync(sandbox)

    setup_cluster(sandbox)

    envs = sandbox.root / "envs"
    assert (sandbox.calls / "post_sync.venvs").read_text().split() == [
        str(envs / "cpu" / ".venv"),
        str(envs / "gpu" / ".venv"),
    ]


def test_an_up_to_date_venv_does_not_rerun_the_post_sync_hook(sandbox: Sandbox):
    _with_post_sync(sandbox)
    setup_cluster(sandbox)

    setup_cluster(sandbox)

    venvs = (sandbox.calls / "post_sync.venvs").read_text().split()
    assert len(venvs) == 2, f"the hook reran for an unchanged venv: {venvs}"


def test_editing_the_post_sync_hook_rebuilds_the_venv(sandbox: Sandbox):
    _with_post_sync(sandbox)
    setup_cluster(sandbox)
    assert len(sandbox.uv_syncs()) == 2, "setup builds both venvs"

    (sandbox.repo / HOOK_PATH).write_text(POST_SYNC_HOOK + "echo edited\n")
    _commit(sandbox.repo, "edit the hook")
    run_py(sandbox, "sweep", "--num-workers", "2", "--slurm")

    assert len(sandbox.uv_syncs()) == 3, "a changed hook must rebuild the venv"


def test_a_post_sync_script_the_commit_lacks_is_refused(sandbox: Sandbox):
    setup_cluster(sandbox)
    _name_post_sync(sandbox, "scripts/missing.sh")

    proc = run_py(sandbox, "sweep", "--num-workers", "2", "--slurm", expect_ok=False)

    assert proc.returncode != 0, "a post_sync script the commit lacks must be refused"
    assert "scripts/missing.sh" in proc.stderr


def test_this_repos_post_sync_script_is_committed():
    config = tomllib.loads((_REPO / "cluster.toml").read_text())

    post_sync = config["project"]["post_sync"]

    assert (_REPO / post_sync).is_file(), f"cluster.toml names {post_sync}"


def test_the_ale_installer_leaves_a_venv_without_the_atari_extra_alone():
    proc = subprocess.run(
        ["bash", str(_REPO / "scripts" / "install_ale_wheel.sh")],
        env={
            **os.environ,
            "EXPERIMENT_EXTRAS": "cuda",
            "EXPERIMENT_VENV": "/nonexistent/.venv",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "nothing to install" in proc.stdout
