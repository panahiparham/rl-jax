"""Shared SLURM-sandbox fixture for the cluster/weekly-benchmark tests.

``EXPERIMENT_LOCAL_MODE=1`` makes :mod:`experiment.slurm` run every "remote"
command in a local shell and drop the ``host:`` prefix from rsync and git paths,
so the real flow (push to the bare repo, ``git archive`` into a run dir, build
the shared venvs, submit, sync) runs against ``tmp_path``.
``sbatch``/``squeue``/``sacct``/``uv`` are stubs on ``PATH`` that record how they
were called, and ``HOME`` is redirected so account detection and the remote root
are the sandbox's rather than the developer's.

The sandbox is a real git repo holding a copy of ``src/`` plus a toy experiment,
so the code under test is this repo's, and ``repo_root()`` resolves to the
sandbox. ``experiment`` itself comes from the venv the sandbox's own ``uv sync``
would install, same as in production.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from experiment.results import _connect_write, _ensure_table
from experiment.slurm import _remote_dir

_REMOTE_DIR = _remote_dir()

# A minimal experiment: enough for plan/consolidate/sync and for --slurm dispatch, with
# no jax in the loop (the cluster side never actually executes, sbatch is a stub).
TOY_RUN_PY = '''\
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.commands import run
from experiment.design import Component, Experiment


@dataclass(frozen=True)
class ToyConfig:
    LR: float = 0.1


def toy_process(configs, seeds):
    """A jax-free shard function: the toy experiment stores no arrays."""
    del configs
    return [{} for _ in seeds]


EXPERIMENT = Experiment(
    name="toy",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[Component(name="comp", config=ToyConfig(), seeds=[0, 1],
                          shard_size=1)],
)


def entry() -> None:
    run(EXPERIMENT, toy_process)


if __name__ == "__main__":
    entry()
'''

CLUSTER_TOML = """\
[cluster]
host = "testcluster"
root = "$HOME/scratch/online-gsp"
account = "def-test"

[venvs]
cpu = []
gpu = ["cuda"]

[slurm]
time = "01:00:00"
cpus_per_task = 1
mem_per_cpu = "4G"
gpus = 0

[experiments.gpu_toy]
gpus = 2
time = "12:00:00"

[experiments.acct_toy]
account = "acct-test"
"""

SBATCH_STUB = """\
#!/usr/bin/env python3
import json, os, pathlib, sys

d = pathlib.Path(os.environ["GSP_TEST_CALLS"])
with (d / "sbatch.jsonl").open("a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
counter = d / "jobid"
n = int(counter.read_text()) if counter.exists() else 1000
counter.write_text(str(n + 1))
print(n + 1)   # --parsable: the job id and nothing else
"""

# `uv sync` here only has to materialise an interpreter where the caller asked for one.
UV_STUB = """\
#!/usr/bin/env bash
set -eu
echo "$*" >> "$GSP_TEST_CALLS/uv.log"
if [ "${1:-}" = "sync" ]; then
  [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] \
    || { echo "stub uv: no UV_PROJECT_ENVIRONMENT" >&2; exit 1; }
  mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
  ln -sf "$GSP_TEST_PYTHON" "$UV_PROJECT_ENVIRONMENT/bin/python"
fi
exit 0
"""

QUIET_STUB = "#!/usr/bin/env bash\nexit 0\n"


@dataclass
class Sandbox:
    repo: Path
    home: Path
    calls: Path
    env: dict[str, str]

    @property
    def root(self) -> Path:
        return self.home / "scratch" / "online-gsp"

    @property
    def run_dirs(self) -> list[Path]:
        runs = self.root / "runs"
        return sorted(p for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []

    @property
    def results_dir(self) -> Path:
        return self.repo / "experiments" / "toy" / "results"

    def sbatch_calls(self) -> list[list[str]]:
        path = self.calls / "sbatch.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def uv_syncs(self) -> list[str]:
        path = self.calls / "uv.log"
        lines = path.read_text().splitlines() if path.exists() else []
        return [line for line in lines if line.startswith("sync")]

    def state(self, label: str = "toy") -> dict:
        return json.loads((self.repo / ".cluster" / f"{label}.json").read_text())

    def wrap_of(self, call: list[str]) -> str:
        return call[call.index("--wrap") + 1]


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    home = tmp_path / "home"
    (home / "projects" / "def-test").mkdir(parents=True)
    # the trial allocation, must be ignored
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
        # The sandbox's own copy of src/, so repo_root() resolves to the sandbox.
        "PYTHONPATH": str(repo / "src"),
    }
    return Sandbox(repo=repo, home=home, calls=calls, env=env)


def run_py(sandbox: Sandbox, *args: str, expect_ok: bool = True):
    """Invoke the toy experiment's run.py exactly as a user would."""
    proc = subprocess.run(
        [sys.executable, "experiments/toy/run.py", *args],
        cwd=str(sandbox.repo),
        env={**os.environ, **sandbox.env},
        capture_output=True,
        text=True,
        check=False,
    )
    if expect_ok and proc.returncode != 0:
        pytest.fail(
            f"run.py {' '.join(args)} failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def setup_cluster(sandbox: Sandbox, expect_ok: bool = True):
    proc = subprocess.run(
        [sys.executable, "scripts/setup_cluster.py"],
        cwd=str(sandbox.repo),
        env={**os.environ, **sandbox.env},
        capture_output=True,
        text=True,
        check=False,
    )
    if expect_ok and proc.returncode != 0:
        pytest.fail(
            f"setup_cluster.py failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def _seed_db(path: Path, run_ids: list[str], component: str = "comp") -> None:
    """Write runs straight into one component's table of an experiment database."""
    conn = _connect_write(path)
    _ensure_table(conn, component)
    for i, rid in enumerate(run_ids):
        conn.execute(
            f'INSERT OR IGNORE INTO "{component}" VALUES (?, ?, ?, ?, ?)',
            (rid, "cfg", i, "{}", None),
        )
    conn.commit()
    conn.close()


def _run_ids(path: Path, component: str = "comp") -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {r[0] for r in conn.execute(f'SELECT run_id FROM "{component}"')}
    finally:
        conn.close()


def _seed_cluster_results(sandbox: Sandbox, run_ids: list[str]) -> None:
    _seed_db(sandbox.root / "results" / "toy" / "toy.db", run_ids)
