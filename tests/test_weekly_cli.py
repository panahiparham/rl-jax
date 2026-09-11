"""End-to-end tests for benchmarks/core/weekly.py's CLI wiring.

Only ``check`` is exercised here - it is side-effect-free (no Vulcan ssh, no
git/gh). ``dispatch`` and ``finish`` would actually try to reach Vulcan or
open a live PR against this repo, so those paths are covered against the
sandbox in test_weekly.py instead, at the benchmarking.weekly function level.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_WEEKLY_PY = _REPO / "benchmarks" / "core" / "weekly.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_WEEKLY_PY), *args],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_runs_and_exits_with_a_known_status_code():
    proc = _run("check")

    assert proc.returncode in (0, 10, 20, 30), proc.stderr
    assert proc.stdout.startswith("["), proc.stdout


def test_no_subcommand_is_an_argparse_error():
    proc = _run()

    assert proc.returncode != 0
