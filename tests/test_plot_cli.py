"""End-to-end test for benchmarks/core/plot.py's CLI wiring.

Only the no-results path runs here: with results on disk the script writes
into the repo's own benchmarks/core/plots, so the rendering itself is covered
against a tmp_path in test_report.py, at the benchmarking.report level.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PLOT_PY = _REPO / "benchmarks" / "core" / "plot.py"

unsynced_only = pytest.mark.skipif(
    (_REPO / "benchmarks" / "core" / "results").exists(),
    reason="local benchmark results would make plot.py write into the repo",
)


@unsynced_only
def test_plot_names_the_environments_it_has_no_results_for():
    proc = subprocess.run(
        [sys.executable, str(_PLOT_PY), "--env", "catch"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "no results yet for: catch" in proc.stderr
