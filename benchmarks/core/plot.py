"""
Plot: the weekly benchmark suite's learning curves, into benchmarks/core/plots.

Cheat sheet:
    All:  uv run benchmarks/core/plot.py
    Some: uv run benchmarks/core/plot.py --env catch --env cartpole

`finish` renders these as part of the weekly run - this is the same plots on
demand, for a suite whose results are already synced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarking.report import plot_environment

from config import ENVIRONMENTS, EXPERIMENT, PLOTS_DIR


def main() -> int:
    ap = argparse.ArgumentParser(prog="plot.py")
    ap.add_argument("--env", action="append", dest="envs", metavar="KEY",
                    choices=[e.key for e in ENVIRONMENTS])
    args = ap.parse_args()

    keys = args.envs or [e.key for e in ENVIRONMENTS]
    missing = []
    for environment in (e for e in ENVIRONMENTS if e.key in keys):
        path = plot_environment(EXPERIMENT, environment, PLOTS_DIR)
        if path is None:
            missing.append(environment.key)
            continue
        print(path.relative_to(_REPO_ROOT))

    if missing:
        print(f"no results yet for: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
