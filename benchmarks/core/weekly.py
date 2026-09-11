"""
CLI: check/dispatch/finish the weekly benchmark - no chat session needed.

Cheat sheet:
    Check:    uv run benchmarks/core/weekly.py check
    Dispatch: uv run benchmarks/core/weekly.py dispatch
    Finish:   uv run benchmarks/core/weekly.py finish

`check`'s exit code is what a cron job or a human branches on:
    0  - idle, or a dispatch is still queued: nothing to do
    10 - a fresh run is due: run `dispatch`
    20 - a dispatch finished: run `finish`
    30 - a dispatch is in flight but Vulcan can't be reached right now

`dispatch` exits 30 for that same reason - a lapsed MFA session - so a caller
can treat it exactly as it treats `check`'s 30.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarking import weekly

from config import ENVIRONMENTS, EXPERIMENT, PLOTS_DIR

LABEL = EXPERIMENT.name
STATE_PATH = _REPO_ROOT / "benchmarks" / "state.json"
RUN_PY = Path(__file__).resolve().parent / "run.py"

_DISPATCH_WORKERS = 1

_EXIT_BY_STATUS = {
    weekly.STATUS_IDLE: 0,
    weekly.STATUS_QUEUED: 0,
    weekly.STATUS_DUE: 10,
    weekly.STATUS_READY_TO_FINISH: 20,
    weekly.STATUS_UNREACHABLE: 30,
}


def _check() -> int:
    status = weekly.check_status(label=LABEL, state_path=STATE_PATH, repo_root=_REPO_ROOT)
    print(f"[{status.status}] {status.detail}")
    return _EXIT_BY_STATUS[status.status]


def _dispatch(num_workers: int) -> int:
    try:
        weekly.dispatch(label=LABEL, run_py=RUN_PY,
                        results_dir=EXPERIMENT.results_dir,
                        num_workers=num_workers)
    except weekly.AuthRefusedError as exc:
        print(f"[{LABEL}] {exc}", file=sys.stderr)
        return _EXIT_BY_STATUS[weekly.STATUS_UNREACHABLE]
    print(f"[{LABEL}] dispatched - run `check` again once the jobs finish")
    return 0


def _finish() -> int:
    sha = weekly.finish(experiment=EXPERIMENT, environments=ENVIRONMENTS,
                        run_py=RUN_PY, plots_dir=PLOTS_DIR,
                        state_path=STATE_PATH, repo_root=_REPO_ROOT)
    url = weekly.open_pr(label=LABEL,
                         plots_path=PLOTS_DIR.relative_to(_REPO_ROOT),
                         repo_root=_REPO_ROOT)
    print(f"[{LABEL}] benchmarked {sha[:7]}, opened {url}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="weekly.py")
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("check")
    dispatch_ap = sub.add_parser("dispatch")
    dispatch_ap.add_argument("--num-workers", type=int,
                             default=_DISPATCH_WORKERS)
    sub.add_parser("finish")
    args = ap.parse_args()

    if args.mode == "check":
        return _check()
    if args.mode == "dispatch":
        return _dispatch(args.num_workers)
    return _finish()


if __name__ == "__main__":
    sys.exit(main())
