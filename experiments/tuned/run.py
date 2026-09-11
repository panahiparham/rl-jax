"""Run: DQN/DDQN at each tuning env's best LR + Random baselines.

Cheat sheet:
    Status: uv run experiments/tuned/run.py status
    Sweep: uv run experiments/tuned/run.py sweep --num-workers 13
    Single: uv run experiments/tuned/run.py single --component dqn_cartpole --seed 10
    On the cluster: add --slurm to single or sweep, then sync
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force single-threaded CPU XLA BEFORE the first jax import (below, via main), so
# N local worker processes don't each grab every core and thrash.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
).strip()

# This dir on sys.path, for `import config`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment.commands import run
from main import process_shard

from config import EXPERIMENT


def entry() -> None:
    run(EXPERIMENT, process_shard)


if __name__ == "__main__":
    entry()
