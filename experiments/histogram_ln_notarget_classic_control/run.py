"""
Run: HL-Gauss DQN grid sweep (SIGMA_RATIO, NUM_BINS, support, LR) on Cartpole, Acrobot
and MountainCar, without LayerNorm vs. with LayerNorm and no target network.

Cheat sheet:
    Status: uv run experiments/histogram_ln_notarget_classic_control/run.py status
    Sweep: uv run experiments/histogram_ln_notarget_classic_control/run.py sweep --num-workers 3
    Single: uv run experiments/histogram_ln_notarget_classic_control/run.py single --component dqn_hl_ln_notarget_cartpole_support100
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
