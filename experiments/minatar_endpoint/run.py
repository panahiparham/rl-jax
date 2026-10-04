"""
Run: DDQN buffer sizes vs endpoint and unanchored replay on MinAtar (Asterix,
Breakout, Freeway, Space Invaders), on deterministic GPU kernels.

Cheat sheet:
    Status: uv run experiments/minatar_endpoint/run.py status
    Sweep: uv run experiments/minatar_endpoint/run.py sweep --num-workers 28
    On the cluster: add --slurm to sweep, then sync
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force single-threaded CPU XLA BEFORE the first jax import (below, via main), so
# N local worker processes don't each grab every core and thrash.
# Deterministic GPU kernels make same-seed runs reproduce exactly.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + " --xla_cpu_multi_thread_eigen=false"
    + " --xla_gpu_deterministic_ops=true"
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
