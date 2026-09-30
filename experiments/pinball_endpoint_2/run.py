"""Run: DDQN buffer sizes vs endpoint replay on Pinball(Easy).

Cheat sheet:
    Status: uv run experiments/pinball_endpoint_2/run.py status
    Sweep: uv run experiments/pinball_endpoint_2/run.py sweep --num-workers 12
    Single: uv run experiments/pinball_endpoint_2/run.py single --component endpoint
    On the cluster: add --slurm to single or sweep, then sync
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Before the first jax import (below, via main): single-threaded CPU XLA keeps
# local workers from thrashing, and deterministic GPU kernels make same-seed
# runs reproduce exactly.
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
