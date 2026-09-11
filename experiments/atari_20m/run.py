"""Run the ``atari_20m`` experiment (components: dqn_pong, random_pong).

Thin wrapper: hands this experiment and the project's shard function to the shared
harness. Atari's ale-py FFI can't be vmapped, so the shard function runs its runs
one at a time; each component saves to its own table in ``results/atari_20m.db``.
Requires the Atari extra - see ``scripts/install_ale_wheel.sh``.

    uv run python experiments/atari_20m/run.py status
    uv run python experiments/atari_20m/run.py sweep --num-workers 2   # 2 components x
    1 seed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
