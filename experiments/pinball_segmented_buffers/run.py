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
