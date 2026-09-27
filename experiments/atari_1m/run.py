import os
import sys
from pathlib import Path

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
).strip()
# Parallel shards share one GPU; preallocating would give the first process
# most of its memory.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# This dir on sys.path, for `import config`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment.commands import run
from main import process_shard

from config import EXPERIMENT


def entry() -> None:
    run(EXPERIMENT, process_shard)


if __name__ == "__main__":
    entry()
