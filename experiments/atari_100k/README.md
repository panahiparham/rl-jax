# atari_100k

Atari **Pong**, 100k steps, one component:

* `dqn_pong` - DQN with the `nature_cnn` Q-network and a 1M-transition replay
  buffer. 10 seeds.

A small example of packing several Atari runs onto one GPU.

## Packing runs onto a GPU

Atari's ale-py env can't be `jax.vmap`'d, so each run is its own shard
(`shard_size=1`). Three settings let one GPU run several of them at once:

* `parallel_shards=5` in `config.py` - each worker runs 5 shards at the same
  time, each in its own process.
* `mps = true` in `cluster.toml` - the job starts a CUDA MPS server, so the
  processes' kernels run on the GPU concurrently instead of taking turns.
* `XLA_PYTHON_CLIENT_PREALLOCATE=false` in `run.py` - each process takes only
  the GPU memory it needs, instead of most of the GPU.

Each run takes about 8.6 GiB of GPU memory, mostly its replay buffer, so an
L40S (46 GB) fits 5. Measured on one L40S, 100k steps per run:

| Setup | Time per run | Throughput |
|---|---|---|
| 1 run | 2m26s | 1x |
| 4 runs, no MPS | about 4m03s | about 2.4x |
| 4 runs, MPS | about 2m06s | about 4.6x |
| 5 runs, MPS | about 2m19s | about 5.2x |

## Usage

```bash
uv run python experiments/atari_100k/run.py status
uv run python experiments/atari_100k/run.py sweep --num-workers 1 --slurm
uv run python experiments/atari_100k/run.py sweep --num-workers 1 --slurm \
    --parallel-shards 4             # override the config's parallel_shards
uv run python experiments/atari_100k/run.py sync
```

`status` suggests a `--num-workers` that gives every worker a full set of
parallel shards. One worker runs the 10 seeds in two rounds of 5, in about
5.5 minutes.
