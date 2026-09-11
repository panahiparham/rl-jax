# atari_200m

Atari **Pong**, 50M steps (200M frames at `FRAMESKIP=4`) - the classic Nature-DQN
protocol, full length. The 10x-longer counterpart of `atari_20m`; same two
components, same hypers, saved to their own databases:

* `dqn_pong` - DQN with `atari_20m`'s hypers (itself reproducing
  `qrc-at-scale/experiments/atari-20m/Pong/dqn.json`, see `config.py` for the
  field-by-field mapping) at `TOTAL_TIMESTEPS=50_000_000` instead of 5M. Uses the
  `nature_cnn` Q-network.
* `random_pong` - the uniform-random agent, a return baseline.

Both use ale-py's XLA env. 1 seed each, no sweep.

**This experiment has not been run.** It exists as a scaffold with SLURM settings
sized from `atari_20m`'s measured throughput - see PR #7.

## Setup

Atari needs the optional `atari` extra with ale-py's PR-#707 XLA build:

```bash
./scripts/install_ale_wheel.sh          # macOS-CPU / Linux-CPU
./scripts/install_ale_wheel.sh --cuda   # Linux-CUDA
```

(See the repo README "Atari (XLA) setup".)

## Scale ⚠️

**Cluster-scale, 10x longer than `atari_20m`** - 50M agent steps. `dqn_pong`'s
`BUFFER_SIZE` is 100k rather than the faithful 1M, for the same reason as
`atari_20m`: obs+next_obs at (84,84,4) uint8 would need ~56GB at 1M, more than a
Vulcan L40S's 48GB. Meant for **Linux-CUDA**, not a laptop. Atari's ale-py env
can't be `jax.vmap`'d, so `src/main.py` reads that from the
environment registry and runs the shard's runs one at a time; both components set
`shard_size=1` (one env + buffer per process) and work is spread across processes
with `--num-workers`.

**No mid-run checkpointing.** A component's whole run is one `jax.lax.scan` -
there is no partial result if a job times out or is preempted. `cluster.toml`'s
`atari_200m` time budget is sized with that failure mode in mind (see the comment
there for the throughput extrapolation and bucket trade-off).

Also note: on **macOS-CPU** the ale-py XLA FFI can intermittently segfault at
episode boundaries under the DQN graph (stable on Linux-CUDA).

## Usage

```bash
uv run python experiments/atari_200m/run.py status
uv run python experiments/atari_200m/run.py sweep --num-workers 2   # 2 components, one process each

# Quick local check (tiny run - override the heavy hypers):
uv run python experiments/atari_200m/run.py single --component dqn_pong --seed 0 \
    --set AGENT_HYPERS.TOTAL_TIMESTEPS=500 --set AGENT_HYPERS.BUFFER_SIZE=1000
```

> `--set PATH=VALUE` overrides a config field by its dotted path, and reads the
> value as the type of the field it replaces. Editing `config.py` avoids the CLI
> entirely.

## Analysis

Results store per-timestep `reward`/`done` curves;
`analysis.ipynb` overlays `dqn_pong` vs `random_pong` as episodic-return
curves (with the CI band collapsing to the line at 1 seed).
