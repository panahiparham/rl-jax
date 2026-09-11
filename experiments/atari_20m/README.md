# atari_20m

Atari **Pong**, 5M steps, one component:

* `dqn_pong` - DQN reproducing `qrc-at-scale/experiments/atari-20m/Pong/dqn.json`
  (see `config.py` for the field-by-field mapping). Uses the `nature_cnn` Q-network.
  2 seeds.

Uses ale-py's XLA env. No sweep.

> The `random_pong` baseline (uniform-random agent) is dropped for now - it was
> never run, and `analysis.ipynb` needs both components' data to overlay them.
> See git history to restore it once it has actually been run.

## Setup

Atari needs the optional `atari` extra with ale-py's PR-#707 XLA build:

```bash
./scripts/install_ale_wheel.sh          # macOS-CPU / Linux-CPU
./scripts/install_ale_wheel.sh --cuda   # Linux-CUDA
```

(See the repo README "Atari (XLA) setup".)

## Scale ⚠️

Faithful to the json this is **cluster-scale** - 5M agent steps. `dqn_pong`'s
`BUFFER_SIZE` is 100k rather than the json's 1M: obs+next_obs at (84,84,4) uint8
would need ~56GB at 1M, more than a Vulcan L40S's 48GB. It is meant for
**Linux-CUDA**, not a laptop. Atari's ale-py env can't be `jax.vmap`'d,
so `src/main.py` reads that from the environment registry and runs the shard's
runs one at a time; the component sets `shard_size=1` (one env + buffer per
process) and work is spread across processes with `--num-workers`.

Also note: on **macOS-CPU** the ale-py XLA FFI can intermittently segfault at
episode boundaries under the DQN graph (stable on Linux-CUDA).

## Usage

```bash
uv run python experiments/atari_20m/run.py status
uv run python experiments/atari_20m/run.py sweep --num-workers 2   # 2 seeds, one process each

# Quick local check (tiny run - override the heavy hypers):
uv run python experiments/atari_20m/run.py single --component dqn_pong --seed 0 \
    --set AGENT_HYPERS.TOTAL_TIMESTEPS=500 --set AGENT_HYPERS.BUFFER_SIZE=1000
```

> `--set PATH=VALUE` overrides a config field by its dotted path, and reads the
> value as the type of the field it replaces. Editing `config.py` avoids the CLI
> entirely.

## Analysis

Results store per-timestep `reward`/`done` curves;
`analysis.ipynb` plots `dqn_pong`'s 2 seeds as individual episodic-return
curves.
