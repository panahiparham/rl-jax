"""Components for the ``atari_20m`` experiment (Atari Pong, 5M steps).

One component on Atari Pong:

* ``dqn_pong`` - DQN reproducing ``qrc-at-
scale/experiments/atari-20m/Pong/dqn.json`` (2 seeds).

The ``random_pong`` baseline is dropped for now (see git history to restore it) -
add it back once it has actually been run, so ``analysis.ipynb`` has both series'
data to overlay.

Atari's ale-py env is a stateful FFI that can't be ``jax.vmap``'d. The shard
function sees that in the environment registry and runs the shard's runs one at a
time; ``shard_size=1`` keeps it to one ~GB-scale replay buffer per process.

⚠️ Faithful to the json this is **cluster-scale** (5M steps) - meant for Linux-CUDA,
not a laptop. ``dqn_pong``'s ``BUFFER_SIZE`` is 100k rather than the json's 1M: a
1M×(84,84,4) uint8 replay needs ~56GB obs+next_obs, more than a Vulcan L40S's 48GB.
Needs the ``atari`` extra; see ``scripts/install_ale_wheel.sh``. For a quick
local check, override on the CLI:
    uv run python experiments/atari_20m/run.py single --component dqn_pong --seed 0 \\
        --set AGENT_HYPERS.TOTAL_TIMESTEPS=300 --set AGENT_HYPERS.BUFFER_SIZE=1000
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment

from agents.dqn import DQNConfig
from environments.atari import AtariConfig
from main import ExperimentConfig

_ATARI_PONG = AtariConfig(
    GAME="pong",                     # json: environment_settings.game
    FRAMESKIP=4,                     # json: environment_settings.frameskip
    STICKY_ACTIONS=0.25,             # json: environment_settings.sticky_actions
    EPISODE_CUTOFF=27_000,           # json: EPISODE_CUTOFF (agent steps)
)

EXPERIMENT = Experiment(
    name="atari_20m",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="dqn_pong",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="atari",
                AGENT_HYPERS=DQNConfig(
                    TOTAL_TIMESTEPS=5_000_000,       # json: TOTAL_TIMESTEPS
                    LR=6.25e-05,                     # json: metaParameters.LR
                    ADAM_EPS=1.5e-4,                 # json: metaParameters.ADAM_EPS
                    BUFFER_SIZE=100_000,
                    BATCH_SIZE=32,                   # json: BATCH_SIZE
                    LEARNING_STARTS=20_000,          # json: LEARNING_STARTS
                    TRAIN_FREQUENCY=4,               # json: TRAIN_FREQUENCY
                    TARGET_NETWORK_FREQUENCY=8_000,  # json: TARGET_NETWORK_FREQUENCY
                    GAMMA=0.99,                      # json: GAMMA
                    EPSILON_START=1.0,               # json: EPSILON_START
                    EPSILON_END=0.01,                # json: EPSILON_END
                    EPSILON_FRACTION=0.05,           # json: EPSILON_FRACTION
                    NETWORK_PRESET="nature_cnn",     # json: NETWORK_PRESET
                ),
                ENV_HYPERS=_ATARI_PONG,
            ),
            seeds=[0, 1],
            shard_size=1,  # one env, and one replay buffer, per process
        ),
    ],
)
