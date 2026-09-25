"""Components for the ``atari_200m`` experiment (Atari Pong, 50M steps, 1 seed).

The full-length counterpart of ``atari_20m``: the classic Nature-DQN protocol of
50M agent steps (200M emulator frames at ``FRAMESKIP=4``), instead of the shorter
20M-frame variant. Two components on Atari Pong, each saved to its own database so
``analysis.ipynb`` can overlay them:

* ``dqn_pong``    - DQN with the same hypers as ``atari_20m``'s ``dqn_pong``
  (itself reproducing ``qrc-at-scale/experiments/atari-20m/Pong/dqn.json``), just
  10x the steps.
* ``random_pong`` - the uniform-random agent, a return baseline.

Atari's ale-py env is a stateful FFI that can't be ``jax.vmap``'d. The shard
function sees that in the environment registry and runs the shard's runs one at a
time; ``shard_size=1`` keeps it to one ~GB-scale replay buffer per process.

⚠️ Cluster-scale, and longer than ``atari_20m`` by 10x - meant for Linux-CUDA, not a
laptop. ``dqn_pong``'s ``BUFFER_SIZE`` is 100k rather than the 1M a faithful
reproduction would use: a 1M×(84,84,4) uint8 replay needs ~56GB obs+next_obs, more
than a Vulcan L40S's 48GB. Needs the ``atari`` extra; see
``scripts/install_ale_wheel.sh``. For a quick local check, override on the CLI:
    uv run python experiments/atari_200m/run.py single --component dqn_pong \\
        --seed 0 --set AGENT_HYPERS.TOTAL_TIMESTEPS=300 \\
        --set AGENT_HYPERS.BUFFER_SIZE=1000

This experiment has not been run - see the atari_20m PR (#7) for a 5M-step trial
run's measured throughput, which this experiment's SLURM settings extrapolate from.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment

from agents.dqn import DQNConfig
from agents.random import RandomConfig
from environments.atari import RevisitingALEConfig
from main import ExperimentConfig

_ATARI_PONG = RevisitingALEConfig(
    GAME="pong",                     # json: environment_settings.game
    FRAMESKIP=4,                     # json: environment_settings.frameskip
    STICKY_ACTIONS=0.25,             # json: environment_settings.sticky_actions
    MAX_FRAMES_PER_EPISODE=108_000,  # json: EPISODE_CUTOFF=27_000 agent steps
)

EXPERIMENT = Experiment(
    name="atari_200m",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(
            name="dqn_pong",
            config=ExperimentConfig(
                AGENT="dqn",
                ENV="atari",
                AGENT_HYPERS=DQNConfig(
                    TOTAL_TIMESTEPS=50_000_000,      # 200M frames at FRAMESKIP=4
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
                    EPSILON_DECAY_STEPS=250_000,
                    NETWORK_PRESET="nature_cnn",     # json: NETWORK_PRESET
                ),
                ENV_HYPERS=_ATARI_PONG,
            ),
            seeds=[0],
            shard_size=1,  # one env, and one replay buffer, per process
        ),
        Component(
            name="random_pong",
            config=ExperimentConfig(
                AGENT="random",
                ENV="atari",
                AGENT_HYPERS=RandomConfig(TOTAL_TIMESTEPS=50_000_000),
                ENV_HYPERS=_ATARI_PONG,
            ),
            seeds=[0],
            shard_size=1,
        ),
    ],
)
