"""
Define: DQN with the HL-Gauss histogram loss on Classic Control environments
(Cartpole, MountainCar), sweeping SIGMA_RATIO - the Gaussian's width as a multiple
of the bin width - over 0.75, 1, 1.5, 2, and 2.5.

Every other hyper follows the ``experiments/classic_control`` DQN recipe, so its
``dqn_cartpole``/``dqn_mountaincar`` results are the MSE baseline. 30 seeds per
SIGMA_RATIO point.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.dqn_hl import DQNHistogramConfig
from environments.classic_control import CartpoleConfig, MountainCarConfig
from main import ExperimentConfig

SIGMA_RATIO_SWEEP = [0.75, 1.0, 1.5, 2.0, 2.5]
_SEEDS = list(range(30))


def _dqn_histogram_hypers(total_timesteps: int = 100_000) -> DQNHistogramConfig:
    # classic_control's DQN hypers, with the histogram agent's LayerNorm
    # network in place of NETWORK_PRESET="mlp".
    return DQNHistogramConfig(
        NETWORK_PRESET="mlp_hl_ln",
        TOTAL_TIMESTEPS=total_timesteps,
        LR=0.001,
        BUFFER_SIZE=10_000,
        BATCH_SIZE=64,
        LEARNING_STARTS=1_000,
        TRAIN_FREQUENCY=1,
        TARGET_NETWORK_FREQUENCY=128,
        GAMMA=0.99,
        EPSILON_START=0.1,
        EPSILON_END=0.1,
    )


EXPERIMENT = Experiment(
    name="histogram_sigma_classic_control",
    results_dir=Path(__file__).resolve().parent / "results",
    slurm=SlurmResources(time="02:59:00"),
    components=[
        Component(
            name="dqn_histogram_cartpole",
            config=ExperimentConfig(
                AGENT="dqn_histogram",
                ENV="cartpole",
                AGENT_HYPERS=_dqn_histogram_hypers(),
                ENV_HYPERS=CartpoleConfig(EPISODE_CUTOFF=500),
            ),
            sweep={"AGENT_HYPERS.SIGMA_RATIO": SIGMA_RATIO_SWEEP},
            seeds=_SEEDS,
            # SIGMA_RATIO isn't traced (it fixes a static network field), so each
            # value is its own shard of every seed.
            shard_size=None,
        ),
        Component(
            name="dqn_histogram_mountaincar",
            config=ExperimentConfig(
                AGENT="dqn_histogram",
                ENV="mountaincar",
                AGENT_HYPERS=_dqn_histogram_hypers(),
                ENV_HYPERS=MountainCarConfig(EPISODE_CUTOFF=1_000),
            ),
            sweep={"AGENT_HYPERS.SIGMA_RATIO": SIGMA_RATIO_SWEEP},
            seeds=_SEEDS,
            shard_size=None,
        ),
    ],
)
