"""
Define: DQN with the HL-Gauss histogram loss on Classic Control environments
(Cartpole, MountainCar), sweeping a grid of SIGMA_RATIO (σ as a multiple of the
bin width), NUM_BINS, support range, and learning rate.

Follows up ``experiments/histogram_sigma_classic_control``, whose SIGMA_RATIO-only
sweep (LR 0.001, 100 bins, support ±100) was weak with the smallest σ best. Every
other hyper follows the ``experiments/classic_control`` DQN recipe. LR covers the
middle of ``experiments/tuning``'s grid. 10 seeds per grid point.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment, SlurmResources

from agents.dqn_hl import DQNHistogramConfig
from environments.classic_control import CartpoleConfig, MountainCarConfig
from main import ExperimentConfig

SIGMA_RATIO_SWEEP = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
NUM_BINS_SWEEP = [50, 100, 200]
LR_SWEEP = [4.0 ** -i for i in (3, 4, 5, 6, 7)]
# Symmetric support half-widths, one component each: a sweep crosses every
# value, so sweeping the two bounds separately would add lopsided ranges.
SUPPORT_RANGES = [100, 200]
_SEEDS = list(range(10))


def _dqn_histogram_hypers(support: int) -> DQNHistogramConfig:
    # classic_control's DQN hypers, minus NETWORK_PRESET="mlp": the histogram
    # agent needs its own "mlp_histogram" default. LR is swept.
    return DQNHistogramConfig(
        TOTAL_TIMESTEPS=100_000,
        BUFFER_SIZE=10_000,
        BATCH_SIZE=64,
        LEARNING_STARTS=1_000,
        TRAIN_FREQUENCY=1,
        TARGET_NETWORK_FREQUENCY=128,
        GAMMA=0.99,
        EPSILON_START=0.1,
        EPSILON_END=0.1,
        SUPPORT_LOWER_BOUND=-support,
        SUPPORT_UPPER_BOUND=support,
    )


def _components(env: str, env_hypers) -> list[Component]:
    """One component per support range for ``env``, each sweeping the grid.

    Args:
        env: The registry ``ENV`` name, e.g. ``"cartpole"``.
        env_hypers: The environment's own hypers (e.g. ``CartpoleConfig``).

    Returns:
        Components named ``dqn_histogram_<env>_support<range>``.
    """
    return [
        Component(
            name=f"dqn_histogram_{env}_support{support}",
            config=ExperimentConfig(
                AGENT="dqn_histogram",
                ENV=env,
                AGENT_HYPERS=_dqn_histogram_hypers(support),
                ENV_HYPERS=env_hypers,
            ),
            sweep={
                "AGENT_HYPERS.SIGMA_RATIO": SIGMA_RATIO_SWEEP,
                "AGENT_HYPERS.NUM_BINS": NUM_BINS_SWEEP,
                "AGENT_HYPERS.LR": LR_SWEEP,
            },
            seeds=_SEEDS,
            # A shard of one LR's seeds. SIGMA_RATIO and NUM_BINS fix static
            # network fields, so each pair is batched apart anyway; LR is traced,
            # and leaving this unset would batch all five LRs into one shard.
            shard_size=len(_SEEDS),
        )
        for support in SUPPORT_RANGES
    ]


EXPERIMENT = Experiment(
    name="histogram_grid_classic_control",
    results_dir=Path(__file__).resolve().parent / "results",
    slurm=SlurmResources(time="02:59:00"),
    components=[
        *_components("cartpole", CartpoleConfig(EPISODE_CUTOFF=500)),
        *_components("mountaincar", MountainCarConfig(EPISODE_CUTOFF=1_000)),
    ],
)
