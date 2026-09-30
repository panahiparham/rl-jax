"""
Define: DDQN with three uniform buffers vs endpoint replay on Pinball(Easy).

A copy of ``pinball_endpoint`` with half the small memory: the small DDQN
buffers hold 500 transitions and endpoint replay keeps 100 recent steps plus
400 long-term transitions. ``pinball_endpoint`` reconstructs coresets'
``experiments/sarsa-test/pinball_1000``. Differences from the original: the
environment is pinball-jax, the network is initialised differently (the
original head was orthogonal(sqrt 2) with zero bias), and the endpoint agent
chunks per episode as described in ``design/composable_endpoint-buffer.md``.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment

from agents.ddqn import DDQNConfig
from agents.endpoint import EndpointConfig
from environments.pinball import PinballConfig
from main import ExperimentConfig

_PINBALL_LEARNER = {
    "TOTAL_TIMESTEPS": 100_000,
    "LR": 0.002,
    "ADAM_EPS": 1e-8,
    "BATCH_SIZE": 32,
    "N_STEP": 1,
    "LEARNING_STARTS": 1_000,
    "TRAIN_FREQUENCY": 1,
    "TARGET_NETWORK_FREQUENCY": 100,
    "EPSILON_START": 0.1,
    "EPSILON_END": 0.1,
    "NETWORK_PRESET": "mlp",
    "HIDDEN_SIZE": 32,
    "GAMMA": 0.99,
    "REWARD_CLIP": False,
}
_ENV_HYPERS = PinballConfig(SETTING="easy", EPISODE_CUTOFF=1_000)
_SEEDS = list(range(100))


def _ddqn(name: str, **hypers: int) -> Component:
    return Component(
        name=name,
        config=ExperimentConfig(
            AGENT="ddqn",
            ENV="pinball",
            AGENT_HYPERS=DDQNConfig(**{**_PINBALL_LEARNER, **hypers}),
            ENV_HYPERS=_ENV_HYPERS,
        ),
        seeds=_SEEDS,
        shard_size=10,
    )


EXPERIMENT = Experiment(
    name="pinball_endpoint_2",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        _ddqn("ddqn_large", BUFFER_SIZE=10_000),
        _ddqn("ddqn_small", BUFFER_SIZE=500),
        _ddqn("ddqn_small_nstep", BUFFER_SIZE=500, N_STEP=10),
        # 100 recency steps, then 400 chained 10-step transitions; 4 of every
        # 32 rows come from the long-term buffer.
        Component(
            name="endpoint",
            config=ExperimentConfig(
                AGENT="endpoint",
                ENV="pinball",
                AGENT_HYPERS=EndpointConfig(
                    **_PINBALL_LEARNER,
                    BUFFER_SIZE=100,
                    LONG_TERM_SIZE=400,
                    LONG_TERM_N_STEP=10,
                    LONG_TERM_BATCH_SIZE=4,
                    EXPECTILE_TAU=0.7,
                ),
                ENV_HYPERS=_ENV_HYPERS,
            ),
            seeds=_SEEDS,
            shard_size=10,
        ),
    ],
)
