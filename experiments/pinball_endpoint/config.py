"""
Define: DDQN buffer sizes vs endpoint replay on Pinball(Easy), at two budgets.

Reconstructs coresets' ``sarsa-test/pinball_1000`` at a medium budget (1k
memory) and a small one (500), with the same component layout as
``atari_50m_endpoint``. The unanchored and reservoir baselines reconstruct
``coreset_ddqn_onestep_squaredloss.json`` and
``coreset_ddqn_reservoir_composite.json``. Differences from the original: the
environment is pinball-jax, the network is initialised differently (the
original head was orthogonal(sqrt 2) with zero bias), and the endpoint agent
chunks per episode as described in ``design/composable_endpoint-buffer.md``.
"""

from __future__ import annotations

from pathlib import Path

from experiment.design import Component, Experiment

from agents.ddqn import DDQNConfig
from agents.endpoint import EndpointConfig
from agents.reservoir import ReservoirConfig
from agents.unanchored import UnanchoredConfig
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


# 100 recency steps, then chained 10-step transitions; 4 of every 32 rows come
# from the long-term buffer.
def _endpoint(name: str, long_term_size: int) -> Component:
    return Component(
        name=name,
        config=ExperimentConfig(
            AGENT="endpoint",
            ENV="pinball",
            AGENT_HYPERS=EndpointConfig(
                **_PINBALL_LEARNER,
                BUFFER_SIZE=100,
                LONG_TERM_SIZE=long_term_size,
                LONG_TERM_N_STEP=10,
                LONG_TERM_BATCH_SIZE=4,
                EXPECTILE_TAU=0.7,
            ),
            ENV_HYPERS=_ENV_HYPERS,
        ),
        seeds=_SEEDS,
        shard_size=10,
    )


# Endpoint replay's layout, storing 1-step transitions instead: every 10th for
# unanchored, a reservoir sample for reservoir. Both train DDQN on all rows.
def _unanchored(name: str, long_term_size: int) -> Component:
    return Component(
        name=name,
        config=ExperimentConfig(
            AGENT="unanchored",
            ENV="pinball",
            AGENT_HYPERS=UnanchoredConfig(
                **_PINBALL_LEARNER,
                BUFFER_SIZE=100,
                LONG_TERM_SIZE=long_term_size,
                LONG_TERM_BATCH_SIZE=4,
                SUBSAMPLE=10,
            ),
            ENV_HYPERS=_ENV_HYPERS,
        ),
        seeds=_SEEDS,
        shard_size=10,
    )


def _reservoir(name: str, long_term_size: int) -> Component:
    return Component(
        name=name,
        config=ExperimentConfig(
            AGENT="reservoir",
            ENV="pinball",
            AGENT_HYPERS=ReservoirConfig(
                **_PINBALL_LEARNER,
                BUFFER_SIZE=100,
                LONG_TERM_SIZE=long_term_size,
                LONG_TERM_BATCH_SIZE=4,
            ),
            ENV_HYPERS=_ENV_HYPERS,
        ),
        seeds=_SEEDS,
        shard_size=10,
    )


EXPERIMENT = Experiment(
    name="pinball_endpoint",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        _ddqn("ddqn_pinball", BUFFER_SIZE=10_000),
        _ddqn("ddqn_medium_pinball", BUFFER_SIZE=1_000),
        _ddqn("ddqn_medium_nstep_pinball", BUFFER_SIZE=1_000, N_STEP=10),
        _ddqn("ddqn_small_pinball", BUFFER_SIZE=500),
        _ddqn("ddqn_small_nstep_pinball", BUFFER_SIZE=500, N_STEP=10),
        _endpoint("endpoint_pinball", long_term_size=900),
        _endpoint("endpoint_small_pinball", long_term_size=400),
        _unanchored("unanchored_pinball", long_term_size=900),
        _unanchored("unanchored_small_pinball", long_term_size=400),
        _reservoir("reservoir_pinball", long_term_size=900),
        _reservoir("reservoir_small_pinball", long_term_size=400),
    ],
)
