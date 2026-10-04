from pathlib import Path

from experiment.design import Component, Experiment

from agents.ddqn import DDQNConfig
from agents.endpoint import EndpointConfig
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


def _endpoint(name: str, recency_size: int, long_term_size: int, subsample: int, long_term_batch: int) -> Component:
    return Component(
        name=name,
        config=ExperimentConfig(
            AGENT="endpoint",
            ENV="pinball",
            AGENT_HYPERS=EndpointConfig(
                **_PINBALL_LEARNER,
                BUFFER_SIZE=recency_size,
                LONG_TERM_SIZE=long_term_size,
                LONG_TERM_N_STEP=subsample,
                LONG_TERM_BATCH_SIZE=long_term_batch,
                EXPECTILE_TAU=0.7,
            ),
            ENV_HYPERS=_ENV_HYPERS,
        ),
        seeds=_SEEDS,
        shard_size=10,
    )


def _unanchored(name: str, recency_size: int, long_term_size: int, subsample: int, long_term_batch: int) -> Component:
    return Component(
        name=name,
        config=ExperimentConfig(
            AGENT="unanchored",
            ENV="pinball",
            AGENT_HYPERS=UnanchoredConfig(
                **_PINBALL_LEARNER,
                BUFFER_SIZE=recency_size,
                LONG_TERM_SIZE=long_term_size,
                LONG_TERM_BATCH_SIZE=long_term_batch,
                SUBSAMPLE=subsample,
            ),
            ENV_HYPERS=_ENV_HYPERS,
        ),
        seeds=_SEEDS,
        shard_size=10,
    )

def _connected(name: str, recency_size: int, long_term_size: int, subsample: int, long_term_batch: int) -> Component:
    return Component(
        name=name,
        config=ExperimentConfig(
            AGENT="unanchored",
            ENV="pinball",
            AGENT_HYPERS=UnanchoredConfig(
                **_PINBALL_LEARNER,
                BUFFER_SIZE=recency_size,
                LONG_TERM_SIZE=long_term_size,
                LONG_TERM_BATCH_SIZE=long_term_batch,
                SUBSAMPLE=1,
                LONG_TERM_N_STEP=subsample,
            ),
            ENV_HYPERS=_ENV_HYPERS,
        ),
        seeds=_SEEDS,
        shard_size=10,
    )


EXPERIMENT = Experiment(
    name="pinball_segmented_buffers",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        # segmenting
        _ddqn("ddqn_pinball", BUFFER_SIZE=10_000),
        _unanchored("segmented", recency_size=1000, long_term_size=9000, subsample=1, long_term_batch=29),
        _unanchored("backward_segmented", recency_size=9000, long_term_size=1000, subsample=1, long_term_batch=3),
        _unanchored("segmented_weighted", recency_size=1000, long_term_size=9000, subsample=1, long_term_batch=16),
        _unanchored("backward_segmented_weighted", recency_size=9000, long_term_size=1000, subsample=1, long_term_batch=16),
        _unanchored("segmented_highly_weighted", recency_size=1000, long_term_size=9000, subsample=1, long_term_batch=3),
        _unanchored("backward_segmented_highly_weighted", recency_size=9000, long_term_size=1000, subsample=1, long_term_batch=29),
        _unanchored("segmented_blocked", recency_size=1000, long_term_size=9000, subsample=1, long_term_batch=0),
        _unanchored("backward_segmented_blocked", recency_size=9000, long_term_size=1000, subsample=1, long_term_batch=32),

        # sub-sampling
        _unanchored("segmented_subsample", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=29),
        _unanchored("segmented_weighted_subsample", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=16),
        _unanchored("segmented_highly_weighted_subsample", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=3),
        _unanchored("segmented_blocked_subsample", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=0),

        # endpoint
        _endpoint("segmented_endpoint", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=29),
        _endpoint("segmented_weighted_endpoint", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=16),
        _endpoint("segmented_highly_weighted_endpoint", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=3),
        _endpoint("segmented_blocked_endpoint", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=0),

        # connected
        _connected("segmented_connected", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=29),
        _connected("segmented_weighted_connected", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=16),
        _connected("segmented_highly_weighted_connected", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=3),
        _connected("segmented_blocked_connected", recency_size=1000, long_term_size=900, subsample=10, long_term_batch=0),

        # tiny recent segment!
        # segmenting
        _unanchored("segmented_tiny", recency_size=100, long_term_size=9900, subsample=1, long_term_batch=29),
        _unanchored("backward_segmented_tiny", recency_size=9900, long_term_size=100, subsample=1, long_term_batch=3),
        _unanchored("segmented_weighted_tiny", recency_size=100, long_term_size=9900, subsample=1, long_term_batch=16),
        _unanchored("backward_segmented_weighted_tiny", recency_size=9900, long_term_size=100, subsample=1, long_term_batch=16),
        _unanchored("segmented_highly_weighted_tiny", recency_size=100, long_term_size=9900, subsample=1, long_term_batch=3),
        _unanchored("backward_segmented_highly_weighted_tiny", recency_size=9900, long_term_size=100, subsample=1, long_term_batch=29),
        _unanchored("segmented_blocked_tiny", recency_size=100, long_term_size=9900, subsample=1, long_term_batch=0),
        _unanchored("backward_segmented_blocked_tiny", recency_size=9900, long_term_size=100, subsample=1, long_term_batch=32),

        # sub-sampling
        _unanchored("segmented_subsample_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=29),
        _unanchored("segmented_weighted_subsample_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=16),
        _unanchored("segmented_highly_weighted_subsample_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=3),
        _unanchored("segmented_blocked_subsample_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=0),

        # endpoint
        _endpoint("segmented_endpoint_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=29),
        _endpoint("segmented_weighted_endpoint_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=16),
        _endpoint("segmented_highly_weighted_endpoint_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=3),
        _endpoint("segmented_blocked_endpoint_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=0),

        # connected
        _connected("segmented_connected_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=29),
        _connected("segmented_weighted_connected_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=16),
        _connected("segmented_highly_weighted_connected_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=3),
        _connected("segmented_blocked_connected_tiny", recency_size=100, long_term_size=990, subsample=10, long_term_batch=0),
    ],
)
