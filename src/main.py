from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, Union

import jax
import jax.numpy as jnp
import numpy as np
from experiment.hypers import merge_traced, split_traced

from agents import AGENTS
from agents import get_config as get_agent_config
from environments import ENVIRONMENTS
from environments import get_config as get_env_config

_DEFAULT_AGENT = "random_buffered"
_DEFAULT_ENV = "pinball"

# Union of every registered agent/env config type, naming what a config's
# hypers may be. Stays in sync with the registries automatically.
_AGENT_TYPES = tuple(spec.config_cls for spec in AGENTS.values())
_ENV_TYPES = tuple(spec.config_cls for spec in ENVIRONMENTS.values())
AgentHypers = (
    Union[_AGENT_TYPES]  # noqa: UP007
    if len(_AGENT_TYPES) > 1
    else _AGENT_TYPES[0]
)
EnvHypers = (
    Union[_ENV_TYPES]  # noqa: UP007
    if len(_ENV_TYPES) > 1
    else _ENV_TYPES[0]
)


@dataclass(frozen=True)
class ExperimentConfig:
    AGENT: str = _DEFAULT_AGENT
    ENV: str = _DEFAULT_ENV
    AGENT_HYPERS: AgentHypers = field(
        default_factory=lambda: get_agent_config(_DEFAULT_AGENT)
    )
    ENV_HYPERS: EnvHypers = field(default_factory=lambda: get_env_config(_DEFAULT_ENV))


class EnvProtocol(Protocol):
    def observation_space(self):
        """Return the observation space, whose ``shape`` and ``dtype`` are static."""
        ...

    def action_space(self):
        """Return the action space."""
        ...

    def init(self, key: jax.Array) -> tuple[Any, jax.Array]:
        """Start the environment, returning ``(state, obs)``."""
        ...

    def step(
        self,
        state: Any,
        key: jax.Array,
        action: jax.Array,
    ) -> tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Step the environment.

        Returns ``(state, reward, terminated, truncated, discount, next_obs)``.
        ``terminated`` marks a terminal state and ``truncated`` a time limit;
        either ends the episode, and the environment autoresets to the next
        episode's start. ``discount`` scales the bootstrap from ``next_obs``: it
        is zero on termination and may be zero on a non-terminal step.
        """
        ...


class AgtProtocol(Protocol):
    def init(self, key: jax.Array, observation_space: Any, action_space: Any) -> Any:
        """Start the agent, returning its initial state."""
        ...

    def act(self, state: Any, key: jax.Array, obs: jax.Array) -> jax.Array:
        """The agent's policy."""
        ...

    def update(
        self,
        state: Any,
        key: jax.Array,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        termination: jax.Array,
        truncation: jax.Array,
        discount: jax.Array,
    ) -> Any:
        """Step the agent on the transition ``obs`` -> ``action`` -> ``reward``.

        ``obs`` is the observation the action was chosen from, not its
        successor.
        """
        ...


def interaction(
    key: jax.Array,
    agent: AgtProtocol,
    environment: EnvProtocol,
    num_steps: int,
):
    agt_key, env_key, scan_key = jax.random.split(key, 3)

    agt_state = agent.init(
        agt_key,
        environment.observation_space(),
        environment.action_space(),
    )

    env_state, obs = environment.init(env_key)

    def _step(carry, _):
        key, agt_state, env_state, obs = carry
        act_key, env_key, update_key, key = jax.random.split(key, 4)

        action = agent.act(agt_state, act_key, obs)
        env_state, reward, termination, truncation, discount, next_obs = (
            environment.step(env_state, env_key, action)
        )
        agt_state = agent.update(
            agt_state,
            update_key,
            obs,
            action,
            reward,
            termination,
            truncation,
            discount,
        )

        carry = (key, agt_state, env_state, next_obs)
        output = {"reward": reward, "done": termination | truncation}

        return carry, output

    final_carry, outputs = jax.lax.scan(
        _step,
        (scan_key, agt_state, env_state, obs),
        None,
        length=num_steps,
    )

    return outputs, final_carry


def _build_train(
    static: ExperimentConfig,
) -> Callable[[dict[str, Any], jax.Array], dict[str, Any]]:
    env = ENVIRONMENTS[static.ENV].build(static.ENV_HYPERS)
    num_steps = static.AGENT_HYPERS.TOTAL_TIMESTEPS

    def train(dynamic: dict[str, Any], rng: jax.Array) -> dict[str, Any]:
        config = merge_traced(static, dynamic)
        agent = AGENTS[config.AGENT].build(config.AGENT_HYPERS)
        metrics, final_carry = interaction(rng, agent, env, num_steps)
        return {"metrics": metrics, "final_carry": final_carry}

    return train


def _split_shard(
    configs: Sequence[Any],
) -> tuple[ExperimentConfig, list[dict[str, Any]]]:
    statics, dynamics = zip(*(split_traced(config) for config in configs), strict=True)
    if any(static != statics[0] for static in statics):
        raise ValueError(
            "a shard's configs must agree on every static field; got "
            f"{len({id(s) for s in statics})} different ones"
        )
    return statics[0], list(dynamics)


def process_shard(
    configs: Sequence[ExperimentConfig], seeds: Sequence[int]
) -> list[dict[str, np.ndarray]]:
    if len(configs) != len(seeds):
        raise ValueError(f"shard has {len(configs)} config(s) and {len(seeds)} seed(s)")
    if not configs:
        return []

    static, dynamics = _split_shard(configs)
    train = _build_train(static)

    if not ENVIRONMENTS[static.ENV].vmappable:
        run_one = jax.jit(train)
        return [
            {
                name: np.asarray(value)
                for name, value in run_one(dynamic, jax.random.key(int(seed)))[
                    "metrics"
                ].items()
            }
            for dynamic, seed in zip(dynamics, seeds, strict=True)
        ]

    stacked = {
        path: jnp.asarray([dynamic[path] for dynamic in dynamics])
        for path in dynamics[0]
    }
    keys = jax.vmap(jax.random.key)(jnp.asarray([int(s) for s in seeds]))
    metrics = jax.jit(jax.vmap(train))(stacked, keys)["metrics"]
    return [
        {name: np.asarray(value[index]) for name, value in metrics.items()}
        for index in range(len(seeds))
    ]
