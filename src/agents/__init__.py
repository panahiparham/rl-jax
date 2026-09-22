from typing import Any, NamedTuple

from agents.agent0 import Agent0Agent, Agent0Config
from agents.ddqn import DDQNAgent, DDQNConfig
from agents.dqn import DQNAgent, DQNConfig
from agents.random import RandomAgent, RandomConfig
from agents.random_buffered import RandomBufferAgent, RandomBufferConfig


class AgentSpec(NamedTuple):
    config_cls: type  # the agent's config dataclass
    agent_cls: type  # the agent class (init/act/update), constructed from a config

    def build(self, config: Any):
        """Instantiate the agent from one of its configs."""
        return self.agent_cls(config)


AGENTS: dict[str, AgentSpec] = {
    "random": AgentSpec(RandomConfig, RandomAgent),
    "random_buffered": AgentSpec(RandomBufferConfig, RandomBufferAgent),
    "dqn": AgentSpec(DQNConfig, DQNAgent),
    "ddqn": AgentSpec(DDQNConfig, DDQNAgent),
    "agent0": AgentSpec(Agent0Config, Agent0Agent),
}


def get_config(name: str):
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; registered: {sorted(AGENTS)}")
    return AGENTS[name].config_cls()


__all__ = [
    "AGENTS",
    "Agent0Config",
    "AgentSpec",
    "DDQNConfig",
    "DQNConfig",
    "RandomBufferConfig",
    "RandomConfig",
    "get_config",
]
