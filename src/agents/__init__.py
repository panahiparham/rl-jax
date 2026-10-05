from typing import Any, NamedTuple

from agents.agent0 import Agent0Agent, Agent0Config
from agents.agent1 import Agent1Agent, Agent1Config
from agents.ddqn import DDQNAgent, DDQNConfig
from agents.dqn import DQNAgent, DQNConfig
from agents.dqn_noheadbias import DQNNoHeadBiasAgent, DQNNoHeadBiasConfig
from agents.endpoint import EndpointAgent, EndpointConfig
from agents.random import RandomAgent, RandomConfig
from agents.random_buffered import RandomBufferAgent, RandomBufferConfig
from agents.reservoir import ReservoirAgent, ReservoirConfig
from agents.unanchored import UnanchoredAgent, UnanchoredConfig


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
    "dqn_noheadbias": AgentSpec(DQNNoHeadBiasConfig, DQNNoHeadBiasAgent),
    "agent0": AgentSpec(Agent0Config, Agent0Agent),
    "agent1": AgentSpec(Agent1Config, Agent1Agent),
    "endpoint": AgentSpec(EndpointConfig, EndpointAgent),
    "unanchored": AgentSpec(UnanchoredConfig, UnanchoredAgent),
    "reservoir": AgentSpec(ReservoirConfig, ReservoirAgent),
}


def get_config(name: str):
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; registered: {sorted(AGENTS)}")
    return AGENTS[name].config_cls()


__all__ = [
    "AGENTS",
    "Agent0Config",
    "Agent1Config",
    "AgentSpec",
    "DDQNConfig",
    "DQNConfig",
    "DQNNoHeadBiasConfig",
    "EndpointConfig",
    "RandomBufferConfig",
    "RandomConfig",
    "ReservoirConfig",
    "UnanchoredConfig",
    "get_config",
]
