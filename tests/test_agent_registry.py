"""Every entry in the ``AGENTS`` registry builds from its own default config."""

from __future__ import annotations

import pytest

from agents import AGENTS, get_config


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_spec_builds_its_agent(name: str):
    spec = AGENTS[name]
    agent = spec.build(get_config(name))
    assert isinstance(agent, spec.agent_cls)
    assert callable(agent.init) and callable(agent.act) and callable(agent.update)


def test_get_config_matches_the_spec():
    for name, spec in AGENTS.items():
        assert isinstance(get_config(name), spec.config_cls)


def test_unknown_agent_is_rejected():
    with pytest.raises(ValueError, match="unknown agent"):
        get_config("nonexistent")
