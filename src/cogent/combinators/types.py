"""Combinator types and data classes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from cogent.kernel import Agent
from cogent.kernel.env import Env


@dataclass(frozen=True)
class MultiState:
    """State for multi-agent execution.

    Attributes:
        current: Name of the currently executing agent.
        shared: Tuple of shared messages between agents.
        locals: Mapping of agent names to their local state.
    """
    current: str
    shared: tuple[Any, ...] = field(default_factory=tuple)
    locals: Mapping[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """Registry for looking up agents by name.

    Attributes:
        _agents: Internal mapping of agent names to Agent instances.
    """

    def __init__(self, agents: dict[str, Agent] | None = None) -> None:
        self._agents = agents or {}

    def get(self, name: str) -> Agent:
        """Get an agent by name."""
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' not found in registry")
        return self._agents[name]

    def __getitem__(self, name: str) -> Agent:
        return self.get(name)

    def __setitem__(self, name: str, agent: Agent) -> None:
        self._agents[name] = agent


@dataclass
class MultiEnv(Env):
    """Environment for multi-agent execution.

    Extends Env with:
    - registry: For looking up agents by name

    Note: State flows through Result, not Env. Each agent receives
    state explicitly via Agent.run(state, env).
    """
    registry: AgentRegistry = field(default_factory=AgentRegistry)


def merge_states(states: list[MultiState]) -> MultiState:
    """Default state merge - combines shared messages."""
    shared = ()
    for s in states:
        shared = shared + s.shared
    return MultiState(current="merged", shared=shared, locals={})
