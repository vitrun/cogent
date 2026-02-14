"""Combinators - higher-order agent composition primitives."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from cogent.kernel import Agent, Control, Result
from cogent.kernel.env import Env

S = TypeVar("S")
V = TypeVar("V")


def concurrent(
    agents: Sequence[Agent[MultiState, Any]],
    merge_state: Callable[[list[MultiState]], MultiState],
) -> Agent[MultiState, list[Result[MultiState, Any]]]:
    """Execute multiple agents concurrently.

    Returns merged state and raw list of branch Results.
    Developers must explicitly write a step to interpret branch Results.
    """
    async def _run(state: MultiState, env: MultiEnv) -> Result[MultiState, list[Result[MultiState, Any]]]:
        trace = env.trace if hasattr(env, 'trace') else None

        # Record parallel_begin
        parallel_id = -1
        if trace is not None:
            parallel_id = trace.record("parallel_begin")

        # Execute all agents concurrently
        tasks = [agent.run(state, env) for agent in agents]
        results: list[Result[MultiState, Any]] = await asyncio.gather(*tasks)

        # Record each branch as child event
        if trace is not None:
            for i, result in enumerate(results):
                trace.record(
                    f"branch_{i}",
                    info={"control": result.control.kind},
                    parent_id=parallel_id,
                )

        # Record parallel_end
        if trace is not None:
            trace.record("parallel_end", parent_id=parallel_id)

        # Merge states
        states = [r.state for r in results]
        merged_state = merge_states(states)

        return Result(
            state=merged_state,
            value=results,
            control=Control.Continue(),
        )

    return Agent(_run)  # type: ignore[arg-type]


def repeat(
    agent: Agent[S, V],
    max_steps: int,
) -> Agent[S, V]:
    """Repeat an agent up to max_steps times.

    Executes the wrapped agent repeatedly, stopping when:
    - Control.Halt is returned
    - Control.Error is returned
    - max_steps is reached

    Args:
        agent: The agent to repeat
        max_steps: Maximum number of iterations (must be > 0)

    Returns:
        New Agent that executes the inner agent up to max_steps times
    """

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    async def repeat_wrapper(state: S, env: Env) -> Result[S, V]:
        """Loop that runs the agent and checks control."""
        current_state = state
        result: Result[S, V] | None = None

        for _ in range(max_steps):
            result = await agent.run(current_state, env)

            # Stop on halt or error
            if result.control.kind == "halt":
                return result
            if result.control.kind == "error":
                return result

            # Continue with next iteration, threading state forward
            current_state = result.state

        # max_steps > 0 is validated above, so result is set
        return result  # type: ignore[return-value]

    return Agent(_run=repeat_wrapper)


def merge_states(states: list[MultiState]) -> MultiState:
    """Default state merge - combines shared messages."""
    shared = ()
    for s in states:
        shared = shared + s.shared
    return MultiState(current="merged", shared=shared, locals={})


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
