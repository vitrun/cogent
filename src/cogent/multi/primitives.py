"""Multi-agent primitives: handoff, emit, route, concurrent."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from cogent.core import Agent, Control, Result

from . import MultiEnv, MultiState, merge_control


def handoff(target: str) -> Agent[MultiState, Any]:
    """Switch execution to the target agent.

    Semantics:
        - Lookup target agent via registry
        - Execute the target agent with current MultiEnv
        - Use the agent's resulting state
        - Control propagates transparently

    Args:
        target: Name of the target agent to hand off to.

    Returns:
        Agent[MultiState, Any]: A new agent that performs the handoff.
    """
    async def _run(env: MultiEnv) -> Result[MultiState, Any]:
        agent = env.registry.get(target)

        # Execute the target agent with current MultiEnv
        result = await agent.run(env)

        # Use the inner agent's resulting state
        # This preserves shared messages and locals from the inner agent
        new_state = result.state

        return Result(
            state=new_state,
            control=result.control,
        )

    return Agent(_run)  # type: ignore[arg-type]


def emit(msg: Any) -> Agent[MultiState, None]:
    """Emit a message to the shared space.

    Semantics:
        - Append the message to the shared tuple
        - Do not change current agent
        - Do not change locals
        - Control is always Continue

    Args:
        msg: The message to emit to the shared space.

    Returns:
        Agent[MultiState, None]: A new agent that emits the message.
    """
    async def _run(env: MultiEnv) -> Result[MultiState, None]:
        ms = env.state
        new_state = MultiState(
            current=ms.current,
            shared=ms.shared + (msg,),
            locals=ms.locals,
        )
        return Result(state=new_state, control=Control.Continue())

    return Agent(_run)  # type: ignore[arg-type]


def route(selector: Callable[[MultiState], str]) -> Agent[MultiState, Any]:
    """Dynamically route to a target agent based on current state.

    Semantics:
        - Select target agent using the selector function
        - Delegate to handoff
        - Control propagates transparently

    Args:
        selector: Function that takes MultiState and returns target agent name.

    Returns:
        Agent[MultiState, Any]: A new agent that routes to the selected target.
    """
    async def _run(env: MultiEnv) -> Result[MultiState, Any]:
        target = selector(env.state)
        return await handoff(target).run(env)

    return Agent(_run)  # type: ignore[arg-type]


def concurrent(
    agents: Sequence[Agent[MultiState, Any]],
    merge_state: Callable[[list[MultiState]], MultiState],
) -> Agent[MultiState, list[Any]]:
    """Execute multiple agents concurrently.

    Semantics:
        - Execute all agents with the same initial env
        - Gather all results
        - Explicitly merge states using merge_state function
        - Merge controls using merge_control logic
        - Does NOT use return_exceptions=True (exceptions must be handled by agents)

    Args:
        agents: Sequence of agents to execute concurrently.
        merge_state: Function to merge the resulting states from all agents.

    Returns:
        Agent[MultiState, list[Any]]: A new agent that runs all input agents concurrently.
    """
    async def _run(env: MultiEnv) -> Result[MultiState, list[Any]]:
        # Execute all agents concurrently
        tasks = [agent.run(env) for agent in agents]
        results = await asyncio.gather(*tasks)

        # Extract states and controls
        states = [r.state for r in results]
        controls = [r.control.kind for r in results]

        # Merge controls
        merged_control = merge_control(controls)

        # Merge states
        merged_state = merge_state(states)

        return Result(
            state=merged_state,
            control=merged_control,
        )

    return Agent(_run)  # type: ignore[arg-type]
