"""Multi-agent primitives: handoff, emit, route, concurrent."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from cogent.core import Agent, Control, Result

from . import MultiEnv, MultiState


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
            value=result.value,
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
) -> Agent[MultiState, list[Result[MultiState, Any]]]:
    """Execute multiple agents concurrently.

    Semantics:
        - Execute all agents with the same initial env
        - Gather all results as list[Result]
        - Explicitly merge states using merge_state function
        - DO NOT merge control - returns raw branch Results
        - DO NOT merge value - returns list of branch values
        - Runtime owns trace - records parallel_begin/parallel_end

    Trace behavior:
        - Automatically records "parallel_begin"
        - Records each branch as child event
        - Records "parallel_end"

    Args:
        agents: Sequence of agents to execute concurrently.
        merge_state: Function to merge the resulting states from all agents.

    Returns:
        Agent[MultiState, list[Result[MultiState, Any]]]:
            A new agent that runs all input agents concurrently.
            Returns merged state and raw list of branch Results.
            Developers must explicitly write a step to interpret branch Results.
    """
    async def _run(env: MultiEnv) -> Result[MultiState, list[Result[MultiState, Any]]]:
        trace = env.trace if hasattr(env, 'trace') else None

        # Record parallel_begin
        parallel_id = -1
        if trace is not None:
            parallel_id = trace.record("parallel_begin")

        # Execute all agents concurrently
        tasks = [agent.run(env) for agent in agents]
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
        merged_state = merge_state(states)

        return Result(
            state=merged_state,
            value=results,
            control=Control.Continue(),
        )

    return Agent(_run)  # type: ignore[arg-type]
