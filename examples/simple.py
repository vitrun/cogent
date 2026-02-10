from __future__ import annotations

from cogent.model import AgentState, ToolCall, ToolRegistry, default_registry
from cogent.agent import Agent, AgentResult


async def plan_action(state: AgentState, task: str) -> AgentResult[AgentState, ToolCall]:
    call = ToolCall(id="tool-1", name="search", args={"query": task})
    next_state = state.with_history(f"Plan: call {call.name} with query='{task}'.")
    return AgentResult(next_state, call, valid=True)


async def execute_tool(
    state: AgentState, call: ToolCall, registry: ToolRegistry | None = None
) -> AgentResult[AgentState, str]:
    registry = registry or default_registry()
    result = registry.run(state, call)
    next_state = state.with_history(f"Tool Result ({call.name}): {result.content}")
    if result.failed:
        return AgentResult(next_state, None, valid=False, error=result.content)
    return AgentResult(next_state, result.content, valid=True)


async def synthesize_answer(state: AgentState, tool_output: str) -> AgentResult[AgentState, str]:
    answer = (
        "Monadic Context Engineering structures agent workflows as composable steps "
        "with built-in state threading, error short-circuiting, and optional parallelism. "
        f"Evidence: {tool_output}"
    )
    next_state = state.with_history("Synthesized final answer.")
    return AgentResult(next_state, answer, valid=True)


async def format_output(state: AgentState, answer: str) -> AgentResult[AgentState, str]:
    formatted = f"Final Report:\n{answer}"
    next_state = state.with_history("Formatted response for delivery.")
    return AgentResult(next_state, formatted, valid=True)


async def run_simple_agent(task: str) -> AgentResult[AgentState, str]:
    initial_state = AgentState(task=task)
    return await (
        Agent.start(initial_state)
        .then(lambda s, _: plan_action(s, task))
        .then(lambda s, call: execute_tool(s, call))
        .then(synthesize_answer)
        .then(format_output)
    ).run()


if __name__ == "__main__":
    import asyncio
    res = asyncio.run(run_simple_agent("What is the capital of France?"))
    print(res.value)