from __future__ import annotations

from cogent import Env, ReActState, ToolCall, ToolRegistry, Agent, Result, default_registry


async def plan_action(state: ReActState, task: str, env: Env) -> Result[ReActState, ToolCall]:
    _ = env
    call = ToolCall(id="tool-1", name="search", args={"query": task})
    next_state = state.with_history(f"Plan: call {call.name} with query='{task}'.")
    return Result(next_state, control=Control.Continue(call))


async def execute_tool(
    state: ReActState, call: ToolCall, env: Env, registry: ToolRegistry | None = None
) -> Result[ReActState, str]:
    _ = env
    registry = registry or default_registry()
    result = await registry.run(env, state, call)
    next_state = state.with_history(f"Tool Result ({call.name}): {result.content}")
    if result.failed:
        return Result(next_state, control=Control.Error(result.content))
    return Result(next_state, control=Control.Continue(result.content))


async def synthesize_answer(state: ReActState, tool_output: str, env: Env) -> Result[ReActState, str]:
    _ = env
    answer = (
        "Monadic Context Engineering structures agent workflows as composable steps "
        "with built-in state threading, error short-circuiting, and optional parallelism. "
        f"Evidence: {tool_output}"
    )
    next_state = state.with_history("Synthesized final answer.")
    return Result(next_state, control=Control.Continue(answer))


async def format_output(state: ReActState, answer: str, env: Env) -> Result[ReActState, str]:
    _ = env
    formatted = f"Final Report:\n{answer}"
    next_state = state.with_history("Formatted response for delivery.")
    return Result(next_state, control=Control.Continue(formatted))


async def run_simple_agent(task: str, env: Env) -> Result[ReActState, str]:
    initial_state = ReActState(task=task)
    return await (
        Agent.start(initial_state, "")
        .then(lambda s, _v, inner_env: plan_action(s, task, inner_env))
        .then(lambda s, call, inner_env: execute_tool(s, call, inner_env))
        .then(synthesize_answer)
        .then(format_output)
    ).run(env)


if __name__ == "__main__":
    import asyncio
    from tests.fakes import make_fake_env

    res = asyncio.run(run_simple_agent("What is the capital of France?", make_fake_env()))
    print(res.value)