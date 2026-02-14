import asyncio
from typing import cast

from fakes import FakeTools, make_fake_env

from cogent.agents import ReActConfig, ReActState
from cogent.agents.react.policy import ReActPolicy, ReActOutput, structured
from cogent.combinators import repeat


def test_react_decide_invalid_json() -> None:
    async def run_test():
        state = ReActState()
        env = make_fake_env()
        # Test structured parsing with invalid JSON
        parse_step = structured(ReActOutput)
        return await parse_step(state, "not-json", env)

    result = asyncio.run(run_test())
    assert result.control.kind == "error"
    assert result.control.reason is not None


def test_react_final_halts() -> None:
    responses = ['{"thought":"done","final":"Answer"}']
    env = make_fake_env(responses)

    initial_state = ReActState()
    policy = ReActPolicy(ReActConfig())
    round_agent = policy.build("")
    agent = repeat(round_agent, 10)
    result = asyncio.run(agent.run(initial_state, env))

    assert result.control.kind == "halt"
    assert result.value == "Answer"


def test_react_tool_then_final() -> None:
    responses = [
        '{"thought":"use tool","action":"echo","action_input":{"q":"hi"}}',
        '{"thought":"done","final":"ok"}',
    ]
    env = make_fake_env(responses)

    def echo_tool(args: dict[str, object]) -> str:
        return f"echo:{args.get('q')}"

    tools: FakeTools = cast(FakeTools, env.tools)
    tools.handlers["echo"] = echo_tool

    initial_state = ReActState()
    policy = ReActPolicy(ReActConfig())
    round_agent = policy.build("")
    agent = repeat(round_agent, 10)
    result = asyncio.run(agent.run(initial_state, env))

    assert result.control.kind == "halt"
    assert result.value == "ok"
    context_entries = result.state.context.snapshot()
    assert "Thought: use tool" in context_entries[0]
    assert any(entry.startswith("Observation: ") for entry in context_entries)
