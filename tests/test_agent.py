import asyncio

from cogent.agent import Agent, AgentResult, AgentState


def test_then_success() -> None:
    async def run_flow():
        async def step(s: AgentState, v: str) -> AgentResult[AgentState, str]:
            return AgentResult(s, v + "-next", valid=True)

        flow = Agent.start("state", "start").then(step)
        return await flow.run()

    result = asyncio.run(run_flow())
    assert result.valid is True
    assert result.value == "start-next"


def test_then_failure_short_circuit() -> None:
    async def run_flow():
        async def failing_step(s: AgentState, v: str) -> AgentResult[AgentState, str]:
            return AgentResult(s, None, valid=False, error="error")

        flow = Agent.start("state", "start").then(failing_step)
        return await flow.run()

    result = asyncio.run(run_flow())
    assert result.valid is False
    assert result.error == "error"


def test_map_apply() -> None:
    async def run_flow():
        base = Agent.start("state", 2)
        mapped = base.map(lambda v: v + 1)
        return await mapped.run()

    result = asyncio.run(run_flow())
    assert result.value == 3

    async def run_apply_flow():
        base = Agent.start("state", 2)
        func_flow = Agent.start("state", lambda v: v * 5)
        applied = base.apply(func_flow)
        return await applied.run()

    apply_result = asyncio.run(run_apply_flow())
    assert apply_result.value == 10


def test_async_gather() -> None:
    async def run_flow():
        flow_a = Agent.start("state", 2).map(lambda v: v * 2)
        flow_b = Agent.start("state", 3).map(lambda v: v * 2)
        gathered = Agent.gather([flow_a, flow_b])
        return await gathered.run()

    result = asyncio.run(run_flow())
    assert result.valid is True
    assert result.value == [4, 6]


def test_map_simplification() -> None:
    """Test that map simplifies the code for simple operations like x2"""

    async def run_flow():
        # Before: would need a full then with async function
        # After: can use simple map
        flow = Agent.start("state", 2).map(lambda v: v * 2)
        return await flow.run()

    result = asyncio.run(run_flow())
    assert result.valid is True
    assert result.value == 4
