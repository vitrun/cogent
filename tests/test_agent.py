import asyncio

from cogent import Agent, Result, ReActState, Control
from fakes import make_fake_env


def test_then_success() -> None:
    async def run_flow():
        async def step(s: ReActState, v: str, env) -> Result[ReActState, str]:
            _ = env
            return Result(s, value=v + "-next", control=Control.Continue())

        flow = Agent.start("state", "start").then(step)
        return await flow.run(make_fake_env())

    result = asyncio.run(run_flow())
    assert result.control.kind == "continue"
    assert result.value == "start-next"


def test_then_failure_short_circuit() -> None:
    async def run_flow():
        async def failing_step(s: ReActState, v: str, env) -> Result[ReActState, str]:
            _ = (v, env)
            return Result(s, control=Control.Error("error"))

        flow = Agent.start("state", "start").then(failing_step)
        return await flow.run(make_fake_env())

    result = asyncio.run(run_flow())
    assert result.control.kind == "error"
    assert result.control.reason == "error"


def test_map() -> None:
    async def run_flow():
        base = Agent.start("state", 2)
        mapped = base.map(lambda v: v + 1)
        return await mapped.run(make_fake_env())

    result = asyncio.run(run_flow())
    assert result.control.kind == "continue"
    assert result.value == 3


def test_map_simplification() -> None:
    """Test that map simplifies the code for simple operations like x2"""

    async def run_flow():
        # Before: would need a full then with async function
        # After: can use simple map
        flow = Agent.start("state", 2).map(lambda v: v * 2)
        return await flow.run(make_fake_env())

    result = asyncio.run(run_flow())
    assert result.control.kind == "continue"
    assert result.value == 4


def test_control_halt_propagates_value() -> None:
    async def run_flow():
        async def halting_step(s: ReActState, v: str, env) -> Result[ReActState, str]:
            _ = env
            return Result(s, value=v + "-halt", control=Control.Halt())

        async def unreachable_step(s: ReActState, v: str, env) -> Result[ReActState, str]:
            _ = env
            return Result(s, value=v + "-next", control=Control.Continue())

        flow = Agent.start("state", "start").then(halting_step).then(unreachable_step)
        return await flow.run(make_fake_env())

    result = asyncio.run(run_flow())
    assert result.control.kind == "halt"
    assert result.value == "start-halt"


def test_control_retry_reruns_current_step() -> None:
    """Test that runtime passes through retry control without retrying.

    In the new design, retry is strictly step-level.
    Runtime does NOT implement retry loops.
    Steps that want retry must handle it internally.
    """
    async def run_flow():
        attempt_count = 0

        async def retrying_step(s: ReActState, v: str, env) -> Result[ReActState, str]:
            _ = env
            nonlocal attempt_count
            attempt_count += 1
            # Runtime passes through retry control - step must handle retry internally
            if attempt_count < 3:
                return Result(s, control=Control.RetryClean("retry"))
            return Result(s, value=v + "-done", control=Control.Continue())

        flow = Agent.start("state", "start").then(retrying_step)
        result = await flow.run(make_fake_env())
        return result, attempt_count

    result, attempt_count = asyncio.run(run_flow())
    # Runtime does NOT retry - step ran once and returned RetryClean
    assert result.control.kind == "retry_clean"
    assert result.value is None
    assert attempt_count == 1


def test_control_propagates_across_combinators() -> None:
    async def run_map_flow():
        async def halting_step(s: ReActState, v: str, env) -> Result[ReActState, str]:
            _ = env
            return Result(s, value=v, control=Control.Halt())

        flow = Agent.start("state", "start").then(halting_step).map(lambda v: v + "-map")
        return await flow.run(make_fake_env())

    map_result = asyncio.run(run_map_flow())
    assert map_result.value == "start"
    assert map_result.control.kind == "halt"


def test_control_retry_dirty_preserves_state() -> None:
    """Test that runtime passes through retry_dirty without retrying.

    In the new design, retry is strictly step-level.
    Runtime does NOT implement retry loops.
    """
    async def run_flow():
        attempt_count = 0

        async def retrying_step(s: ReActState, v: str, env) -> Result[ReActState, str]:
            _ = env
            nonlocal attempt_count
            attempt_count += 1
            # On each retry, modify the state
            new_state = f"{s}-attempt-{attempt_count}"
            if attempt_count < 3:
                return Result(new_state, control=Control.RetryDirty("retry"))
            return Result(new_state, value=v + "-done", control=Control.Continue())

        flow = Agent.start("initial-state", "start").then(retrying_step)
        result = await flow.run(make_fake_env())
        return result, attempt_count

    result, attempt_count = asyncio.run(run_flow())
    # Runtime does NOT retry - step ran once
    assert result.control.kind == "retry_dirty"
    assert attempt_count == 1
    # State evolved once
    assert result.state == "initial-state-attempt-1"

