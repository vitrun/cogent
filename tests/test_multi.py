import asyncio

from cogent.combinators import AgentRegistry, MultiEnv, MultiState
from cogent.combinators.ops import concurrent, emit, handoff, route
from cogent.kernel import Agent, Control, Result
from cogent.ports.env import ModelPort


class MockModel(ModelPort):
    async def complete(self, prompt: str) -> str:
        return "mock"

    async def stream_complete(self, prompt: str, ctx: object) -> str:
        return "mock"


def make_env(state: MultiState, registry: AgentRegistry) -> MultiEnv:
    return MultiEnv(model=MockModel(), registry=registry, state=state)


def test_handoff_changes_current() -> None:
    async def run():
        # Create simple agents that set current in their state
        def make_agent(name: str) -> Agent[MultiState, str]:
            async def _run(env: MultiEnv) -> Result[MultiState, str]:
                new_state = MultiState(
                    current=name,
                    shared=env.state.shared,
                    locals=env.state.locals,
                )
                return Result(state=new_state, value=name, control=Control.Continue())
            return Agent(_run)  # type: ignore

        registry = AgentRegistry({"a": make_agent("a"), "b": make_agent("b")})
        state = MultiState(current="", shared=(), locals={})
        env = make_env(state, registry)

        result = await handoff("a").run(env)
        assert result.state.current == "a"

        result = await handoff("b").run(env.with_state(result.state))
        assert result.state.current == "b"

    asyncio.run(run())


def test_emit_adds_to_shared() -> None:
    async def run():
        registry = AgentRegistry({})
        state = MultiState(current="", shared=("msg1",), locals={})
        env = make_env(state, registry)

        result = await emit("msg2").run(env)

        assert "msg1" in result.state.shared
        assert "msg2" in result.state.shared

    asyncio.run(run())


def test_route_selects_target() -> None:
    async def run():
        def make_agent(name: str) -> Agent[MultiState, str]:
            async def _run(env: MultiEnv) -> Result[MultiState, str]:
                new_state = MultiState(
                    current=name,
                    shared=env.state.shared,
                    locals=env.state.locals,
                )
                return Result(state=new_state, value=name, control=Control.Continue())
            return Agent(_run)  # type: ignore

        registry = AgentRegistry({"foo": make_agent("foo"), "bar": make_agent("bar")})
        state = MultiState(current="", shared=(), locals={})
        env = make_env(state, registry)

        def selector(_: MultiState) -> str:
            return "foo"

        result = await route(selector).run(env)
        assert result.state.current == "foo"

    asyncio.run(run())


def test_concurrent_merges_states() -> None:
    async def run():
        def make_agent(name: str) -> Agent[MultiState, str]:
            async def _run(env: MultiEnv) -> Result[MultiState, str]:
                new_state = MultiState(
                    current=name,
                    shared=env.state.shared + (f"from-{name}",),
                    locals=env.state.locals,
                )
                return Result(state=new_state, value=name, control=Control.Continue())
            return Agent(_run)  # type: ignore

        def merge(states: list[MultiState]) -> MultiState:
            all_shared = sum([list(s.shared) for s in states], [])
            return MultiState(current="merged", shared=tuple(all_shared), locals={})

        registry = AgentRegistry({"a": make_agent("a"), "b": make_agent("b")})
        state = MultiState(current="", shared=(), locals={})
        env = make_env(state, registry)

        result = await concurrent(
            [handoff("a"), handoff("b")],
            merge_state=merge,
        ).run(env)

        assert "from-a" in result.state.shared
        assert "from-b" in result.state.shared

    asyncio.run(run())
