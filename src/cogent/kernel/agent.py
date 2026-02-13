"""Agent monad - core orchestration primitive."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar

from cogent.kernel.result import Control, Result
from cogent.kernel.ports import Env
from cogent.kernel.types import ToolUse

S = TypeVar("S")
V = TypeVar("V")
R = TypeVar("R")
T = TypeVar("T")


Step = Callable[[S, V, Env], Awaitable[Result[S, R]]]


# Extension registry - class-level storage for Agent capabilities
_extensions_registry: dict[str, Callable] = {}


@dataclass(frozen=True)
class Agent(Generic[S, V]):
    """Agent monad - a wrapper around Kernel that includes Env interaction.

    Capabilities can be registered via register_op() for extensibility.
    """

    _run: Callable[[Env], Awaitable[Result[S, V]]]

    @classmethod
    def register_op(cls, name: str, fn: Callable[..., Any]) -> None:
        """Register an operation capability on the Agent class.

        Args:
            name: The operation name (e.g., "cast")
            fn: The function to register
        """
        _extensions_registry[name] = fn

    def __getattr__(self, name: str) -> Any:
        """Allow calling registered extension methods."""
        if name in _extensions_registry:
            fn = _extensions_registry[name]
            # Bind the function to this instance
            return lambda *args, **kwargs: fn(self, *args, **kwargs)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    async def run(self, env: Env, on_stream_chunk: Callable[[str], None] | None = None) -> Result[S, V]:
        """Run the agent and return a Result.

        Args:
            env: Environment with model and tools
            on_stream_chunk: Optional callback for streaming LLM output

        Returns:
            Result of agent execution

        Note: This method does NOT implement retry semantics.
        Retry is strictly step-level logic handled within each step function.
        """
        class SimpleRuntimeContext:
            def __init__(self, callback: Callable[[str], None]):
                self.callback = callback

            async def emit(self, chunk: str) -> None:
                self.callback(chunk)

            async def close(self) -> None:
                pass

        ctx: SimpleRuntimeContext | None = None
        env_to_run = env
        if on_stream_chunk:
            ctx = SimpleRuntimeContext(on_stream_chunk)
            env_to_run = replace(env, runtime_context=ctx)

        # Get trace context
        trace = env.trace if env else None
        step_id = -1

        try:
            # Record step_begin if tracing enabled
            if trace is not None:
                step_id = trace.record("step_begin")

            # Execute step exactly once - no retry loop at runtime level
            start_time = time.perf_counter()
            result = await self._run(env_to_run)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Record step_end with control info
            if trace is not None:
                trace.record(
                    "step_end",
                    info={"control": result.control.kind},
                    parent_id=step_id,
                    duration_ms=duration_ms,
                )

            return result
        finally:
            if ctx is not None:
                await ctx.close()

    def _create(self, run_func: Callable[[Env], Awaitable[Result[S, R]]]) -> Agent[S, R]:
        """Create a new agent instance."""
        return Agent(_run=run_func)

    def then(self, func: Step[S, V, R]) -> Agent[S, R]:
        """Chain a step function to the agent.

        Args:
            func: Async function that takes state, value, and env, returns Result

        Returns:
            New agent with the chained step

        Note: This method does NOT implement retry semantics.
        Each step is fully responsible for handling its own retry_clean / retry_dirty logic.
        Runtime only interprets Control and records trace.
        """
        async def new_run(env: Env) -> Result[S, R]:
            current_flow = await self.run(env)
            if current_flow.control.kind == "error":
                return Result(
                    state=current_flow.state,
                    value=current_flow.value,
                    control=current_flow.control,
                )  # type: ignore[return-value]
            if current_flow.control.kind != "continue":
                # Preserve value when propagating non-continue controls
                return Result(
                    state=current_flow.state,
                    value=current_flow.value,
                    control=current_flow.control,
                )  # type: ignore[return-value]
            try:
                value = current_flow._require_value()
                # Execute step exactly once - retry is step-internal
                step_result = await func(current_flow.state, value, env)
                return step_result
            except Exception as exc:
                return Result(
                    state=current_flow.state,
                    control=Control.Error(exc),
                )  # type: ignore[return-value]

        return self._create(new_run)

    def map(self, func: Callable[[V], R]) -> Agent[S, R]:
        async def new_run(env: Env) -> Result[S, R]:
            current_flow = await self.run(env)
            if current_flow.control.kind == "error":
                return Result(
                    state=current_flow.state,
                    value=current_flow.value,
                    control=current_flow.control,
                )  # type: ignore[return-value]
            if current_flow.control.kind != "continue":
                # Preserve value when propagating non-continue controls
                return Result(
                    state=current_flow.state,
                    value=current_flow.value,
                    control=current_flow.control,
                )  # type: ignore[return-value]
            try:
                value = current_flow._require_value()
                return Result(
                    state=current_flow.state,
                    value=func(value),
                    control=Control.Continue(),
                )
            except Exception as exc:
                return Result(
                    state=current_flow.state,
                    control=Control.Error(exc),
                )  # type: ignore[return-value]

        return self._create(new_run)

    def recover(self, recovery_func: Callable[[Any], V]) -> Agent[S, V]:
        """Recover from error with recovery function."""
        async def new_run(env: Env) -> Result[S, V]:
            current_flow = await self.run(env)
            if current_flow.control.kind != "error":
                return Result(
                    state=current_flow.state,
                    value=current_flow.value,
                    control=current_flow.control,
                )
            try:
                recovered_value = recovery_func(current_flow.control.reason)
                return Result(
                    state=current_flow.state,
                    value=recovered_value,
                    control=Control.Continue(),
                )
            except Exception as exc:
                return Result(
                    state=current_flow.state,
                    control=Control.Error(exc),
                )

        return self._create(new_run)

    @staticmethod
    def start(state: S, initial_value: V | None = None) -> Agent[S, V]:
        async def run_func(_: Env) -> Result[S, V]:
            if initial_value is not None:
                return Result(state, value=initial_value, control=Control.Continue())
            # When initial_value is None, use state as value (type cast for V=S case)
            return Result(state, value=state, control=Control.Continue())  # type: ignore[arg-type]

        return Agent(_run=run_func)
