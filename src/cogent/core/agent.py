from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar

from cogent.structured.cast import make_cast_step
from cogent.structured.schema import OutputSchema

from .env import Env
from .result import Control, Result

S = TypeVar("S")
V = TypeVar("V")
R = TypeVar("R")
T = TypeVar("T")


Step = Callable[[S, V, Env], Awaitable[Result[S, R]]]


@dataclass(frozen=True)
class Agent(Generic[S, V]):
    """Agent monad - a wrapper around Kernel that includes Env interaction."""

    _run: Callable[[Env], Awaitable[Result[S, V]]]

    async def run(self, env: Env, on_stream_chunk: Callable[[str], None] | None = None) -> Result[S, V]:
        """Run the agent and return a Result.
        
        Args:
            env: Environment with model and tools
            on_stream_chunk: Optional callback for streaming LLM output
        
        Returns:
            Result of agent execution
        """
        class SimpleRuntimeContext:
            def __init__(self, callback: Callable[[str], None]):
                self.callback = callback
            
            async def emit(self, chunk: str) -> None:
                self.callback(chunk)
            
            async def close(self) -> None:
                pass
        
        ctx = None
        if on_stream_chunk:
            ctx = SimpleRuntimeContext(on_stream_chunk)
            env_with_ctx = replace(env, runtime_context=ctx)
        
        try:
            if ctx:
                return await self._run(env_with_ctx)
            else:
                return await self._run(env)
        finally:
            if ctx:
                await ctx.close()

    def _create(self, run_func: Callable[[Env], Awaitable[Result[S, R]]]) -> Agent[S, R]:
        """Create a new agent instance."""
        return Agent(_run=run_func)

    def then(self, func: Step[S, V, R]) -> Agent[S, R]:
        """
        Chain a step function to the agent.
        
        Args:
            func: Async function that takes state, value, and env, returns Result
            
        Returns:
            New agent with the chained step
            
        Retry behavior:
        - retry_clean: Uses initial state and memory from step start (rollback)
        - retry_dirty: Uses latest state and memory from previous attempt (adapt)
        - Default max retry attempts: 3
        """
        async def new_run(env: Env) -> Result[S, R]:
            current_flow = await self.run(env)
            if current_flow.control.kind == "error":
                return Result(
                    current_flow.state,
                    control=current_flow.control,
                )
            if current_flow.control.kind != "continue":
                return Result(
                    current_flow.state,
                    control=current_flow.control,
                )
            try:
                value = current_flow._require_value()
                # Save initial state for retry_clean
                initial_state = current_flow.state
                step_result = await func(current_flow.state, value, env)
                
                max_retries = 3
                retry_count = 0
                
                while step_result.control.kind in ("retry_clean", "retry_dirty") and retry_count < max_retries:
                    retry_count += 1
                    if step_result.control.kind == "retry_clean":
                        # Use initial state for retry_clean
                        step_result = await func(initial_state, value, env)
                    else:  # retry_dirty
                        # Use latest state for retry_dirty
                        step_result = await func(step_result.state, value, env)
                
                return step_result
            except Exception as exc:
                return Result(
                    current_flow.state,
                    control=Control.Error(exc),
                )

        return self._create(new_run)

    def map(self, func: Callable[[V], R]) -> Agent[S, R]:
        async def new_run(env: Env) -> Result[S, R]:
            current_flow = await self.run(env)
            if current_flow.control.kind == "error":
                return Result(
                    current_flow.state,
                    control=current_flow.control,
                )
            if current_flow.control.kind != "continue":
                return Result(
                    current_flow.state,
                    control=current_flow.control,
                )
            try:
                value = current_flow._require_value()
                return Result(
                    current_flow.state,
                    control=Control.Continue(func(value)),
                )
            except Exception as exc:
                return Result(
                    current_flow.state,
                    control=Control.Error(exc),
                )

        return self._create(new_run)



    def recover(self, recovery_func: Callable[[Any], V]) -> Agent[S, V]:
        """Recover from error with recovery function."""
        async def new_run(env: Env) -> Result[S, V]:
            current_flow = await self.run(env)
            if current_flow.control.kind != "error":
                return Result(
                    current_flow.state,
                    control=current_flow.control,
                )
            try:
                recovered_value = recovery_func(current_flow.control.reason)
                return Result(
                    current_flow.state,
                    control=Control.Continue(recovered_value),
                )
            except Exception as exc:
                return Result(
                    current_flow.state,
                    control=Control.Error(exc),
                )

        return self._create(new_run)

    def cast(self, schema: OutputSchema[T]) -> Agent[S, T]:
        """Cast the agent's value to a new type using a schema.

        This method validates and transforms the agent's output value using
        the provided schema. On validation failure, the step will be
        retried with adaptation (retry_dirty) up to 3 times.

        Args:
            schema: The output schema to validate against

        Returns:
            New agent with the value type transformed to T

        Example:
            >>> from cogent.structured import CallableSchema
            >>> def parse_user(data):
            ...     return UserProfile(**data)
            >>> agent = agent.cast(CallableSchema(parse_user))
        """
        return self.then(make_cast_step(schema))

    @staticmethod
    def start(state: S, initial_value: V | None = None) -> Agent[S, V]:
        async def run_func(_: Env) -> Result[S, V]:
            value = initial_value if initial_value is not None else state
            return Result(state, control=Control.Continue(value))

        return Agent(_run=run_func)



