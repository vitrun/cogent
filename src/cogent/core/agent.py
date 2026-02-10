from __future__ import annotations

import asyncio
from typing import TypeVar, Generic, Callable, Awaitable, Sequence
from dataclasses import dataclass

from .result import Control, Result
from .env import Env


S = TypeVar("S")
V = TypeVar("V")
R = TypeVar("R")


Step = Callable[[S, V, Env], Awaitable[Result[S, R]]]


@dataclass(frozen=True)
class Agent(Generic[S, V]):
    """Agent monad - a wrapper around Kernel that includes Env interaction."""

    _run: Callable[[Env], Awaitable[Result[S, V]]]

    async def run(self, env: Env) -> Result[S, V]:
        """Run the agent and return a Result."""
        return await self._run(env)

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

    @staticmethod
    def start(state: S, initial_value: V | None = None) -> Agent[S, V]:
        async def run_func(_: Env) -> Result[S, V]:
            value = initial_value if initial_value is not None else state
            return Result(state, control=Control.Continue(value))

        return Agent(_run=run_func)



