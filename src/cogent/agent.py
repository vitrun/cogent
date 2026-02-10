"""Evidence-enabled monads for first-class traceability."""

from __future__ import annotations
import asyncio
from typing import Any, TypeVar, Generic, cast, Self
from dataclasses import dataclass
from collections.abc import Awaitable, Callable, Sequence
from .model import AgentState

S = TypeVar("S", bound=AgentState)
V = TypeVar("V")
R = TypeVar("R")
M = TypeVar("M", bound="Agent[Any, Any, Any]")


@dataclass(frozen=True)
class AgentResult(Generic[S, V]):
    """
    A container for state evolution with traceability. It captures execution evidence as state evolves.
    """

    state: S
    value: V | None
    valid: bool = True
    error: Any = None

    def _require_value(self) -> V:
        if self.value is None:
            raise ValueError("AgentResult has no value.")
        return self.value

    def with_evidence(
        self,
        action: str,
        input_data: Any | None = None,
        output_data: Any | None = None,
        **evidence_kwargs,
    ) -> AgentResult[S, V]:
        """Add evidence to current state without changing value."""
        # Create child evidence in current state's evidence tree
        # This appends the child to self.state.evidence.children
        new_evidence = self.state.evidence.child(action, info=evidence_kwargs)

        # Create new state with the new evidence
        current_context = {
            k: v
            for k, v in self.state.__dict__.items()
            if k not in ["evidence"] and not k.startswith("_")
        }
        new_state = self.state.model_copy(update={**current_context})
        new_state.evidence = new_evidence

        return AgentResult(state=new_state, value=self.value, valid=self.valid, error=self.error)

    def trace(
        self, action: str, input_data: Any | None = None, **evidence_kwargs
    ) -> AgentResult[S, V]:
        """Simple evidence recording - just add to current evidence trace."""
        return self.with_evidence(action, input_data, **evidence_kwargs)


AsyncStep = Callable[[S, V], Awaitable[AgentResult[S, R]]]


class Agent(Generic[S, V, M]):
    """Agent monad - a specialized Monad for AgentState that returns AgentResult."""

    def __init__(self, run_func: Callable[[], Awaitable[AgentResult[S, V]]]) -> None:
        self._run = run_func

    async def run(self) -> AgentResult[S, V]:
        """Run the agent and return an AgentResult."""
        return await self._run()

    def _create(self: M, run_func: Callable[[], Awaitable[AgentResult[S, R]]]) -> M:
        """Create a new monad instance. Subclasses can override this to return their own type."""
        return type(self)(run_func)

    def trace(self: M, action: str, input_data: Any | None = None, **evidence_kwargs) -> M:
        """Add tracing to the async execution."""

        async def traced_run() -> AgentResult[S, V]:
            result = await self.run()
            if result.valid:
                return result.trace(action, input_data, **evidence_kwargs)
            return result

        return self._create(traced_run)

    @classmethod
    def start(cls, state: S, initial_value: V | None = None) -> Self:
        async def run_func() -> AgentResult[S, V]:
            value = initial_value if initial_value is not None else cast(V, state)
            return AgentResult(state, value, valid=True)

        return cls(run_func)

    def then(self: M, func: AsyncStep[S, V, R]) -> M:
        async def new_run() -> AgentResult[S, R]:
            current_flow = await self.run()
            if not current_flow.valid:
                return AgentResult(current_flow.state, None, valid=False, error=current_flow.error)
            try:
                value = current_flow._require_value()
                return await func(current_flow.state, value)
            except Exception as exc:  # pragma: no cover - defensive catch
                return AgentResult(current_flow.state, None, valid=False, error=exc)

        return self._create(new_run)

    def map(self: M, func: Callable[[V], R]) -> M:
        async def new_run() -> AgentResult[S, R]:
            current_flow = await self.run()
            if not current_flow.valid:
                return AgentResult(current_flow.state, None, valid=False, error=current_flow.error)
            try:
                value = current_flow._require_value()
                return AgentResult(current_flow.state, func(value), valid=True)
            except Exception as exc:  # pragma: no cover - defensive catch
                return AgentResult(current_flow.state, None, valid=False, error=exc)

        return self._create(new_run)

    def apply(self: M, func_flow: "Agent[S, Callable[[V], R], Any]") -> M:
        async def new_run() -> AgentResult[S, R]:
            current_flow = await self.run()
            func_result = await func_flow.run()

            error = current_flow.error if not current_flow.valid else func_result.error
            if error is not None:
                return AgentResult(current_flow.state, None, valid=False, error=error)

            try:
                value = current_flow._require_value()
                func = func_result._require_value()
                return AgentResult(current_flow.state, func(value), valid=True)
            except Exception as exc:  # pragma: no cover - defensive catch
                return AgentResult(current_flow.state, None, valid=False, error=exc)

        return self._create(new_run)

    @staticmethod
    def gather(
        flows: Sequence["Agent[S, Any, Any]"],
        merge_state: Callable[[Sequence[S]], S] | None = None,
    ) -> "Agent[S, list[Any], Any]":
        async def new_run() -> AgentResult[S, list[Any]]:
            if not flows:
                return AgentResult(cast(S, None), None, valid=False, error="No flows provided")

            results = await asyncio.gather(*(flow.run() for flow in flows))
            errors = [result for result in results if not result.valid]
            if errors:
                failing = errors[0]
                return AgentResult(failing.state, None, valid=False, error=failing.error)

            states = [result.state for result in results]
            final_state = merge_state(states) if merge_state else states[-1]
            values = [result.value for result in results]
            return AgentResult(final_state, values, valid=True)

        return Agent(new_run)
