"""Core kernel abstractions - pure and dependency-free."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

S = TypeVar("S")
V = TypeVar("V")


@dataclass(frozen=True)
class Control:
    """
    Control flow directives for agent execution.

    Kinds:
    - continue: Proceed to the next step with the provided value
    - halt: Stop execution and return the provided value
    - retry_clean: Retry the current step with initial state and memory (rollback)
      - Use case: External dependency failures, temporary errors (analogous to HTTP request retries)
    - retry_dirty: Retry the current step with latest state and memory (adapt)
      - Use case: Logical errors or reasoning corrections (analogous to human learning adjustments)
    - error: Stop execution with an error
    """

    kind: Literal["continue", "halt", "retry_clean", "retry_dirty", "error"]
    reason: Any | None = None

    @staticmethod
    def Continue() -> Control:
        return Control(kind="continue")

    @staticmethod
    def Halt() -> Control:
        return Control(kind="halt")

    @staticmethod
    def RetryClean(reason: Any = None) -> Control:
        return Control(kind="retry_clean", reason=reason)

    @staticmethod
    def RetryDirty(reason: Any = None) -> Control:
        return Control(kind="retry_dirty", reason=reason)

    @staticmethod
    def Error(reason: Any) -> Control:
        return Control(kind="error", reason=reason)


@dataclass(frozen=True)
class Result(Generic[S, V]):
    """
    A container for state evolution with traceability.

    Attributes:
        state: The next domain state
        value: Output of this step
        control: Local execution directive
    """

    state: S
    value: V | None = None
    control: Control = Control.Continue()

    def _require_value(self) -> V:
        if self.value is None:
            raise ValueError("Result has no value.")
        return self.value
