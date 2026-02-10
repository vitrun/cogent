from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar, Generic, Literal

S = TypeVar("S")
V = TypeVar("V")
T = TypeVar("T")


@dataclass(frozen=True)
class Control(Generic[T]):
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
    value: T | None = None
    reason: Any | None = None

    @staticmethod
    def Continue(value: Any | None = None) -> "Control[Any]":
        return Control(kind="continue", value=value)

    @staticmethod
    def Halt(value: Any) -> "Control[Any]":
        return Control(kind="halt", value=value)

    @staticmethod
    def RetryClean(reason: Any = None) -> "Control[None]":
        return Control(kind="retry_clean", reason=reason)

    @staticmethod
    def RetryDirty(reason: Any = None) -> "Control[None]":
        return Control(kind="retry_dirty", reason=reason)

    @staticmethod
    def Error(reason: Any) -> "Control[None]":
        return Control(kind="error", reason=reason)


@dataclass(frozen=True)
class Result(Generic[S, V]):
    """
    A container for state evolution with traceability. It captures execution evidence as state evolves.
    """

    state: S
    control: Control[V] = Control.Continue()

    def _require_value(self) -> V:
        if self.control.value is None:
            raise ValueError("StepResult has no value.")
        return self.control.value
