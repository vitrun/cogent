"""ReAct state for agents module."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Self, TypeVar

from cogent.kernel.env import Context, InMemoryContext
from cogent.kernel.trace import Evidence

S = TypeVar("S")


class ReActStateProtocol(Protocol[S]):
    """Protocol for states that can be used with ReAct."""

    context: Context
    scratchpad: str
    evidence: Evidence

    def with_context(self, entry: str) -> S:
        ...

    def with_scratchpad(self, text: str) -> S:
        ...

    def with_evidence(self, action: str, **kwargs) -> S:
        ...

    def with_formatted_anthropic_messages(self, messages: list[dict]) -> S:
        ...

    def with_formatted_messages(self, messages: list[dict]) -> S:
        ...


@dataclass(frozen=True)
class ReActState:
    """Default opinionated state that combines task information, context, and evidence for agents."""

    context: Context = field(default_factory=InMemoryContext)
    scratchpad: str = field(default="")
    evidence: Evidence = field(default_factory=lambda: Evidence(action="start"))  # type: ignore

    def with_context(self, entry: str) -> Self:
        new_context = self.context.append(entry)
        return replace(self, context=new_context)

    def with_scratchpad(self, text: str) -> Self:
        return replace(self, scratchpad=text)

    def with_evidence(
        self,
        action: str,
        input_data: Any = None,
        output_data: Any = None,
        info: dict[str, Any] | None = None,
        **kwargs,
    ) -> Self:
        new_evidence = self.evidence.child(
            action,
            info=info or {},
        )

        # Create new state with the new evidence
        return replace(self, evidence=new_evidence)
