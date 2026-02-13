"""ReAct state for agents module."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Self

from cogent.runtime.memory.in_memory import Context, InMemoryContext
from cogent.runtime.trace.evidence import Evidence


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
