from __future__ import annotations

import time

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Self
from pydantic import BaseModel, Field
from collections.abc import Callable
from typing import Any


@dataclass
class Evidence:
    """Evidence represents the "evidence" of execution: what happened, when,
    with what context. It's designed to be naturally captured as part of
    monadic state progression rather than as a separate tracking system.
    """

    # Core identification
    action: str  # What was done ("tool_call", "think", "write_file", etc.)

    # Optional fields
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    info: dict[str, Any] = field(default_factory=dict)

    # Relationships - natural tree structure
    step_id: str = field(default_factory=lambda: str(time.time())[-8:])
    parent_id: str | None = None

    # Timing
    duration_ms: float | None = None  # Set when complete

    # Child evidence
    children: list[Evidence] = field(default_factory=list)

    def child(self, action: str, **kwargs) -> Evidence:
        """Create child evidence - allows building evidence tree naturally."""
        child = Evidence(action=action, parent_id=self.step_id, **kwargs)
        self.children.append(child)
        return child

    def find_all(self, **filters) -> Iterator[Evidence]:
        """Find all evidence matching filters."""
        # Simple filters
        matches = True
        for field, expected in filters.items():
            if not hasattr(self, field):
                matches = False
                break
            actual = getattr(self, field)
            if callable(expected):  # Support predicate functions
                if not expected(actual):
                    matches = False
                    break
            elif actual != expected:
                matches = False
                break

        if matches:
            yield self

        # Recurse
        for child in self.children:
            yield from child.find_all(**filters)

    # Query helpers for traceability
    def as_tree(self, indent: int = 0) -> str:
        """Simple tree visualization for debugging."""
        prefix = "  " * indent
        tree = f"{prefix}├── {self.action}"
        tree += f" ({self.duration_ms or '?'}ms)"
        tree += "\n"

        for child in self.children:
            tree += child.as_tree(indent + 1)
        return tree

    def __str__(self) -> str:
        """Human readable representation."""
        duration = f" took {self.duration_ms}ms" if self.duration_ms else ""
        return f"{self.action}{duration}"


class AgentState(BaseModel):
    """State that combines task information, history, and evidence for agents."""

    task: str = Field(default="")
    history: list[str] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=lambda: Evidence(action="start"))  # type: ignore

    def with_history(self, entry: str) -> Self:
        return self.model_copy(update={"history": [*self.history, entry]})

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

        # Return state with the new evidence
        current_context = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["evidence"] and not k.startswith("_")
        }
        # Since we can't directly modify evidence, create new state with the new evidence
        new_state = self.model_copy(update={**current_context})
        new_state.evidence = new_evidence
        return new_state


QueryFn = Callable[[Evidence], bool]


def query_tools() -> QueryFn:
    """Return a query function that finds tool calls."""
    return lambda e: e.action.endswith("tool")


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]


class ToolResult(BaseModel):
    id: str
    content: str
    failed: bool = False

    @classmethod
    def success(cls, id: str, content: str) -> Self:
        return cls(id=id, content=content, failed=False)

    @classmethod
    def failure(cls, id: str, content: str) -> Self:
        return cls(id=id, content=content, failed=True)


ToolHandler = Callable[[AgentState, ToolCall], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        self._tools[name] = handler

    def run(self, state: AgentState, call: ToolCall) -> ToolResult:
        handler = self._tools.get(call.name)
        if handler is None:
            return ToolResult.failure(call.id, f"Tool not found: {call.name}")
        try:
            content = handler(state, call)
            return ToolResult.success(call.id, content)
        except Exception as exc:
            return ToolResult.failure(call.id, f"Tool error: {exc}")


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def example_tool(state: AgentState, call: ToolCall) -> str:
        query = call.args.get("query", state.task)
        return f"No results for query: {query}"

    registry.register("example", example_tool)
    return registry
