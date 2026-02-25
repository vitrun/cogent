"""Runtime trace infrastructure - separate from domain state.

This module provides trace/evidence capture for execution profiling and debugging.
Trace is runtime infrastructure - it does not participate in state transformation.
Tree relationships are reconstructed only during visualization via as_tree().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Evidence:
    """Evidence represents execution events captured at runtime.

    This is runtime infrastructure, not business state.
    Tree reconstruction happens only during visualization via as_tree().
    Supports hierarchical parent-child relationships via _children.
    """

    action: str = ""
    id: int = field(default=0)
    parent_id: int | None = field(default=None)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    info: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = field(default=None)
    _children: tuple[Evidence, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Handle legacy argument order - first positional arg can be action or id."""
        # If action looks like an int (e.g., Evidence(0)), swap with id
        if isinstance(self.action, int):
            # Swap action and id
            object.__setattr__(self, "id", self.action)
            object.__setattr__(self, "action", "")

    @property
    def step_id(self) -> int:
        """Alias for id for backwards compatibility."""
        return self.id

    @property
    def children(self) -> tuple[Evidence, ...]:
        """Get child evidence entries."""
        return self._children

    def child(self, action: str, info: dict[str, Any] | None = None) -> Evidence:
        """Create a child evidence entry (immutable - returns new Evidence).

        Creates a new Evidence with updated children chain.
        Tree relationships are also reconstructed via as_tree() on the Trace.
        """
        child_evidence = Evidence(
            action=action,
            id=self.id + 1,
            parent_id=self.id,
            info=info or {},
        )
        # Return new Evidence with updated children
        return Evidence(
            action=self.action,
            id=self.id,
            parent_id=self.parent_id,
            timestamp=self.timestamp,
            info=self.info,
            duration_ms=self.duration_ms,
            _children=self._children + (child_evidence,),
        )

    def find_all(self, **kwargs: Any) -> list[Evidence]:
        """Find all evidence entries matching the given criteria.

        Args:
            **kwargs: Criteria to match (e.g., action="tool.search")

        Returns:
            List of matching evidence entries including self and all descendants
        """
        results: list[Evidence] = [self]
        for child in self._children:
            results.extend(child.find_all(**kwargs))
        if kwargs:
            results = [
                e
                for e in results
                if all(e.info.get(k) == v or getattr(e, k, None) == v for k, v in kwargs.items())
            ]
        return results


class Trace:
    """Runtime trace context for capturing execution events.

    Uses stack-based nesting via push/pop for hierarchical parent-child relationships.
    Thread-safe for single-threaded execution (async).

    Performance guarantees:
    - Trace disabled → single None check overhead
    - Evidence append is O(1)
    - No recursive tree construction during execution
    - No UUID allocation
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._events: list[Evidence] = []
        self._next_id: int = 0
        self._stack: list[int] = []

    def push(self, event_id: int) -> None:
        """Push an event onto the stack for nested tracing.

        Args:
            event_id: The event ID to push as current parent
        """
        self._stack.append(event_id)

    def pop(self) -> int | None:
        """Pop the current stack frame.

        Returns:
            The parent event ID that was on top of stack, or None if stack is empty
        """
        if self._stack:
            return self._stack.pop()
        return None

    def record(
        self,
        action: str,
        info: dict[str, Any] | None = None,
        parent_id: int | None = None,
        duration_ms: float | None = None,
    ) -> int | None:
        """Record an evidence event.

        Args:
            action: What happened (e.g., "step_begin", "parallel_branch")
            info: Additional context
            parent_id: Explicit parent event ID for tree relationships
            duration_ms: Execution duration

        Returns:
            Event ID for linking child events, or None if tracing disabled
        """
        if not self.enabled:
            return None

        # Use stack top as parent if no explicit parent_id provided
        if parent_id is not None:
            effective_parent = parent_id
        elif self._stack:
            effective_parent = self._stack[-1]
        else:
            effective_parent = None

        event_id = self._next_id
        self._next_id += 1

        self._events.append(
            Evidence(
                action=action,
                id=event_id,
                parent_id=effective_parent,
                timestamp=datetime.now(UTC),
                info=info or {},
                duration_ms=duration_ms,
            )
        )

        return event_id

    def get_events(self) -> list[Evidence]:
        """Get all recorded events (for visualization)."""
        return list(self._events)

    def as_tree(self) -> dict[int | None, list[int]]:
        """Reconstruct parent-child relationships for visualization.

        Returns:
            Dict mapping parent_id to list of child_ids
        """
        tree: dict[int | None, list[int]] = {}
        for ev in self._events:
            parent = ev.parent_id
            if parent not in tree:
                tree[parent] = []
            tree[parent].append(ev.id)
        return tree

    def __len__(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        """Clear all events (for reuse)."""
        self._events.clear()
        self._next_id = 0
        self._stack.clear()
