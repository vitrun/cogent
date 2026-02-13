"""Runtime trace infrastructure - separate from domain state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Evidence:
    """Evidence represents execution events captured at runtime.

    This is runtime infrastructure, not business state.
    Tree reconstruction happens only during visualization.
    """
    id: int
    parent_id: int | None = None
    action: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    info: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None


@dataclass
class TraceContext:
    """Runtime-owned trace context for capturing execution events.

    Performance guarantees:
    - Trace disabled → single None check overhead
    - Evidence append is O(1)
    - No recursive tree construction during execution
    - No UUID allocation
    """
    enabled: bool = True
    _events: list[Evidence] = field(default_factory=list)
    _next_id: int = 0

    def record(
        self,
        action: str,
        info: dict[str, Any] | None = None,
        parent_id: int | None = None,
        duration_ms: float | None = None,
    ) -> int:
        """Record an evidence event.

        Args:
            action: What happened (e.g., "step_begin", "parallel_branch")
            info: Additional context
            parent_id: Parent event ID for tree relationships
            duration_ms: Execution duration

        Returns:
            Event ID for linking child events, or -1 if tracing disabled
        """
        if not self.enabled:
            return -1

        event_id = self._next_id
        self._next_id += 1

        self._events.append(
            Evidence(
                id=event_id,
                parent_id=parent_id,
                action=action,
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
