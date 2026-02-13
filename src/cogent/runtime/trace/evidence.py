"""Domain-level evidence for execution history."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Evidence:
    """Evidence represents the "evidence" of execution: what happened, when,
    with what context. It's designed to be naturally captured as part of
    monadic state progression rather than as a separate tracking system.
    """

    # Core identification
    action: str  # What was done ("tool_call", "think", "write_file", etc.)

    # Optional fields
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    info: dict[str, Any] = field(default_factory=dict)

    # Relationships - natural tree structure
    step_id: str = field(default_factory=lambda: str(time.time())[-8:])
    parent_id: str | None = None

    # Timing
    duration_ms: float | None = None  # Set when complete

    # Child evidence - using tuple for immutability
    children: tuple[Evidence, ...] = field(default_factory=tuple)

    def child(self, action: str, **kwargs) -> Evidence:
        """Create child evidence - allows building evidence tree naturally."""
        child = Evidence(action=action, parent_id=self.step_id, **kwargs)
        # Return new Evidence with child added to children tuple
        return Evidence(
            action=self.action,
            timestamp=self.timestamp,
            info=self.info,
            step_id=self.step_id,
            parent_id=self.parent_id,
            duration_ms=self.duration_ms,
            children=(*self.children, child)
        )

    def find_all(self, **filters) -> Iterator[Evidence]:
        """Find all evidence matching filters."""
        # Simple filters
        matches = True
        for attr_name, expected in filters.items():
            if not hasattr(self, attr_name):
                matches = False
                break
            actual = getattr(self, attr_name)
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
