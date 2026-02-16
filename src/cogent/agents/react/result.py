"""ReactAgent result types."""

from __future__ import annotations

from dataclasses import dataclass

from cogent.kernel import Trace


@dataclass(frozen=True)
class ReactResult:
    """Return type for ReactAgent - hides kernel Result.

    Attributes:
        value: The final answer from the agent.
        steps: Number of ReAct iterations executed.
        trace: Trace object for debugging (None if trace=False).
    """

    value: str
    steps: int
    trace: Trace | None = None
