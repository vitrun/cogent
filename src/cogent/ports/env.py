"""Port definitions for Cogent environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from cogent.kernel.trace import TraceContext


class Sink(Protocol):
    """Output streaming port."""

    async def send(self, chunk: Any) -> None:
        """Send a chunk to the output stream."""
        ...

    async def close(self) -> None:
        """Close the output stream."""
        ...


class ModelPort(Protocol):
    """Model API port."""

    async def complete(self, prompt: str) -> str: ...
    async def stream_complete(self, prompt: str, ctx: Sink) -> str: ...


class ToolPort(Protocol):
    """Tool execution port."""

    async def call(self, name: str, args: dict[str, Any]) -> Any: ...


class MemoryPort(Protocol):
    """
    External knowledge store.
    Infrastructure-level.
    Not part of state.
    """

    async def append(self, records: list[Any]) -> None:
        """Append records to the external store."""
        ...

    async def query(self, query: Any) -> list[Any]:
        """Query the external store."""
        ...

    async def clear(self) -> None:
        """Clear the external store."""
        ...

    async def close(self) -> None:
        """Close the external store."""
        ...


@dataclass
class Env:
    """Environment aggregation - combines all ports."""
    model: ModelPort
    tools: ToolPort | None = None
    memory: MemoryPort | None = None
    trace: TraceContext | None = None
    sink: Sink | None = None
