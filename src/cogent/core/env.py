from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .trace import TraceContext


class RuntimeContext(Protocol):
    async def emit(self, chunk: str) -> None: ...
    async def close(self) -> None: ...


class ModelPort(Protocol):
    async def complete(self, prompt: str) -> str: ...
    async def stream_complete(self, prompt: str, ctx: RuntimeContext) -> str: ...


class ToolPort(Protocol):
    async def call(self, name: str, args: dict[str, Any]) -> Any: ...


class MemoryPort(Protocol):
    async def recall(self, state: Any) -> Any: ...
    async def store(self, state: Any) -> None: ...


@dataclass
class Env:
    model: ModelPort
    tools: ToolPort | None = None
    memory: MemoryPort | None = None
    runtime_context: RuntimeContext | None = None
    trace: TraceContext | None = None
