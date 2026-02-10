from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ModelPort(Protocol):
    async def complete(self, prompt: str) -> str: ...


class ToolPort(Protocol):
    async def call(self, name: str, args: dict[str, Any]) -> Any: ...


class MemoryPort(Protocol):
    async def recall(self, state: Any) -> Any: ...
    async def store(self, state: Any) -> None: ...


@dataclass
class Env:
    model: ModelPort
    tools: ToolPort
