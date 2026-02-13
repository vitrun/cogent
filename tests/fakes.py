from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from cogent.agents import ReActState
from cogent.ports.env import Env, MemoryPort, ModelPort, Sink, ToolPort


@dataclass
class FakeModel(ModelPort):
    responses: list[str]

    async def complete(self, prompt: str) -> str:
        _ = prompt
        if not self.responses:
            raise RuntimeError("No model responses available")
        return self.responses.pop(0)

    async def stream_complete(self, prompt: str, ctx: Sink) -> str:
        _ = prompt
        if not self.responses:
            raise RuntimeError("No model responses available")
        response = self.responses.pop(0)
        # Simulate token streaming
        for char in response:
            await ctx.send(char)
            await asyncio.sleep(0.01)
        await ctx.close()
        return response


@dataclass
class FakeTools(ToolPort):
    handlers: dict[str, Any]

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        handler = self.handlers.get(name)
        if handler is None:
            raise RuntimeError(f"Tool not found: {name}")
        return handler(args)


@dataclass
class FakeMemory(MemoryPort):
    stored: list[list[Any]] = field(default_factory=list)

    async def append(self, records: list[Any]) -> None:
        self.stored.append(records)

    async def query(self, query: Any) -> list[Any]:
        return []

    async def clear(self) -> None:
        self.stored.clear()

    async def close(self) -> None:
        pass


def make_fake_env(responses: list[str] | None = None) -> Env:
    return Env(
        model=FakeModel(responses or []),
        tools=FakeTools({}),
    )
