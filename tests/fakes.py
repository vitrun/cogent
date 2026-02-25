from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from cogent.kernel import Control, Env, Result
from cogent.kernel.ports import MemoryPort, ModelPort, SinkPort, ToolPort
from cogent.kernel.tool import ToolCall


@dataclass
class FakeModel(ModelPort):
    responses: list[str]

    async def complete(self, prompt: str) -> str:
        _ = prompt
        if not self.responses:
            raise RuntimeError("No model responses available")
        return self.responses.pop(0)

    async def stream_complete(self, prompt: str, ctx: SinkPort) -> str:
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

    async def call(self, state: Any, call: ToolCall) -> Result[Any, Any]:
        handler = self.handlers.get(call.name)
        if handler is None:
            return Result(state, control=Control.Error(f"Tool not found: {call.name}"))
        try:
            value = handler(call.args)
            return Result(state, value=value)
        except Exception as e:
            return Result(state, control=Control.Error(str(e)))


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
