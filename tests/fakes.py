from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from cogent.core import Env, MemoryPort, ModelPort, ToolPort
from cogent.starter import ReActState

@dataclass
class FakeModel(ModelPort):
    responses: list[str]

    async def complete(self, prompt: str) -> str:
        _ = prompt
        if not self.responses:
            raise RuntimeError("No model responses available")
        return self.responses.pop(0)

    async def stream_complete(self, prompt: str, ctx) -> str:
        _ = prompt
        if not self.responses:
            raise RuntimeError("No model responses available")
        response = self.responses.pop(0)
        # 模拟token流
        for char in response:
            await ctx.emit(char)
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
    stored: list[ReActState]

    async def recall(self, state: ReActState) -> ReActState:
        return state

    async def store(self, state: ReActState) -> None:
        self.stored.append(state)


def make_fake_env(responses: list[str] | None = None) -> Env:
    return Env(
        model=FakeModel(responses or []),
        tools=FakeTools({}),
    )
