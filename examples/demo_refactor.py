#!/usr/bin/env python3
"""
Minimal demonstration of the refactored Cogent Agent runtime.

Shows:
1) InMemoryContext creation and immutable operations
2) DummyMemoryPort implementation
3) DummySink implementation
4) Explicit sink and memory usage in steps
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from cogent.core import Agent, Control, Env, MemoryPort, ModelPort, Result, Sink, ToolPort
from cogent.core.memory import Context, InMemoryContext


# =============================================================================
# PART 1: Context (was Memory) - Immutable state
# =============================================================================

def demo_context():
    """Demonstrate InMemoryContext immutable operations."""
    print("=== Context Demo ===")

    # 1) Create InMemoryContext
    ctx = InMemoryContext()
    print(f"Initial context entries: {ctx.snapshot()}")

    # 2) Append entries immutably
    ctx1 = ctx.append("first entry")
    ctx2 = ctx1.append("second entry")
    ctx3 = ctx2.append("third entry")

    print(f"After appends: {ctx3.snapshot()}")

    # 3) Trim context
    def keep_last_two(entries: list[Any]) -> list[Any]:
        return entries[-2:]

    ctx_trimmed = ctx3.trim(keep_last_two)
    print(f"After trim (keep last 2): {ctx_trimmed.snapshot()}")

    # 4) Show original context unchanged (immutability)
    print(f"Original ctx still empty: {ctx.snapshot()}")
    print(f"Original ctx1 still has 1: {ctx1.snapshot()}")
    print()


# =============================================================================
# PART 2: Dummy implementations
# =============================================================================

@dataclass
class DummyModel(ModelPort):
    """Dummy model that returns static responses."""

    async def complete(self, prompt: str) -> str:
        return f"Response to: {prompt[:20]}..."

    async def stream_complete(self, prompt: str, ctx: Sink) -> str:
        response = f"Streaming: {prompt[:20]}..."
        for char in response:
            await ctx.send(char)
        await ctx.close()
        return response


@dataclass
class DummyTools(ToolPort):
    """Dummy tools that return static responses."""

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        return f"Tool {name} called with {args}"


class DummySink(Sink):
    """Dummy sink that collects chunks."""

    def __init__(self):
        self.chunks: list[Any] = []

    async def send(self, chunk: Any) -> None:
        self.chunks.append(chunk)

    async def close(self) -> None:
        pass


class DummyMemoryPort(MemoryPort):
    """Dummy memory port that stores records."""

    def __init__(self):
        self.records: list[list[Any]] = []

    async def append(self, records: list[Any]) -> None:
        self.records.append(records)

    async def query(self, query: Any) -> list[Any]:
        return []

    async def clear(self) -> None:
        self.records.clear()

    async def close(self) -> None:
        pass


# =============================================================================
# PART 3: Steps that explicitly use sink and memory
# =============================================================================

async def prompt_step(state: Context, task: str, env: Env) -> Result[Context, str]:
    """Step that builds a prompt."""
    context_entries = state.snapshot()
    prompt = f"Task: {task}\nContext: {context_entries}"
    return Result(state, value=prompt, control=Control.Continue())


async def think_step(state: Context, prompt: str, env: Env) -> Result[Context, str]:
    """Step that calls model - optionally uses sink for streaming."""
    if env.sink:
        # Explicitly use sink for streaming
        response = await env.model.stream_complete(prompt, env.sink)
    else:
        response = await env.model.complete(prompt)
    return Result(state, value=response, control=Control.Continue())


async def store_step(state: Context, value: str, env: Env) -> Result[Context, str]:
    """Step that explicitly stores to memory port."""
    if env.memory:
        # Explicitly call memory.append() - NOT automatic
        await env.memory.append([{"content": value, "type": "response"}])
    return Result(state, value=value, control=Control.Continue())


# =============================================================================
# PART 4: Run demonstration
# =============================================================================

async def main():
    # Create dummy implementations
    sink = DummySink()
    memory = DummyMemoryPort()

    # 7) Attach them to Env
    env = Env(
        model=DummyModel(),
        tools=DummyTools(),
        sink=sink,
        memory=memory,
    )

    # Initial context state
    initial_context = InMemoryContext().append("initial context entry")

    # 8) Explicitly call sink.send() from a mock step
    print("=== Sink Demo ===")
    print(f"Sink chunks before: {sink.chunks}")

    # Run a step that uses sink
    agent = (
        Agent.start(initial_context, "test task")
        .then(prompt_step)
        .then(think_step)
    )
    result = await agent.run(env)

    print(f"Sink chunks after: {sink.chunks}")
    print()

    # 9) Explicitly call memory.append() from a mock step
    print("=== Memory Demo ===")
    print(f"Memory records before: {memory.records}")

    # Run a step that stores to memory
    agent2 = (
        Agent.start(result.state, "store task")
        .then(store_step)
    )
    result2 = await agent2.run(env)

    print(f"Memory records after: {memory.records}")
    print()

    print("=== All demonstrations complete ===")


if __name__ == "__main__":
    demo_context()
    asyncio.run(main())
