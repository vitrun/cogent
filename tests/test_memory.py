from __future__ import annotations

import pytest
from dataclasses import dataclass
from cogent.core.memory import Context, InMemoryContext
from cogent.core.env import Env, ModelPort, ToolPort


@dataclass
class MockModelPort(ModelPort):
    async def complete(self, prompt: str) -> str:
        return "mock response"


@dataclass
class MockToolPort(ToolPort):
    async def call(self, name: str, args: dict[str, any]) -> any:
        return "mock result"


# Create a mock Env object
mock_env = Env(
    model=MockModelPort(),
    tools=MockToolPort()
)


@dataclass
class StateWithContext:
    """State class containing context."""
    counter: int
    context: Context


@pytest.mark.asyncio
async def test_context_basic_operations():
    """
    Test Context basic operations.
    """
    # Test basic context write and read
    initial_context = InMemoryContext()

    # Test append operation
    context1 = initial_context.append({"obs": "foo"})
    context2 = context1.append({"obs": "bar"})

    # Test query operation
    foo_entries = list(context2.query(lambda e: e["obs"] == "foo"))
    assert len(foo_entries) == 1
    assert foo_entries[0]["obs"] == "foo"

    # Test snapshot operation
    snapshot = context2.snapshot()
    assert len(snapshot) == 2
    assert snapshot[0]["obs"] == "foo"
    assert snapshot[1]["obs"] == "bar"

    # Test trim operation
    trimmed_context = context2.trim(lambda entries: entries[-1:])  # Keep last one
    trimmed_snapshot = trimmed_context.snapshot()
    assert len(trimmed_snapshot) == 1
    assert trimmed_snapshot[0]["obs"] == "bar"


@pytest.mark.asyncio
async def test_context_immutability():
    """
    Test Context immutability - original context unchanged after operations.
    """
    initial_context = InMemoryContext()

    # Append to create new context
    new_context = initial_context.append({"obs": "test"})

    # Original should be unchanged
    original_snapshot = initial_context.snapshot()
    assert len(original_snapshot) == 0

    # New context should have the entry
    new_snapshot = new_context.snapshot()
    assert len(new_snapshot) == 1
    assert new_snapshot[0]["obs"] == "test"


@pytest.mark.asyncio
async def test_context_determinism():
    """
    Test Context determinism.
    """
    context1 = InMemoryContext()
    context1 = context1.append({"obs": "test"})
    snapshot1 = context1.snapshot()

    context2 = InMemoryContext()
    context2 = context2.append({"obs": "test"})
    snapshot2 = context2.snapshot()

    assert snapshot1 == snapshot2


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_context_basic_operations())
    asyncio.run(test_context_immutability())
    asyncio.run(test_context_determinism())
    print("All tests passed!")
