from __future__ import annotations

import pytest
from dataclasses import dataclass
from cogent.core.memory import Memory, SimpleMemory
from cogent.core.env import Env, ModelPort, ToolPort


@dataclass
class MockModelPort(ModelPort):
    async def complete(self, prompt: str) -> str:
        return "mock response"


@dataclass
class MockToolPort(ToolPort):
    async def call(self, name: str, args: dict[str, any]) -> any:
        return "mock result"


# 创建一个模拟的 Env 对象
mock_env = Env(
    model=MockModelPort(),
    tools=MockToolPort()
)


@dataclass
class StateWithMemory:
    """包含内存的状态类"""
    counter: int
    memory: Memory


@pytest.mark.asyncio
async def test_memory_basic_operations():
    """
    测试 Memory 的基本操作
    """
    # 测试基本的内存写入和读取
    initial_memory = SimpleMemory()
    
    # 测试 append 操作
    memory1 = initial_memory.append({"obs": "foo"})
    memory2 = memory1.append({"obs": "bar"})
    
    # 测试 query 操作
    foo_entries = list(memory2.query(lambda e: e["obs"] == "foo"))
    assert len(foo_entries) == 1
    assert foo_entries[0]["obs"] == "foo"
    
    # 测试 snapshot 操作
    snapshot = memory2.snapshot()
    assert len(snapshot) == 2
    assert snapshot[0]["obs"] == "foo"
    assert snapshot[1]["obs"] == "bar"
    
    # 测试 trim 操作
    trimmed_memory = memory2.trim(lambda entries: entries[-1:])  # 保留最后一个
    trimmed_snapshot = trimmed_memory.snapshot()
    assert len(trimmed_snapshot) == 1
    assert trimmed_snapshot[0]["obs"] == "bar"


@pytest.mark.asyncio
async def test_memory_integration():
    """
    测试 Memory 的集成使用
    """
    # 测试内存的确定性
    memory1 = SimpleMemory()
    memory1 = memory1.append({"obs": "test"})
    snapshot1 = memory1.snapshot()
    
    memory2 = SimpleMemory()
    memory2 = memory2.append({"obs": "test"})
    snapshot2 = memory2.snapshot()
    
    assert snapshot1 == snapshot2


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_memory_basic_operations())
    asyncio.run(test_memory_integration())
    print("All tests passed!")
