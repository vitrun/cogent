# Cogent Sequential Handoffs Multi-Agent 实现方案

## 1. 设计目标

实现 Agent 之间的顺序传递（Sequential Handoffs），即：

```
User -> AgentA -> AgentB -> AgentC -> Result
```

当前一个 Agent 完成执行后，将控制权传递给下一个 Agent，形成链式调用。

## 2. 核心约束

基于 rule.md 的原则：

- **单一数据通道**: 状态必须通过显式 State 传递
- **无隐式主循环**: 使用尾递归组合而非 while 循环
- **无调度中心**: 不存在"控制器"组件控制流程
- **纯函数边界**: 每个 Step 不修改输入 State，返回新 State

## 3. 状态建模

### 3.1 HandoffState

```python
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar, Sequence

S = TypeVar("S")

@dataclass(frozen=True)
class HandoffState(Generic[S]):
    """顺序传递状态：记录当前执行到的 agent 索引和共享结果"""

    # 当前应该执行的 agent 索引
    current_index: int = 0

    # 所有 agent 的完整列表
    agent_sequence: tuple[S, ...] = field(default_factory=tuple)

    # 到目前为止收集的所有结果
    results: tuple[S, ...] = field(default_factory=tuple)

    # 共享的上下文数据（如 messages）
    shared_context: dict = field(default_factory=dict)

    def with_index(self, index: int) -> HandoffState[S]:
        return replace(self, current_index=index)

    def with_results(self, results: tuple[S, ...]) -> HandoffState[S]:
        return replace(self, results=results)

    def with_context(self, key: str, value: any) -> HandoffState[S]:
        new_context = {**self.shared_context, key: value}
        return replace(self, shared_context=new_context)

    def next(self) -> HandoffState[S]:
        """移动到下一个 agent"""
        return replace(self, current_index=self.current_index + 1)

    @property
    def current(self) -> S:
        """获取当前 agent"""
        return self.agent_sequence[self.current_index]

    @property
    def is_complete(self) -> bool:
        """是否已完成所有 agent"""
        return self.current_index >= len(self.agent_sequence)
```

### 3.2 Message (用于 agent 间传递)

```python
@dataclass(frozen=True)
class HandoffMessage:
    """Agent 之间传递的消息"""

    from_agent: str
    to_agent: str | None  # None 表示广播
    content: str
    metadata: dict = field(default_factory=dict)
```

## 4. 核心 Step 实现

### 4.1 handoff_step

```python
from cogent.core import Agent, Env, Result, Control, Step

async def handoff_step(
    state: HandoffState,
    agent: Agent,
    env: Env,
) -> Result[HandoffState, any]:
    """
    执行当前 agent 并将结果传递给下一个 agent。

    这是一个组合 Step，不是独立执行。
    """

    # 检查是否还有 agent 需要执行
    if state.is_complete:
        return Result(state, control=Control.Halt(state.results))

    # 执行当前 agent
    agent_result = await agent.run(env)

    # 提取结果
    if agent_result.control.kind == "error":
        return Result(state, control=Control.Error(agent_result.control.reason))

    if agent_result.control.kind == "halt":
        # 当前 agent 提前终止
        new_results = state.results + (agent_result.control.value,)
        return Result(
            state.with_results(new_results),
            control=Control.Halt(new_results)
        )

    # 继续执行：收集结果并准备传递给下一个 agent
    value = agent_result.control.value
    new_results = state.results + (value,)

    # 将结果注入到 shared_context 供下一个 agent 使用
    # 这里我们使用一个固定的 key "handoff_input"
    next_context = state.shared_context.copy()
    next_context["handoff_input"] = value

    next_state = (
        state
        .with_results(new_results)
        .with_context("handoff_input", value)
        .next()
    )

    return Result(next_state, control=Control.Continue(next_state))
```

### 4.2 包装下游 Agent 为 Tool

为了让上游 Agent 能够调用下游 Agent，需要将下游 Agent 封装为 Tool：

```python
class HandoffTool:
    """将 Agent 封装为 Tool，允许其他 Agent 通过 tool call 调用"""

    def __init__(self, agent: Agent, env: Env, agent_name: str):
        self.agent = agent
        self.env = env
        self.agent_name = agent_name

    async def call(self, name: str, args: dict) -> any:
        # 从 args 中提取传递给下游 agent 的输入
        task = args.get("task", "")

        # 使用下游 agent 执行
        result = await self.agent.run(self.env)

        if result.control.kind == "error":
            raise Exception(f"Agent {self.agent_name} failed: {result.control.reason}")

        return result.control.value
```

## 5. 组合 API

### 5.1 基本组合函数

```python
def handoff_to(
    next_agent: Agent,
    extract_input: callable = lambda ctx: ctx.get("handoff_input"),
) -> Step[HandoffState, HandoffState]:
    """
    创建一个 handoff step，传递给下一个 agent。

    Args:
        next_agent: 下一个要执行的 agent
        extract_input: 从 shared_context 中提取输入的函数
    """

    async def step(state: HandoffState, _: any, env: Env) -> Result[HandoffState, any]:
        if state.is_complete:
            return Result(state, control=Control.Halt(state.results))

        # 提取输入
        input_data = extract_input(state.shared_context)

        # 执行下一个 agent（这里需要包装成可以接受 input_data 的方式）
        # 实际实现中，可能需要用 Agent.start() 传入初始值
        agent_with_input = Agent.start(input_data).then(next_agent)

        result = await agent_with_input.run(env)

        # 处理结果
        if result.control.kind == "error":
            return Result(state, control=Control.Error(result.control.reason))

        value = result.control.value
        new_results = state.results + (value,)

        next_state = (
            state
            .with_results(new_results)
            .with_context("handoff_input", value)
            .next()
        )

        if state.next().is_complete:
            return Result(next_state, control=Control.Halt(new_results))

        return Result(next_state, control=Control.Continue(next_state))

    return step
```

### 5.2 使用示例

```python
# 定义三个顺序执行的 agent
agent_a = create_agent_a()  # 负责分析问题
agent_b = create_agent_b()  # 负责生成代码
agent_c = create_agent_c()  # 负责审查结果

# 构建 pipeline
handoff_pipeline = (
    Agent.start(initial_state)
    .then(agent_a.run)           # 执行 A
    .then(handoff_to(agent_b))   # 传递给 B
    .then(handoff_to(agent_c))   # 传递给 C
    .map(lambda results: results[-1])  # 取最终结果
)
```

## 6. 更简洁的 API 设计

### 6.1 chain 函数

```python
def chain(*agents: Agent) -> Agent[HandoffState, any]:
    """
    将多个 agent 链接在一起顺序执行。

    Usage:
        chain(agent_a, agent_b, agent_c)
    """

    async def run(env: Env) -> Result[HandoffState, any]:
        state = HandoffState(
            agent_sequence=tuple(agents),
            current_index=0,
            results=(),
            shared_context={}
        )

        while not state.is_complete:
            current_agent = state.current
            result = await current_agent.run(env)

            if result.control.kind == "error":
                return Result(state, control=Control.Error(result.control.reason))

            if result.control.kind == "halt":
                return Result(state, control=Control.Halt(result.control.value))

            # 继续
            value = result.control.value
            state = (
                state
                .with_results(state.results + (value,))
                .with_context("handoff_input", value)
                .next()
            )

        return Result(state, control=Control.Halt(state.results))

    return Agent(_run=run)
```

### 6.2 使用示例

```python
# 简洁用法
pipeline = chain(agent_analyze, agent_code, agent_review)

# 执行
result = await pipeline.run(env)
```

## 7. 错误处理策略

### 7.1 短路策略

当某个 agent 执行失败时：

```python
# 默认行为：短路，后续 agent 不再执行
# 失败通过 Result.control.kind == "error" 传播
```

### 7.2 恢复策略

允许在失败时恢复：

```python
pipeline = (
    chain(agent_a, agent_b, agent_c)
    .recover(lambda err: {"fallback": "default_value"})
)
```

## 8. 文件结构

```
src/cogent/multi/
├── __init__.py
├── state.py           # HandoffState, HandoffMessage
├── steps.py           # handoff_step, handoff_to
├── patterns.py        # chain, sequential
└── examples/
    └── simple_handoff.py
```

## 9. 实现顺序

1. **Phase 1**: 实现 HandoffState 和基础结构
2. **Phase 2**: 实现 chain 组合函数
3. **Phase 3**: 添加错误处理和恢复机制
4. **Phase 4**: 完善文档和测试

## 10. 待确认问题

1. **输入传递方式**: 当前设计使用 shared_context，是否需要更明确的输入建模？
2. **结果聚合**: 最终返回 list[tuple] 还是直接取最后一个？
3. **类型泛化**: HandoffState 是否需要支持泛型类型？
