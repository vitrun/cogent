# Cogent Multi-Agent Primitives 实现方案

严格封闭于 `Agent[MultiState, V]` 的代数算子。

---

## 1. 类型定义（框架）

```python
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Awaitable, TypeVar, Generic, Sequence
from enum import Enum
import asyncio

S = TypeVar("S")
V = TypeVar("V")
U = TypeVar("U")


class Control(Enum):
    Continue = 0
    Retry = 1
    Abort = 2

    @staticmethod
    def merge(cs: list["Control"]) -> "Control":
        if Control.Abort in cs:
            return Control.Abort
        if Control.Retry in cs:
            return Control.Retry
        return Control.Continue


@dataclass(frozen=True)
class Result(Generic[S, V]):
    state: S
    value: V
    control: Control

    def map_state(self, f: Callable[[S], S]) -> "Result[S, V]":
        return Result(f(self.state), self.value, self.control)

    def map_value(self, f: Callable[[V], U]) -> "Result[S, U]":
        return Result(self.state, f(self.value), self.control)


class Agent(Generic[S, V]):
    def __init__(self, run: Callable[["Env[S]"], Awaitable[Result[S, V]]]):
        self._run = run

    async def run(self, env: "Env[S]") -> Result[S, V]:
        return await self._run(env)

    def map_value(self, f: Callable[[V], U]) -> "Agent[S, U]":
        async def _run(env: Env[S]) -> Result[S, U]:
            r = await self.run(env)
            return r.map_value(f)
        return Agent(_run)


@dataclass(frozen=True)
class MultiState:
    current: str
    shared: tuple[Any, ...]
    locals: Mapping[str, Any]


class AgentRegistry:
    def get(self, name: str) -> Agent[Any, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class Env(Generic[S]):
    state: S
    registry: AgentRegistry

    def with_state(self, new_state: S) -> "Env[S]":
        return replace(self, state=new_state)
```

---

## 2. Primitive 1: handoff

```python
def handoff(target: str) -> Agent[MultiState, Any]:
    """
    切换到目标 agent 执行。

    Semantics:
        - 通过 registry 查找目标 agent
        - 从 MultiState.locals 提取其本地状态
        - 执行目标 agent
        - 更新 locals[target]
        - 设置 current = target
        - Control 透明传递，不 collapse

    Returns:
        Agent[MultiState, Any]
    """
    async def _run(env: Env[MultiState]) -> Result[MultiState, Any]:
        ms = env.state
        agent = env.registry.get(target)

        local_state = ms.locals.get(target)
        sub_env = env.with_state(local_state)

        result = await agent.run(sub_env)

        new_locals = dict(ms.locals)
        new_locals[target] = result.state

        new_state = MultiState(
            current=target,
            shared=ms.shared,
            locals=new_locals,
        )

        return Result(
            state=new_state,
            value=result.value,
            control=result.control,  # 透明传递
        )

    return Agent(_run)
```

---

## 3. Primitive 2: emit

```python
def emit(msg: Any) -> Agent[MultiState, None]:
    """
    发送消息到共享空间。

    Semantics:
        - 将消息追加到 shared tuple
        - 不改变 current agent
        - 不改变 locals
        - Control 固定为 Continue

    Args:
        msg: 要发送的消息

    Returns:
        Agent[MultiState, None]
    """
    async def _run(env: Env[MultiState]) -> Result[MultiState, None]:
        ms = env.state

        new_state = MultiState(
            current=ms.current,
            shared=ms.shared + (msg,),
            locals=ms.locals,
        )

        return Result(
            state=new_state,
            value=None,
            control=Control.Continue,
        )

    return Agent(_run)
```

---

## 4. Primitive 3: route

```python
def route(selector: Callable[[MultiState], str]) -> Agent[MultiState, Any]:
    """
    动态路由到目标 agent。

    Semantics:
        - 根据当前 MultiState 选择目标 agent
        - 本质是 handoff 的高阶版本
        - Control 透明传递

    Args:
        selector: 接收 MultiState，返回目标 agent 名称

    Returns:
        Agent[MultiState, Any]
    """
    async def _run(env: Env[MultiState]) -> Result[MultiState, Any]:
        target = selector(env.state)
        return await handoff(target).run(env)

    return Agent(_run)
```

---

## 5. Primitive 4: concurrent

```python
def concurrent(
    agents: list[Agent[MultiState, Any]],
    merge_state: Callable[[list[MultiState]], MultiState],
) -> Agent[MultiState, list[Any]]:
    """
    并发执行多个 agent。

    Semantics:
        - 使用相同初始 env 执行所有 agent
        - 收集所有结果
        - 显式合并状态
        - 通过 Control.merge 合并控制流
        - 不使用 return_exceptions=True
        - Agent.run 内部已吸收异常

    Args:
        agents: 要并发执行的 agent 列表
        merge_state: 如何合并多个 MultiState

    Returns:
        Agent[MultiState, list[Any]]
    """
    async def _run(env: Env[MultiState]) -> Result[MultiState, list[Any]]:
        tasks = [agent.run(env) for agent in agents]
        results = await asyncio.gather(*tasks)

        states = [r.state for r in results]
        values = [r.value for r in results]
        controls = [r.control for r in results]

        merged_state = merge_state(states)
        merged_control = Control.merge(controls)

        return Result(
            state=merged_state,
            value=values,
            control=merged_control,
        )

    return Agent(_run)
```

---

## 6. 组合算子（衍生）

### 6.1 then（Sequential 链接）

```python
def then(
    self: Agent[MultiState, Any],
    next_agent: str,
) -> Agent[MultiState, Any]:
    """链接当前 agent 和下一个 agent"""
    async def _run(env: Env[MultiState]) -> Result[MultiState, Any]:
        result = await self.run(env)

        if result.control != Control.Continue:
            return result

        return await handoff(next_agent).run(env.with_state(result.state))

    return Agent(_run)
```

### 6.2 sequential（顺序执行）

```python
def sequential(
    agent_names: Sequence[str],
) -> Agent[MultiState, list[Any]]:
    """顺序执行一系列 agent"""
    if not agent_names:
        async def _run(env: Env[MultiState]) -> Result[MultiState, list[Any]]:
            return Result(state=env.state, value=[], control=Control.Continue)
        return Agent(_run)

    async def _run(env: Env[MultiState]) -> Result[MultiState, list[Any]]:
        results: list[Any] = []

        for target in agent_names:
            result = await handoff(target).run(env)

            results.append(result.value)

            if result.control == Control.Abort:
                return Result(state=result.state, value=results, control=Control.Abort)

        return Result(state=env.state, value=results, control=Control.Continue)

    return Agent(_run)
```

### 6.3 broadcast（广播消息）

```python
def broadcast(msg: Any, targets: list[str]) -> Agent[MultiState, None]:
    """向多个 agent 发送消息"""
    async def _run(env: Env[MultiState]) -> Result[MultiState, None]:
        ms = env.state

        # 依次 emit 到共享空间
        for _ in targets:
            result = await emit(msg).run(env)
            if result.control != Control.Continue:
                return Result(state=result.state, value=None, control=result.control)

        return Result(state=env.state, value=None, control=Control.Continue)

    return Agent(_run)
```

---

## 7. 使用示例

```python
# --- 定义 Agent ---

def create_analyzer() -> Agent[MultiState, str]:
    async def _run(env: Env[MultiState]) -> Result[MultiState, str]:
        return Result(state=env.state, value="analyzed", control=Control.Continue)
    return Agent(_run)


def create_executor() -> Agent[MultiState, str]:
    async def _run(env: Env[MultiState]) -> Result[MultiState, str]:
        return Result(state=env.state, value="executed", control=Control.Continue)
    return Agent(_run)


def create_reviewer() -> Agent[MultiState, str]:
    async def _run(env: Env[MultiState]) -> Result[MultiState, str]:
        return Result(state=env.state, value="reviewed", control=Control.Continue)
    return Agent(_run)


class Registry(AgentRegistry):
    def __init__(self):
        self._agents = {
            "analyzer": create_analyzer(),
            "executor": create_executor(),
            "reviewer": create_reviewer(),
        }

    def get(self, name: str) -> Agent[Any, Any]:
        return self._agents[name]


# --- 组合使用 ---

# handoff + then
pipeline = handoff("analyzer").then("executor").then("reviewer")

# sequential
pipeline = sequential(["analyzer", "executor", "reviewer"])

# concurrent + merge
def merge_all(states: list[MultiState]) -> MultiState:
    return MultiState(
        current=states[-1].current,
        shared=sum([list(s.shared) for s in states], []),
        locals={k: v for s in states for k, v in s.locals.items()},
    )

pipeline = concurrent(
    [handoff("analyzer"), handoff("executor"), handoff("reviewer")],
    merge_state=merge_all,
)

# route
def selector(ms: MultiState) -> str:
    if "analyze" in str(ms.shared):
        return "analyzer"
    return "executor"

pipeline = route(selector)


# --- 执行 ---

async def main():
    initial_state = MultiState(
        current="",
        shared=("task",),
        locals={},
    )

    env = Env(state=initial_state, registry=Registry())

    result = await pipeline.run(env)
    print(result.value)
```

---

## 8. 代数律

| 律 | 验证 |
|---|------|
| **Closure** | 所有 primitive 返回 `Agent[MultiState, V]` |
| **Control Monotonicity** | Abort > Retry > Continue |
| **No Runtime Privilege** | 无 scheduler |
| **No Mutation** | MultiState frozen |
| **Registry NOT in State** | AgentRegistry 在 Env |

---

## 9. 文件结构

```
src/cogent/multi/
├── __init__.py
├── state.py       # MultiState
├── primitives.py  # handoff, emit, route, concurrent
├── combinators.py # then, sequential, broadcast
└── examples/
    └── demo.py
```
