# Cogent Multi-Agent Primitives

在现有 `Agent[S, V]`, `MultiState`, `Control`, `Env` 基础上扩展四个 primitive。

---

## 现有类型扩展

```python
from cogent.core import Agent, Env, Control, Result

# 新增 MultiState
@dataclass(frozen=True)
class MultiState:
    """多 agent 共享状态"""
    current: str
    shared: tuple[Any, ...]
    locals: Mapping[str, Any]


class AgentRegistry:
    """Agent 注册表，通过名称查找 agent"""
    def get(self, name: str) -> Agent[Any, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class MultiEnv(Env[MultiState]):
    """扩展 Env，包含 AgentRegistry"""
    registry: AgentRegistry
```

---

## Primitive 1: handoff

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
        - Control 透明传递
    """
    async def _run(env: MultiEnv) -> Result[MultiState, Any]:
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
            control=result.control,
        )

    return Agent(_run)
```

---

## Primitive 2: emit

```python
def emit(msg: Any) -> Agent[MultiState, None]:
    """
    发送消息到共享空间。

    Semantics:
        - 将消息追加到 shared tuple
        - 不改变 current / locals
        - Control 固定为 Continue
    """
    async def _run(env: MultiEnv) -> Result[MultiState, None]:
        ms = env.state
        new_state = MultiState(
            current=ms.current,
            shared=ms.shared + (msg,),
            locals=ms.locals,
        )
        return Result(state=new_state, value=None, control=Control.Continue)

    return Agent(_run)
```

---

## Primitive 3: route

```python
def route(selector: Callable[[MultiState], str]) -> Agent[MultiState, Any]:
    """
    动态路由到目标 agent。

    Semantics:
        - 根据当前 MultiState 选择目标 agent
        - Control 透明传递
    """
    async def _run(env: MultiEnv) -> Result[MultiState, Any]:
        target = selector(env.state)
        return await handoff(target).run(env)

    return Agent(_run)
```

---

## Primitive 4: concurrent

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
    """
    async def _run(env: MultiEnv) -> Result[MultiState, list[Any]]:
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

## 使用示例

```python
# examples/multi_agent_demo.py

import asyncio
from dataclasses import dataclass, replace
from typing import Any, Mapping
from cogent.core import Agent, Env, Control, Result


# --- MultiState & Env ---

@dataclass(frozen=True)
class MultiState:
    current: str
    shared: tuple[Any, ...]
    locals: Mapping[str, Any]


class AgentRegistry:
    def __init__(self, agents: dict[str, Agent]):
        self._agents = agents

    def get(self, name: str) -> Agent[Any, Any]:
        return self._agents[name]


@dataclass(frozen=True)
class MultiEnv(Env[MultiState]):
    registry: AgentRegistry


# --- Primitives ---

def handoff(target: str) -> Agent[MultiState, Any]:
    async def _run(env: MultiEnv) -> Result[MultiState, Any]:
        ms = env.state
        agent = env.registry.get(target)
        local_state = ms.locals.get(target)
        sub_env = env.with_state(local_state)
        result = await agent.run(sub_env)
        new_locals = dict(ms.locals)
        new_locals[target] = result.state
        new_state = MultiState(current=target, shared=ms.shared, locals=new_locals)
        return Result(state=new_state, value=result.value, control=result.control)
    return Agent(_run)


def emit(msg: Any) -> Agent[MultiState, None]:
    async def _run(env: MultiEnv) -> Result[MultiState, None]:
        ms = env.state
        new_state = MultiState(current=ms.current, shared=ms.shared + (msg,), locals=ms.locals)
        return Result(state=new_state, value=None, control=Control.Continue)
    return Agent(_run)


def route(selector: Callable[[MultiState], str]) -> Agent[MultiState, Any]:
    async def _run(env: MultiEnv) -> Result[MultiState, Any]:
        target = selector(env.state)
        return await handoff(target).run(env)
    return Agent(_run)


def concurrent(
    agents: list[Agent[MultiState, Any]],
    merge_state: Callable[[list[MultiState]], MultiState],
) -> Agent[MultiState, list[Any]]:
    import asyncio
    async def _run(env: MultiEnv) -> Result[MultiState, list[Any]]:
        tasks = [agent.run(env) for agent in agents]
        results = await asyncio.gather(*tasks)
        states = [r.state for r in results]
        values = [r.value for r in results]
        controls = [r.control for r in results]
        merged_state = merge_state(states)
        merged_control = Control.merge(controls)
        return Result(state=merged_state, value=values, control=merged_control)
    return Agent(_run)


# --- 定义 Agent ---

def create_analyzer() -> Agent[MultiState, str]:
    async def _run(env: MultiEnv) -> Result[MultiState, str]:
        return Result(state=env.state, value="analyzed", control=Control.Continue)
    return Agent(_run)


def create_executor() -> Agent[MultiState, str]:
    async def _run(env: MultiEnv) -> Result[MultiState, str]:
        return Result(state=env.state, value="executed", control=Control.Continue)
    return Agent(_run)


def create_reviewer() -> Agent[MultiState, str]:
    async def _run(env: MultiEnv) -> Result[MultiState, str]:
        return Result(state=env.state, value="reviewed", control=Control.Continue)
    return Agent(_run)


# --- 执行 ---

async def main():
    # 注册
    registry = AgentRegistry({
        "analyzer": create_analyzer(),
        "executor": create_executor(),
        "reviewer": create_reviewer(),
    })

    # 初始状态
    initial_state = MultiState(
        current="",
        shared=("task",),
        locals={},
    )

    env = MultiEnv(state=initial_state, registry=registry)

    # 方式1: handoff chain
    pipeline = handoff("analyzer").then("executor").then("reviewer")

    # 方式2: concurrent
    def merge_all(states: list[MultiState]) -> MultiState:
        return MultiState(
            current=states[-1].current,
            shared=sum([list(s.shared) for s in states], []),
            locals={k: v for s in states for k, v in s.locals.items()},
        )

    pipeline = concurrent(
        [handoff("analyzer"), handoff("executor")],
        merge_state=merge_all,
    )

    result = await pipeline.run(env)
    print(result.value)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 代数律

| 律 | 验证 |
|---|------|
| **Closure** | 所有 primitive 返回 `Agent[MultiState, V]` |
| **Control Monotonicity** | Abort > Retry > Continue |
| **No Runtime Privilege** | 无 scheduler |
| **No Mutation** | MultiState frozen |
| **Registry NOT in State** | AgentRegistry 在 Env |

---

## 文件结构

```
src/cogent/multi/
├── __init__.py
├── state.py      # MultiState
├── primitives.py  # handoff, emit, route, concurrent
└── types.py      # AgentRegistry, MultiEnv

examples/
└── multi_agent_demo.py
```
