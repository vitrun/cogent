from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Self, Dict, Optional, TypeVar, Generic
from pydantic import BaseModel, Field

from .agent import Step
from .result import Control, Result


class ToolUse(BaseModel):
    id: str
    name: str
    args: dict[str, Any]


class ToolResult(BaseModel):
    id: str
    content: str
    failed: bool = False

    @classmethod
    def success(cls, id: str, content: str) -> Self:
        return cls(id=id, content=content, failed=False)

    @classmethod
    def failure(cls, id: str, content: str) -> Self:
        return cls(id=id, content=content, failed=True)


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ToolParameter]

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        for param_name, param in self.parameters.items():
            if param.required and param_name not in params:
                return False
        return True


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, name: str, handler: ToolHandler, definition: Optional[ToolDefinition] = None) -> None:
        self._tools[name] = handler
        if definition:
            self._definitions[name] = definition

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        return self._definitions.get(name)

    async def run(self, env: Any, state: Any, call: ToolUse) -> ToolResult:
        handler = self._tools.get(call.name)
        if handler is None:
            return ToolResult.failure(call.id, f"Tool not found: {call.name}")
        
        definition = self._definitions.get(call.name)
        if definition and not definition.validate_parameters(call.args):
            return ToolResult.failure(call.id, f"Invalid parameters for tool: {call.name}")
            
        try:
            content = handler(env, state, call)
            if inspect.isawaitable(content):
                content = await content
            return ToolResult.success(call.id, content)
        except Exception as exc:
            return ToolResult.failure(call.id, f"Tool error: {exc}")


ToolHandler = Callable[[Any, Any, ToolUse], Awaitable[str] | str]


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def example_tool(_env: Any, state: Any, call: ToolUse) -> str:
        query = call.args.get("query")
        return f"No results for query: {query}"

    registry.register("example", example_tool)
    return registry


S = TypeVar("S")
V = TypeVar("V")
R = TypeVar("R")


def create_tool_execution_step(registry: Optional[ToolRegistry] = None) -> Step[S, ToolUse, ToolResult]:
    """
    创建工具执行步骤函数
    
    Args:
        registry: 工具注册表，默认为default_registry()
        
    Returns:
        异步步骤函数，接收状态、工具调用和环境，返回步骤结果
    """
    async def step(state: S, call: ToolUse, env: Any) -> Result[S, ToolResult]:
        """
        工具执行步骤
        
        Args:
            state: 当前状态
            call: 工具调用
            env: 环境
            
        Returns:
            包含工具执行结果的步骤结果
        """
        used_registry = registry or default_registry()
        try:
            result = await used_registry.run(env, state, call)
            if result.failed:
                return Result(state, control=Control.Error(result.content))
            return Result(state, control=Control.Continue(result))
        except Exception as exc:
            return Result(state, control=Control.Error(f"Tool execution failed: {exc}"))
    
    return step
