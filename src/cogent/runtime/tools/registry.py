"""Tool registry implementation."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from cogent.kernel.result import Control, Result
from cogent.kernel.tool import ToolDefinition, ToolResult, ToolUse

ToolHandler = Callable[[Any, Any, ToolUse], Awaitable[str] | str]


class ToolRegistry:
    """Registry for managing tool handlers and definitions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, name: str, handler: ToolHandler, definition: ToolDefinition | None = None) -> None:
        """Register a tool with its handler and optional definition."""
        self._tools[name] = handler
        if definition:
            self._definitions[name] = definition

    def get_definition(self, name: str) -> ToolDefinition | None:
        """Get tool definition by name."""
        return self._definitions.get(name)

    async def run(self, env: Any, state: Any, call: ToolUse) -> ToolResult:
        """Execute a tool call."""
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


def default_registry() -> ToolRegistry:
    """Create a default registry with example tools."""
    registry = ToolRegistry()

    def example_tool(_env: Any, state: Any, call: ToolUse) -> str:
        query = call.args.get("query")
        return f"No results for query: {query}"

    registry.register("example", example_tool)
    return registry


S = TypeVar("S")
V = TypeVar("V")
R = TypeVar("R")


def create_tool_execution_step(registry: ToolRegistry | None = None):
    """Create a tool execution step function.

    Args:
        registry: Tool registry, defaults to default_registry()

    Returns:
        Async step function that executes tool calls
    """
    async def step(state: S, call: ToolUse, env: Any) -> Result[S, ToolResult]:
        used_registry = registry or default_registry()
        try:
            result = await used_registry.run(env, state, call)
            if result.failed:
                return Result(state, control=Control.Error(result.content))
            return Result(state, value=result, control=Control.Continue())
        except Exception as exc:
            return Result(state, control=Control.Error(f"Tool execution failed: {exc}"))

    return step
