"""Core types for the Cogent kernel - pure data definitions."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel


class ToolUse(BaseModel):
    """Tool call request."""
    id: str
    name: str
    args: dict[str, Any]


class ToolResult(BaseModel):
    """Tool call result."""
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
    """Tool parameter definition."""
    name: str
    type: str
    description: str
    required: bool
    default: Any | None = None


class ToolDefinition(BaseModel):
    """Tool definition for registration."""
    name: str
    description: str
    parameters: dict[str, ToolParameter]

    def validate_parameters(self, params: dict[str, Any]) -> bool:
        for param_name, param in self.parameters.items():
            if param.required and param_name not in params:
                return False
        return True
