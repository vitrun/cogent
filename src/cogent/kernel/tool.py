"""Pure semantic tool abstractions."""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class ToolCall(BaseModel):
    """
    A semantic request to invoke a tool.

    Pure data. No execution semantics.
    """
    name: str
    args: dict[str, Any]
