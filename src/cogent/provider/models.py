"""Common message models for provider formatters."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextBlock:
    """Text content block."""
    text: str
    type: str = "text"


@dataclass(frozen=True)
class ImageBlock:
    """Image content block."""
    source: dict[str, Any]
    type: str = "image"


@dataclass(frozen=True)
class ToolUseBlock:
    """Tool use content block."""
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass(frozen=True)
class ToolResultBlock:
    """Tool result content block."""
    tool_use_id: str
    content: list[dict[str, Any]]
    type: str = "tool_result"


@dataclass(frozen=True)
class Message:
    """Base message model."""
    role: str
    content: str | list[TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock]
    name: str | None = None
    
    def get_content_blocks(self) -> list[TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock]:
        """Get content blocks from the message.
        
        Returns:
            List[Union[TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock]]:
                List of content blocks.
        """
        if isinstance(self.content, str):
            return [TextBlock(text=self.content)]
        return self.content
