"""LiteLLM-specific formatter implementation."""

from typing import Any

from cogent.ports.model import (
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from cogent.providers.base import FormatterBase


class LiteLLMFormatter(FormatterBase):
    """LiteLLM formatter for chat scenarios.
    
    This formatter handles the specific requirements of the LiteLLM API,
    which uses OpenAI-compatible format as the standard.
    """

    support_tools_api: bool = True
    """Whether support tools API"""

    support_vision: bool = True
    """Whether support vision data"""

    supported_blocks: list[type] = [
        TextBlock,
        ImageBlock,
        ToolUseBlock,
        ToolResultBlock,
    ]
    """The list of supported message blocks"""

    async def format(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Format messages into LiteLLM API format.
        
        Args:
            messages (List[Message]):
                The list of message objects to format.
        
        Returns:
            List[Dict[str, Any]]:
                The formatted messages as a list of dictionaries.
        """
        self.assert_list_of_messages(messages)

        formatted_messages: list[dict] = []
        for msg in messages:
            content_blocks = []
            function_calls = []
            tool_results = []

            for block in msg.get_content_blocks():
                if isinstance(block, TextBlock):
                    content_blocks.append({
                        "type": "text",
                        "text": block.text,
                    })

                elif isinstance(block, ImageBlock):
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": block.source,
                    })

                elif isinstance(block, ToolUseBlock):
                    # LiteLLM uses OpenAI-compatible function calling
                    function_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    })

                elif isinstance(block, ToolResultBlock):
                    # LiteLLM uses OpenAI-compatible tool results
                    tool_results.append({
                        "tool_call_id": block.tool_use_id,
                        "role": "tool",
                        "name": "",  # OpenAI requires a name field
                        "content": block.content,
                    })

            # Handle content blocks
            msg_litellm = {
                "role": msg.role,
            }

            if content_blocks:
                msg_litellm["content"] = content_blocks if len(content_blocks) > 1 else content_blocks[0]

            if function_calls:
                # For LiteLLM, function calls use function_call field
                msg_litellm["function_call"] = {
                    "name": function_calls[0]["name"],
                    "arguments": function_calls[0]["arguments"],
                }

            if msg.name:
                msg_litellm["name"] = msg.name

            # Add tool results as separate messages
            for tool_result in tool_results:
                formatted_messages.append(tool_result)

            # When both content and function_calls are None, skipped
            if "content" in msg_litellm or "function_call" in msg_litellm:
                formatted_messages.append(msg_litellm)

        return formatted_messages
