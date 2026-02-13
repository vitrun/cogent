import asyncio
from cogent.model import (
    Message,
    TextBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
)
from cogent.providers.litellm import LiteLLMFormatter


class TestLiteLLMFormatter:
    """Test LiteLLM formatter functionality."""
    
    def test_format_text_messages(self):
        """Test formatting text messages."""
        async def run_test():
            formatter = LiteLLMFormatter()
            msgs = [
                Message(role="user", content="Hello, how are you?"),
                Message(role="assistant", content="I'm fine, thank you!"),
            ]
            
            result = await formatter.format(msgs)
            assert len(result) == 2
            assert result[0]["role"] == "user"
            assert result[0]["content"]["type"] == "text"
            assert result[0]["content"]["text"] == "Hello, how are you?"
            assert result[1]["role"] == "assistant"
            assert result[1]["content"]["type"] == "text"
            assert result[1]["content"]["text"] == "I'm fine, thank you!"
        
        asyncio.run(run_test())
    
    def test_format_system_message(self):
        """Test formatting system message."""
        async def run_test():
            formatter = LiteLLMFormatter()
            msgs = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello!"),
            ]
            
            result = await formatter.format(msgs)
            assert len(result) == 2
            assert result[0]["role"] == "system"
            assert result[0]["content"]["type"] == "text"
            assert result[0]["content"]["text"] == "You are a helpful assistant."
            assert result[1]["role"] == "user"
            assert result[1]["content"]["type"] == "text"
            assert result[1]["content"]["text"] == "Hello!"
        
        asyncio.run(run_test())
    
    def test_format_image_message(self):
        """Test formatting image message."""
        async def run_test():
            formatter = LiteLLMFormatter()
            msgs = [
                Message(
                    role="user",
                    content=[
                        TextBlock(text="What's in this image?"),
                        ImageBlock(source={"url": "https://example.com/image.jpg"}),
                    ],
                ),
            ]
            
            result = await formatter.format(msgs)
            assert len(result) == 1
            assert result[0]["role"] == "user"
            assert len(result[0]["content"]) == 2
            assert result[0]["content"][0]["type"] == "text"
            assert result[0]["content"][0]["text"] == "What's in this image?"
            assert result[0]["content"][1]["type"] == "image_url"
            assert result[0]["content"][1]["image_url"] == {"url": "https://example.com/image.jpg"}
        
        asyncio.run(run_test())
    
    def test_format_tool_use_message(self):
        """Test formatting tool use message."""
        async def run_test():
            formatter = LiteLLMFormatter()
            msgs = [
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="tool-1",
                            name="search",
                            input={"query": "Python programming"},
                        ),
                    ],
                ),
            ]
            
            result = await formatter.format(msgs)
            assert len(result) == 1
            assert result[0]["role"] == "assistant"
            assert "function_call" in result[0]
            assert result[0]["function_call"]["name"] == "search"
            assert result[0]["function_call"]["arguments"] == {"query": "Python programming"}
        
        asyncio.run(run_test())
    
    def test_format_tool_result_message(self):
        """Test formatting tool result message."""
        async def run_test():
            formatter = LiteLLMFormatter()
            msgs = [
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id="tool-1",
                            content=[{"type": "text", "text": "Python is a programming language"}],
                        ),
                    ],
                ),
            ]
            
            result = await formatter.format(msgs)
            assert len(result) == 1
            assert result[0]["role"] == "tool"
            assert result[0]["tool_call_id"] == "tool-1"
            assert result[0]["content"] == [{"type": "text", "text": "Python is a programming language"}]
        
        asyncio.run(run_test())
