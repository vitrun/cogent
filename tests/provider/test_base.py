import asyncio
from cogent.model import (
    Message,
    TextBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
)
from cogent.providers import FormatterBase


class TestMessageModel:
    """Test message model functionality."""
    
    def test_text_message(self):
        """Test text message creation and content blocks."""
        msg = Message(role="user", content="Hello, how are you?")
        blocks = msg.get_content_blocks()
        assert len(blocks) == 1
        assert blocks[0].type == "text"
        assert blocks[0].text == "Hello, how are you?"
    
    def test_multiple_blocks_message(self):
        """Test message with multiple content blocks."""
        blocks = [
            TextBlock(text="Hello"),
            ImageBlock(source={"url": "https://example.com/image.jpg"}),
        ]
        msg = Message(role="user", content=blocks)
        result_blocks = msg.get_content_blocks()
        assert len(result_blocks) == 2
        assert result_blocks[0].type == "text"
        assert result_blocks[1].type == "image"
    
    def test_tool_use_block(self):
        """Test tool use block creation."""
        block = ToolUseBlock(
            id="tool-1",
            name="search",
            input={"query": "Python programming"},
        )
        assert block.type == "tool_use"
        assert block.id == "tool-1"
        assert block.name == "search"
        assert block.input == {"query": "Python programming"}
    
    def test_tool_result_block(self):
        """Test tool result block creation."""
        block = ToolResultBlock(
            tool_use_id="tool-1",
            content=[{"type": "text", "text": "Python is a programming language"}],
        )
        assert block.type == "tool_result"
        assert block.tool_use_id == "tool-1"
        assert len(block.content) == 1
        assert block.content[0]["text"] == "Python is a programming language"


class TestFormatterBase:
    """Test formatter base class functionality."""
    
    def test_assert_list_of_messages_valid(self):
        """Test assert_list_of_messages with valid input."""
        # Create a concrete subclass of FormatterBase for testing
        class TestFormatter(FormatterBase):
            async def format(self, messages):
                return []
        
        formatter = TestFormatter()
        msgs = [Message(role="user", content="Hello")]
        # Should not raise an exception
        formatter.assert_list_of_messages(msgs)
    
    def test_assert_list_of_messages_invalid_type(self):
        """Test assert_list_of_messages with invalid input type."""
        # Create a concrete subclass of FormatterBase for testing
        class TestFormatter(FormatterBase):
            async def format(self, messages):
                return []
        
        formatter = TestFormatter()
        invalid_input = "not a list"
        try:
            formatter.assert_list_of_messages(invalid_input)
            assert False, "Should have raised TypeError"
        except TypeError:
            pass
    
    def test_assert_list_of_messages_invalid_item(self):
        """Test assert_list_of_messages with invalid item type."""
        # Create a concrete subclass of FormatterBase for testing
        class TestFormatter(FormatterBase):
            async def format(self, messages):
                return []
        
        formatter = TestFormatter()
        invalid_input = ["not a message"]
        try:
            formatter.assert_list_of_messages(invalid_input)
            assert False, "Should have raised TypeError"
        except TypeError:
            pass
