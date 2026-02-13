"""Formatter steps for Cogent."""

from typing import Any, TypeVar

from ..core import Control, Env, Result
from ..provider import Message
from ..provider.litellm import LiteLLMFormatter
from .protocols import ReActStateProtocol

S = TypeVar("S")


async def format_messages_step(
    state: ReActStateProtocol[S],
    msgs: list[Message],
    env: Env,
) -> Result[S, list[dict[str, Any]]]:
    """Step that formats messages into LiteLLM API format.
    
    Args:
        state: The current state
        msgs: List of messages to format
        env: The environment
        
    Returns:
        Result with new state containing formatted messages and the formatted messages as value
    """
    try:
        # Create formatter instance
        formatter = LiteLLMFormatter()
        
        # Format messages
        formatted_messages = await formatter.format(msgs)
        
        # Create new state with formatted messages
        next_state = state.with_formatted_messages(formatted_messages)
        next_state = next_state.with_evidence("format_messages", info={"message_count": len(formatted_messages)})
        
        return Result(next_state, control=Control.Continue(formatted_messages))
        
    except Exception as e:
        # Handle formatting errors
        next_state = state.with_evidence("format_messages", info={"error": str(e)})
        return Result(next_state, control=Control.Error(f"Failed to format messages: {e}"))


async def build_prompt_step(
    state: ReActStateProtocol[S],
    _: str,
    env: Env,
) -> Result[S, str]:
    """Step that builds prompt for models.
    
    Args:
        state: The current state
        _: Placeholder value
        env: The environment
        
    Returns:
        Result with new state and the built prompt
    """
    # This step assumes that formatted_messages is already set
    # In a real implementation, you would collect messages and format them
    # For now, we'll keep it simple
    prompt = "You are a helpful assistant."
    next_state = state.with_evidence("build_prompt", info={})
    return Result(next_state, control=Control.Continue(prompt))
