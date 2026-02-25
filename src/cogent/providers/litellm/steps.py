"""LiteLLM-specific formatter steps for agents."""

from typing import Any

from cogent.kernel.env import Env
from cogent.kernel.result import Control, Result
from cogent.model import Message
from cogent.providers.litellm import LiteLLMFormatter


async def format_messages_step(
    state: Any,
    msgs: list[Message],
    env: Env,
) -> Result[Any, list[dict[str, Any]]]:
    """Step that formats messages into LiteLLM API format.

    Args:
        state: The current state
        msgs: List of messages to format
        env: The environment

    Returns:
        Result with new state containing formatted messages and the formatted messages as value
    """
    try:
        formatter = LiteLLMFormatter()
        formatted_messages = await formatter.format(msgs)

        # Try to use with_formatted_messages if available, otherwise just return
        if hasattr(state, "with_formatted_messages"):
            next_state = state.with_formatted_messages(formatted_messages)
            if hasattr(state, "with_evidence"):
                next_state = next_state.with_evidence(
                    "format_messages", info={"message_count": len(formatted_messages)}
                )
            return Result(next_state, value=formatted_messages, control=Control.Continue())

        return Result(state, value=formatted_messages, control=Control.Continue())

    except Exception as e:
        if hasattr(state, "with_evidence"):
            next_state = state.with_evidence("format_messages", info={"error": str(e)})
            return Result(next_state, control=Control.Error(f"Failed to format messages: {e}"))
        return Result(state, control=Control.Error(f"Failed to format messages: {e}"))


async def build_prompt_step(
    state: Any,
    _: str,
    env: Env,
) -> Result[Any, str]:
    """Step that builds prompt for models."""
    prompt = "You are a helpful assistant."
    if hasattr(state, "with_evidence"):
        next_state = state.with_evidence("build_prompt", info={})
        return Result(next_state, value=prompt, control=Control.Continue())
    return Result(state, value=prompt, control=Control.Continue())
