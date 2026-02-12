#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat example using Anthropic model with streaming output and memory support.
"""

import os
import asyncio
from litellm import completion

from cogent import Env, ReActState, Agent, Result, Control
from cogent.core import ModelPort
from cogent.core.env import RuntimeContext
from cogent.core.memory import Memory, SimpleMemory


class LiteLLMModel(ModelPort):
    """LiteLLM-based model implementation for Anthropic."""

    def __init__(self, model_name: str = "anthropic/claude-sonnet-4-20250514"):
        """Initialize LiteLLMModel.
        
        Args:
            model_name: The model name to use for completion.
        """
        self.model_name = model_name
        
    async def complete(self, prompt: str) -> str:
        """Complete a prompt using LiteLLM.
        
        Args:
            prompt: The prompt to complete.
            
        Returns:
            The completed text.
        """
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    async def stream_complete(self, prompt: str, ctx: RuntimeContext) -> str:
        """Stream complete a prompt using LiteLLM.
        
        Args:
            prompt: The prompt to complete.
            ctx: The runtime context to emit chunks to.
            
        Returns:
            The completed text.
        """
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        full_content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content
                if content:
                    await ctx.emit(content)
                    full_content += content
        
        await ctx.close()
        return full_content


class TTYRuntimeContext(RuntimeContext):
    """Runtime context for TTY streaming output."""

    async def emit(self, chunk: str) -> None:
        """Emit a chunk of text to the TTY.
        
        Args:
            chunk: The chunk of text to emit.
        """
        print(chunk, end="", flush=True)
    
    async def close(self) -> None:
        """Close the context."""
        print()


def make_chat_env() -> Env:
    """Create a chat environment using LiteLLM.
    
    Returns:
        The created environment.
    """
    return Env(
        model=LiteLLMModel(),
        runtime_context=TTYRuntimeContext()
    )


async def process_message(state: ReActState, message: str, env: Env) -> Result[ReActState, str]:
    """Process a single message using the agent."""
    # Build prompt with history
    history_entries = state.history.snapshot()
    history_str = "\n".join(history_entries)
    
    prompt = f"{history_str}\n\nUser: {message}\nAssistant: "
    
    # Call model with streaming
    if env.runtime_context:
        response = await env.model.stream_complete(prompt, env.runtime_context)
    else:
        response = await env.model.complete(prompt)
    
    # Update history
    new_history = state.history.append(f"User: {message}")
    new_history = new_history.append(f"Assistant: {response}")
    new_state = state.with_history(f"Assistant: {response}")
    
    return Result(new_state, control=Control.Continue(response))


async def chat():
    """Run the chat loop."""
    print("Chat with Anthropic model. Type 'exit' to quit.")
    print("=" * 50)
    
    # Create environment
    env = make_chat_env()
    
    # Initialize state with memory
    initial_state = ReActState()
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            
            # Create agent and run
            print("Assistant: ", end="", flush=True)
            
            agent = Agent.start(initial_state, user_input).then(process_message)
            result = await agent.run(env)
            
            # Update initial state for next iteration
            initial_state = result.state
            
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    # Check environment variables
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY environment variable not set")
    
    # Run chat
    asyncio.run(chat())
