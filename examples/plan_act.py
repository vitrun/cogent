from __future__ import annotations

import os
from litellm import completion

from cogent import Env, ReActState, ToolUse, ToolRegistry, Agent, Result, Control, default_registry
from cogent.kernel import ModelPort, ToolPort
from cogent.kernel.ports import SinkPort


async def plan_action(state: ReActState, task: str, env: Env) -> Result[ReActState, ToolUse]:
    _ = env
    call = ToolUse(id="tool-1", name="search", args={"query": task})
    next_state = state.with_context(f"Plan: call {call.name} with query='{task}'.")
    return Result(next_state, value=call, control=Control.Continue())


async def execute_tool(
    state: ReActState, call: ToolUse, env: Env, registry: ToolRegistry | None = None
) -> Result[ReActState, str]:
    try:
        result = await env.tools.call(call.name, call.args)
        next_state = state.with_context(f"Tool Result ({call.name}): {result}")
        return Result(next_state, value=result, control=Control.Continue())
    except Exception as e:
        error_msg = f"Tool error: {e}"
        next_state = state.with_context(f"Tool Result ({call.name}): {error_msg}")
        return Result(next_state, control=Control.Error(error_msg))


async def synthesize_answer(state: ReActState, tool_output: str, env: Env) -> Result[ReActState, str]:
    # Use LLM to synthesize answer based on tool output
    prompt = f"Based on the following search results, answer the question: 'What is the capital of France?'\n\n{tool_output}"
    
    # Call model to get answer
    answer = await env.model.complete(prompt)
    
    next_state = state.with_context("Synthesized final answer using LLM.")
    return Result(next_state, value=answer, control=Control.Continue())


async def format_output(state: ReActState, answer: str, env: Env) -> Result[ReActState, str]:
    _ = env
    formatted = f"Final Report:\n{answer}"
    next_state = state.with_context("Formatted response for delivery.")
    return Result(next_state, value=formatted, control=Control.Continue())


class LiteLLMModel(ModelPort):
    """LiteLLM-based model implementation."""

    def __init__(self, model_name: str = "anthropic/claude-sonnet-4-20250514"):
        """Initialize LiteLLMModel."""
        self.model_name = model_name
        
    async def complete(self, prompt: str) -> str:
        """Complete a prompt using LiteLLM."""
        # Set environment variables if provided
        if os.environ.get("ANTHROPIC_BASE_URL"):
            os.environ["ANTHROPIC_BASE_URL"] = os.environ["ANTHROPIC_BASE_URL"]
        
        # Make completion request
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract the content from response
        return response.choices[0].message.content
    
    async def stream_complete(self, prompt: str, ctx: SinkPort) -> str:
        """Stream complete a prompt using LiteLLM."""
        # Set environment variables if provided
        if os.environ.get("ANTHROPIC_BASE_URL"):
            os.environ["ANTHROPIC_BASE_URL"] = os.environ["ANTHROPIC_BASE_URL"]
        
        # Make streaming completion request
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        # Process streaming response
        full_content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content
                if content:
                    await ctx.send(content)
                    full_content += content
        
        await ctx.close()
        return full_content


class SimpleTools(ToolPort):
    """Simple tool implementation."""

    async def call(self, name: str, args: dict[str, object]) -> object:
        """Call a tool by name."""
        if name == "search":
            query = args.get("query", "")
            # For demonstration purposes, return a mock search result
            return f"Search results for: {query}\n1. Result 1\n2. Result 2\n3. Result 3"
        else:
            raise ValueError(f"Tool not found: {name}")


def make_litellm_env() -> Env:
    """Create a real environment using LiteLLM."""
    return Env(
        model=LiteLLMModel(),
        tools=SimpleTools(),
    )


async def run_simple_agent(task: str, env: Env) -> Result[ReActState, str]:
    initial_state = ReActState()
    return await (
        Agent.start(initial_state, task)
        .then(lambda s, _v, inner_env: plan_action(s, task, inner_env))
        .then(lambda s, call, inner_env: execute_tool(s, call, inner_env))
        .then(synthesize_answer)
        .then(format_output)
    ).run(env)


if __name__ == "__main__":
    import asyncio

    # Set environment variables if not already set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY environment variable not set")
    
    res = asyncio.run(run_simple_agent("What is the capital of France?", make_litellm_env()))
    print(res.value)