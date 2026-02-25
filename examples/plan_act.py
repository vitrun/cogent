from __future__ import annotations

import os
import logging
from typing import Any

# Suppress LiteLLM warnings
logging.getLogger("litellm").setLevel(logging.ERROR)

from litellm import completion

from cogent import Env, ReActState, Agent, Result, Control
from cogent.kernel import ModelPort, ToolPort, ToolCall
from cogent.kernel.ports import SinkPort
from cogent.kernel.result import Result as KernelResult


async def plan_action(state: ReActState, task: str, env: Env) -> KernelResult[ReActState, ToolCall]:
    _ = env
    call = ToolCall(name="search", args={"query": task})
    next_state = state.with_context(f"Plan: call {call.name} with query='{task}'.")
    return KernelResult(state=next_state, value=call, control=Control.Continue())


async def execute_tool(
    state: ReActState, call: ToolCall, env: Env
) -> KernelResult[ReActState, str]:
    try:
        result = await env.tools.call(state, call)
        next_state = state.with_context(f"Tool Result ({call.name}): {result.value}")
        return KernelResult(next_state, value=result.value, control=Control.Continue())
    except Exception as e:
        error_msg = f"Tool error: {e}"
        next_state = state.with_context(f"Tool Result ({call.name}): {error_msg}")
        return KernelResult(next_state, control=Control.Error(error_msg))


async def synthesize_answer(state: ReActState, tool_output: str, env: Env) -> KernelResult[ReActState, str]:
    # Use LLM to synthesize answer based on tool output
    prompt = f"Based on the following search results, answer the question: 'What is the capital of France?'\n\n{tool_output}"

    # Call model to get answer
    answer = await env.model.complete(prompt)

    next_state = state.with_context("Synthesized final answer using LLM.")
    return KernelResult(next_state, value=answer, control=Control.Continue())


async def format_output(state: ReActState, answer: str, env: Env) -> KernelResult[ReActState, str]:
    _ = env
    formatted = f"Final Report:\n{answer}"
    next_state = state.with_context("Formatted response for delivery.")
    return KernelResult(next_state, value=formatted, control=Control.Continue())


class LiteLLMModel(ModelPort):
    """LiteLLM-based model implementation."""

    def __init__(self, model_name: str = "anthropic/claude-sonnet-4-20250514"):
        """Initialize LiteLLMModel."""
        self.model_name = model_name

    async def complete(self, prompt: str) -> str:
        """Complete a prompt using LiteLLM."""
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def stream_complete(self, prompt: str, ctx: SinkPort) -> str:
        """Stream complete a prompt using LiteLLM."""
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
                    await ctx.send(content)
                    full_content += content

        await ctx.close()
        return full_content


class SimpleTools(ToolPort[ReActState]):
    """Simple tool implementation."""

    async def call(self, state: ReActState, call: ToolCall) -> KernelResult[ReActState, str]:
        """Call a tool by name."""
        if call.name == "search":
            query = call.args.get("query", "")
            result = f"Search results for: {query}\n1. Paris\n2. Lyon\n3. Marseille"
            return KernelResult(state, value=result)
        else:
            raise ValueError(f"Tool not found: {call.name}")


def make_litellm_env() -> Env:
    """Create a real environment using LiteLLM."""
    return Env(
        model=LiteLLMModel(),
        tools=SimpleTools(),
    )


async def run_simple_agent(task: str, env: Env) -> KernelResult[ReActState, str]:
    initial_state = ReActState()
    return await (
        Agent.start(task)
        .then(lambda s, _v, inner_env: plan_action(s, task, inner_env))
        .then(lambda s, call, inner_env: execute_tool(s, call, inner_env))
        .then(synthesize_answer)
        .then(format_output)
    ).run(initial_state, env)


if __name__ == "__main__":
    import asyncio

    # Set environment variables if not already set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY environment variable not set")

    res = asyncio.run(run_simple_agent("What is the capital of France?", make_litellm_env()))
    print(res.value)
