#!/usr/bin/env python3
"""Multi-agent example with real LLM calls.

This example demonstrates four agents working together:
- researcher: Breaks down a topic into sub-topics
- fetcher: Gathers information on each sub-topic
- synthesizer: Combines findings into a summary
- reviewer: Reviews and improves the final output

Requirements:
- ANTHROPIC_API_KEY environment variable
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Any

# Suppress LiteLLM warnings
logging.getLogger("litellm").setLevel(logging.ERROR)

from litellm import completion

from cogent.kernel import Agent, Control, Result, ModelPort
from cogent.combinators import AgentRegistry, MultiEnv, MultiState, concurrent
from cogent.combinators.ops import handoff


# ==================== Model Provider ====================

class LiteLLMModel(ModelPort):
    """LiteLLM-based model implementation."""

    def __init__(self, model_name: str = "anthropic/claude-sonnet-4.6"):
        self.model_name = model_name

    async def complete(self, prompt: str) -> str:
        """Complete a prompt using LiteLLM."""
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def stream_complete(self, prompt: str, ctx: Any) -> str:
        """Stream complete - not used in this example."""
        return await self.complete(prompt)


# ==================== Real Agents ====================

def create_researcher() -> Agent[MultiState, str]:
    """Agent that breaks down a research topic into sub-topics."""

    async def _run(state: MultiState, env: MultiEnv) -> Result[MultiState, str]:
        ms = state
        task = ms.shared[-1] if ms.shared else ""

        prompt = f"""You are a research planner. Break down the following topic into 2-3 specific sub-topics to research.

Topic: {task}

Respond with just a list of sub-topics, one per line. Keep it brief."""

        response = await env.model.complete(prompt)

        new_state = MultiState(
            current="researcher",
            shared=ms.shared + (f"Sub-topics:\n{response}",),
            locals=ms.locals,
        )
        return Result(state=new_state, value=response, control=Control.Continue())

    return Agent(_run)  # type: ignore[arg-type]


def create_fetcher() -> Agent[MultiState, str]:
    """Agent that fetches/gathers information on a topic."""

    async def _run(state: MultiState, env: MultiEnv) -> Result[MultiState, str]:
        ms = state
        topic = ms.shared[-1] if ms.shared else ""

        prompt = f"""You are a research assistant. Provide a brief, factual summary (2-3 sentences) about:

{topic}

Be accurate and concise."""

        response = await env.model.complete(prompt)

        new_state = MultiState(
            current="fetcher",
            shared=ms.shared + (f"Findings on '{topic}':\n{response}",),
            locals=ms.locals,
        )
        return Result(state=new_state, value=response, control=Control.Continue())

    return Agent(_run)  # type: ignore[arg-type]


def create_synthesizer() -> Agent[MultiState, str]:
    """Agent that synthesizes findings into a summary."""

    async def _run(state: MultiState, env: MultiEnv) -> Result[MultiState, str]:
        ms = state
        # Get all findings from shared
        findings = [s for s in ms.shared if s.startswith("Findings on")]

        prompt = f"""You are a technical writer. Synthesize the following findings into a coherent summary:

{chr(10).join(findings)}

Write a brief, unified summary (2-3 paragraphs)."""

        response = await env.model.complete(prompt)

        new_state = MultiState(
            current="synthesizer",
            shared=ms.shared + (f"Summary:\n{response}",),
            locals=ms.locals,
        )
        return Result(state=new_state, value=response, control=Control.Continue())

    return Agent(_run)  # type: ignore[arg-type]


def create_reviewer() -> Agent[MultiState, str]:
    """Agent that reviews and improves the output."""

    async def _run(state: MultiState, env: MultiEnv) -> Result[MultiState, str]:
        ms = state
        summary = [s for s in ms.shared if s.startswith("Summary:")]

        if not summary:
            # No summary yet, just pass through
            new_state = MultiState(
                current="reviewer",
                shared=ms.shared,
                locals=ms.locals,
            )
            return Result(state=new_state, value="No summary to review", control=Control.Continue())

        prompt = f"""You are an editor. Review and improve the following text for clarity and accuracy:

{summary[0]}

Provide the improved version. If it's good as-is, just return it unchanged."""

        response = await env.model.complete(prompt)

        new_state = MultiState(
            current="reviewer",
            shared=ms.shared + (f"Final:\n{response}",),
            locals=ms.locals,
        )
        return Result(state=new_state, value=response, control=Control.Continue())

    return Agent(_run)  # type: ignore[arg-type]


# ==================== Helpers ====================

def merge_states(states: list[MultiState]) -> MultiState:
    """Merge multiple states into one."""
    all_shared: tuple = ()
    for s in states:
        all_shared = all_shared + s.shared
    return MultiState(
        current="merged",
        shared=all_shared,
        locals={},
    )


# ==================== Main ====================

async def main():
    """Run the multi-agent research pipeline."""

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        print("Please set it in your environment:")
        print("  export ANTHROPIC_API_KEY=your_key")
        return

    # Create model
    model = LiteLLMModel(model_name="anthropic/claude-sonnet-4.6")

    # Create registry with agents
    registry = AgentRegistry({
        "researcher": create_researcher(),
        "fetcher": create_fetcher(),
        "synthesizer": create_synthesizer(),
        "reviewer": create_reviewer(),
    })

    # Initial state with research topic
    topic = "What are the key benefits of Python async/await for web development?"
    initial_state = MultiState(
        current="",
        shared=(topic,),
        locals={},
    )

    # Create MultiEnv
    env = MultiEnv(
        model=model,
        registry=registry,
    )

    print("=" * 60)
    print("Multi-Agent Research Pipeline")
    print("=" * 60)
    print(f"Topic: {topic}")
    print()

    # Step 1: Research - break down into sub-topics
    print("Step 1: Researcher breaking down topic...")
    r1 = await handoff("researcher").run(initial_state, env)
    print(f"  → {r1.value[:100]}...")
    print()

    # Step 2: Fetch - get info on sub-topics (simplified: just use original topic)
    print("Step 2: Fetcher gathering information...")
    r2 = await handoff("fetcher").run(r1.state, env)
    print(f"  → {r2.value[:100]}...")
    print()

    # Step 3: Synthesize
    print("Step 3: Synthesizer creating summary...")
    r3 = await handoff("synthesizer").run(r2.state, env)
    print(f"  → {r3.value[:100]}...")
    print()

    # Step 4: Review
    print("Step 4: Reviewer improving output...")
    r4 = await handoff("reviewer").run(r3.state, env)
    print(f"  → {r4.value[:100]}...")
    print()

    # Show final result
    print("=" * 60)
    print("Final Output:")
    print("=" * 60)
    print(r4.value)
    print()

    # Show all shared messages
    print("=" * 60)
    print("All Shared Messages:")
    print("=" * 60)
    for i, msg in enumerate(r4.state.shared):
        print(f"{i+1}. {msg[:80]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
