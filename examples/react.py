#!/usr/bin/env python3
"""
ReAct Agent example using real tools and LiteLLM model provider.

This agent can:
- Search the web (DuckDuckGo HTML)
- Fetch URL content
- Calculate mathematical expressions

Requirements:
- OPENROUTER_API_KEY or ANTHROPIC_API_KEY environment variable
"""

from __future__ import annotations

import os
import asyncio
import urllib.request
import urllib.parse
import re
import ast
import argparse
import logging

# Suppress LiteLLM warnings
logging.getLogger("litellm").setLevel(logging.ERROR)

from cogent.agents.react import ReactAgent, ReActState
from cogent.kernel import ToolPort, ToolCall
from cogent.kernel.result import Result


class SimpleTools(ToolPort[ReActState]):
    """Tool implementation for search, get_url, and calculate."""

    async def call(self, state: ReActState, call: ToolCall) -> Result[ReActState, str]:
        name = call.name
        args = call.args

        if name == "search":
            result = _search(args.get("query", ""))
        elif name == "get_url":
            result = _get_url(args.get("url", ""))
        elif name == "calculate":
            result = _calculate(args.get("expression", ""))
        else:
            result = f"Error: Unknown tool '{name}'"

        return Result(state, value=result)


# ==================== Tool Implementations ====================

def _search(query: str) -> str:
    """Search using DuckDuckGo HTML (no API key needed)."""
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CogentAgent/1.0 (Research Assistant)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8")

        # Extract results from DuckDuckGo HTML
        results = []
        for match in re.finditer(r'<a class="result__a"[^>]*>([^<]+)</a>', html):
            title = match.group(1).strip()
            # Decode HTML entities
            title = title.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            title = title.replace("&quot;", '"').replace("&#39;", "'")
            if title and len(title) > 3:
                results.append(title)
                if len(results) >= 5:
                    break

        if results:
            return "Search Results for: " + query + "\n" + "\n".join(
                f"{i+1}. {r}" for i, r in enumerate(results)
            )
        else:
            return f"No results found for: {query}"

    except Exception as e:
        return f"Search error: {e}"


def _get_url(url: str) -> str:
    """Fetch content from a URL."""
    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        return f"Error: Invalid URL protocol. URL must start with http:// or https://"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CogentAgent/1.0 (Research Assistant)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8")

        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)

        # Extract text content
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Decode HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")

        # Limit to reasonable length
        if len(text) > 2000:
            text = text[:2000] + "\n... (truncated)"

        return text

    except Exception as e:
        return f"Fetch error: {e}"


def _calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        # Only allow safe mathematical operations
        parsed = ast.parse(expression, mode="eval")
        # Check that the expression only contains allowed operations
        allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                       ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                       ast.Mod, ast.USub, ast.UAdd)
        for node in ast.walk(parsed):
            if not isinstance(node, allowed_nodes):
                return f"Error: Invalid expression"

        result = eval(compile(parsed, "<string>", "eval"))
        return str(result)

    except SyntaxError:
        return f"Error: Invalid expression syntax"
    except ZeroDivisionError:
        return f"Error: Division by zero"
    except Exception as e:
        return f"Error: {e}"

# ==================== Main ====================

async def run_task(task: str) -> str:
    """Run a single research task.

    Args:
        task: The research task to perform

    Returns:
        The final answer from the agent
    """
    model_name = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6")

    agent = ReactAgent(
        model=model_name,
        tools=SimpleTools(),
        max_steps=10,
    )

    result = await agent.run(task)
    return result.value


async def main():
    """Run the ReAct research assistant."""
    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: OPENROUTER_API_KEY or ANTHROPIC_API_KEY environment variable not set")
        print("Please set one of these to run the agent:")
        print("  export OPENROUTER_API_KEY=your_key")
        print("  export ANTHROPIC_API_KEY=your_key")
        return

    parser = argparse.ArgumentParser(
        description="ReAct Research Assistant with web search capabilities"
    )
    parser.add_argument(
        "task", nargs="?", type=str,
        help="Research task to perform (optional, for interactive mode)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ReAct Research Assistant")
    print("=" * 60)
    print("Available tools: search, get_url, calculate")
    print("Type 'quit' to exit")
    print()

    if args.task:
        # Single task mode
        print(f"Task: {args.task}")
        print("-" * 60)
        result = await run_task(args.task)
        print(f"\nFinal Answer:\n{result}")
    else:
        # Interactive mode
        while True:
            task = input("Research task: ").strip()

            if task.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not task:
                continue

            print("\n" + "-" * 60)
            result = await run_task(task)
            print(f"\nFinal Answer:\n{result}")
            print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
