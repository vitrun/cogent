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

from litellm import completion

from cogent import Env, ReActState
from cogent.starter.react import ReActPolicy, ReActConfig
from cogent.core import ModelPort, ToolPort
from cogent.core.env import RuntimeContext


# ==================== Model Provider ====================

class LiteLLMModel(ModelPort):
    """LiteLLM-based model implementation."""

    def __init__(self, model_name: str = "anthropic/claude-sonnet-4-20250514"):
        self.model_name = model_name

    async def complete(self, prompt: str) -> str:
        """Complete a prompt using LiteLLM."""
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def stream_complete(self, prompt: str, ctx: RuntimeContext) -> str:
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
                    await ctx.emit(content)
                    full_content += content

        await ctx.close()
        return full_content


# ==================== Real Tools ====================

class WebTools(ToolPort):
    """Real web search and fetch tools (no API key required)."""

    async def call(self, name: str, args: dict[str, object]) -> object:
        """Call a tool by name."""
        if name == "search":
            query = str(args.get("query", ""))
            return self._search(query)
        elif name == "get_url":
            url = str(args.get("url", ""))
            return self._get_url(url)
        elif name == "calculate":
            expr = str(args.get("expression", ""))
            return self._calculate(expr)
        else:
            raise ValueError(f"Tool not found: {name}")

    def _search(self, query: str) -> str:
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

    def _get_url(self, url: str) -> str:
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

    def _calculate(self, expression: str) -> str:
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


# ==================== Environment ====================

def make_env() -> Env:
    """Create environment with LiteLLM model and web tools."""
    model_name = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-20250514")
    return Env(
        model=LiteLLMModel(model_name=model_name),
        tools=WebTools(),
    )


# ==================== Main ====================

async def run_task(task: str) -> str:
    """Run a single research task.

    Args:
        task: The research task to perform

    Returns:
        The final answer from the agent
    """
    # Create environment
    env = make_env()

    # Create policy and agent
    config = ReActConfig(max_steps=10)
    policy = ReActPolicy(config)

    # Run the agent
    initial_state = ReActState()
    result = await policy.run(initial_state, task).run(env)

    return result.control.value


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
