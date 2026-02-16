"""ReactAgent - OOP façade for ReAct agent."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from cogent.agents.react.state import ReActState
from cogent.combinators import repeat
from cogent.kernel import ModelPort
from cogent.kernel.env import Env
from cogent.kernel.ports import SinkPort
from cogent.kernel.trace import Trace
from cogent.runtime.tools import ToolRegistry

from .result import ReactResult

if TYPE_CHECKING:
    from typing import Self



class _ToolAdapter:
    """Adapter to make ToolRegistry compatible with ToolPort interface."""

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def call(self, name: str, args: dict[str, str]) -> str:
        """Call a tool by name with args."""
        from cogent.kernel.tool import ToolUse

        tool_use = ToolUse(id=f"tool-{id(self)}", name=name, args=args)
        result = await self._registry.run(None, None, tool_use)
        return result.content


class _StreamSink:
    """Simple async queue-based sink for streaming."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._closed = False

    async def send(self, chunk: str) -> None:
        await self._queue.put(chunk)

    async def close(self) -> None:
        await self._queue.put(None)
        self._closed = True

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> str:
        if self._closed:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item


class ReactAgent:
    """OOP façade for ReAct agent - hides kernel complexity.

    Simple, opinionated API for running ReAct-style agents.

    Example:
        agent = ReactAgent(
            model="anthropic/claude-sonnet-4-20250514",
            tools=my_tools,
        )
        result = await agent.run("What is 2+2?")
        print(result.value)
    """

    def __init__(
        self,
        model: str | ModelPort,
        tools: ToolRegistry | None = None,
        *,
        max_steps: int = 20,
        trace: bool = True,
        debug: bool = False,
    ):
        """Initialize ReactAgent.

        Args:
            model: Model name string or ModelPort instance.
            tools: Optional ToolRegistry with tool definitions and handlers.
            max_steps: Maximum ReAct iterations (default 20).
            trace: Enable trace for debugging (default True).
            debug: Enable verbose output (default False).
        """
        self._model = model
        self._tools = tools
        self._max_steps = max_steps
        self._trace_enabled = trace
        self._debug = debug

    def _build_model(self) -> ModelPort:
        """Build or wrap the model."""
        if isinstance(self._model, str):
            return _StringModelWrapper(self._model)
        return self._model

    def _build_env(self, sink: SinkPort | None = None) -> Env:
        """Build the environment."""
        trace = Trace(enabled=self._trace_enabled) if self._trace_enabled else None
        tools = _ToolAdapter(self._tools) if self._tools else None
        return Env(
            model=self._build_model(),
            tools=tools,
            trace=trace,
            sink=sink,
        )

    async def run(self, task: str) -> ReactResult:
        """Run the agent with a task.

        Args:
            task: The task/prompt for the agent.

        Returns:
            ReactResult with final value, step count, and optional trace.
        """
        from cogent.agents.react import ReActConfig, ReActPolicy

        env = self._build_env()
        config = ReActConfig()
        policy = ReActPolicy(config)
        agent = policy.build(task)
        repeated = repeat(agent, self._max_steps)

        initial_state = ReActState()
        result = await repeated.run(initial_state, env)

        # Count steps from trace if available
        steps = 0
        if env.trace:
            for evidence in env.trace._events:
                if evidence.action == "step_end":
                    steps += 1
            # If no step_end events, estimate from trace
            if steps == 0:
                steps = len([e for e in env.trace._events if e.action == "think"])

        if self._debug and env.trace:
            print(f"[DEBUG] Steps: {steps}")
            print(f"[DEBUG] Trace: {env.trace._events}")

        return ReactResult(
            value=result.value or "",
            steps=steps,
            trace=env.trace if self._trace_enabled else None,
        )

    async def stream(self, task: str) -> AsyncIterator[str]:
        """Stream the agent's output as it runs.

        Args:
            task: The task/prompt for the agent.

        Yields:
            Chunks of the agent's output as they arrive.
        """
        from cogent.agents.react import ReActConfig, ReActPolicy

        sink = _StreamSink()
        env = self._build_env(sink=sink)
        config = ReActConfig()
        policy = ReActPolicy(config)
        agent = policy.build(task)
        repeated = repeat(agent, self._max_steps)

        initial_state = ReActState()

        # Run in background, yield from sink
        async def run_and_yield():
            result = await repeated.run(initial_state, env)
            return result

        task_handle = asyncio.create_task(run_and_yield())

        try:
            async for chunk in sink:
                yield chunk
        finally:
            await task_handle


class _StringModelWrapper:
    """Wrapper to create model from string name.

    Uses environment variables to configure the model.
    Supports OpenRouter model names like "anthropic/claude-sonnet-4-20250514".
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def complete(self, prompt: str) -> str:
        from litellm import completion

        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content  # type: ignore[attr-defined]

    async def stream_complete(self, prompt: str, ctx: SinkPort) -> str:
        from litellm import completion

        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        full_content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:  # type: ignore[attr-defined]
                content = chunk.choices[0].delta.content  # type: ignore[attr-defined]
                if content:
                    await ctx.send(content)
                    full_content += content

        await ctx.close()
        return full_content
