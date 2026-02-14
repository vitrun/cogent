"""ReAct policy implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from cogent.kernel.agent import Agent
from cogent.kernel.env import Env
from cogent.kernel.result import Control, Result
from cogent.kernel.tool import ToolResult, ToolUse

from .state import ReActStateProtocol

S = TypeVar("S")
T = TypeVar("T")


class ReActOutput(BaseModel):
    """Default ReAct output schema - can be overridden."""
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    final: str | None = None


@dataclass(frozen=True)
class ReActConfig:
    """Configuration for ReAct policy."""
    pass


def _append_scratchpad(state: ReActStateProtocol[S], line: str) -> S:
    """Helper function to append a line to the scratchpad."""
    if state.scratchpad:
        return state.with_scratchpad(f"{state.scratchpad}\n{line}")
    return state.with_scratchpad(line)


def _clean_json_output(text: str) -> str:
    """Clean markdown code blocks from JSON output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 2:
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
    return text


def structured(
    schema: type[T] = ReActOutput,
) -> Any:
    """Create a Step that parses and validates LLM output against a schema.

    Args:
        schema: Pydantic model to parse output into

    Returns:
        A Step function that parses raw string output
    """
    async def parse_step(
        state: S,
        model_output: str,
        env: Env,
    ) -> Result[S, T]:
        cleaned = _clean_json_output(model_output)
        try:
            parsed = schema.model_validate_json(cleaned)
            return Result(state=state, value=parsed, control=Control.Continue())
        except ValidationError as exc:
            return Result(state=state, control=Control.Error(exc))

    return parse_step


class ReActPolicy(Generic[S]):
    """ReAct policy that constructs the pipeline, independent of provider."""

    def __init__(self, config: ReActConfig):
        self.config = config

    async def prompt(self, state: S, task: str, env: Env) -> Result[S, str]:
        """Prompt step that formats context and scratchpad into a prompt."""
        context_entries = state.context.snapshot()
        context_block = "\n".join(context_entries)

        task_context = f"\nTask: {task}" if task else ""

        prompt = (
            "You are a ReAct agent. Respond only with valid JSON using keys "
            '"thought", "action", "action_input", "final". '
            "Set exactly one of action or final.\n"
            '"action_input" must be an object/dictionary, not a string.\n\n'
            f"History:\n{context_block}\n\n"
            f"Scratchpad:\n{state.scratchpad}{task_context}\n"
        )
        next_state = state.with_evidence("prompt", info={})
        return Result(next_state, value=prompt, control=Control.Continue())

    async def think(self, state: S, prompt: str, env: Env) -> Result[S, str]:
        """Think step that calls the model to generate a response."""
        if env.sink:
            response = await env.model.stream_complete(prompt, env.sink)
        else:
            response = await env.model.complete(prompt)
        next_state = state.with_evidence("think")
        return Result(next_state, value=response, control=Control.Continue())

    async def decide(
        self,
        state: S,
        parsed: ReActOutput,
        env: Env,
    ) -> Result[S, ToolUse | str]:
        """Decide step that interprets parsed model output."""
        next_state = state
        thought_line = f"Thought: {parsed.thought}"
        next_state = next_state.with_context(thought_line)
        next_state = _append_scratchpad(next_state, thought_line)

        action_input = parsed.action_input or {}
        action_line = ""
        if parsed.action:
            action_line = f"Action: {parsed.action} {action_input}"
            next_state = next_state.with_context(action_line)
            next_state = _append_scratchpad(next_state, action_line)

        next_state = next_state.with_evidence(
            "decide",
            info={"action": parsed.action, "final": parsed.final is not None},
        )

        if parsed.final:
            return Result(next_state, value=parsed.final, control=Control.Halt())

        if parsed.action:
            tool_call = ToolUse(
                id=f"tool-{int(time.time() * 1000)}",
                name=parsed.action,
                args=action_input,
            )
            return Result(next_state, value=tool_call, control=Control.Continue())

        return Result(next_state, control=Control.Error("Missing action or final"))

    async def act(self, state: S, call: ToolUse | str, env: Env) -> Result[S, ToolResult]:
        """Act step that executes a tool call."""
        if not isinstance(call, ToolUse):
            return Result(state, control=Control.Error("No tool call to execute"))
        next_state = state.with_evidence("tool_call", info={"name": call.name, "args": call.args})
        try:
            content = await env.tools.call(call.name, call.args)
            result = ToolResult.success(call.id, str(content))
            return Result(next_state, value=result, control=Control.Continue())
        except Exception as e:
            result = ToolResult.failure(call.id, str(e))
            return Result(next_state, control=Control.Error(f"Tool execution failed: {e}"))

    async def observe(self, state: S, result: ToolResult, env: Env) -> Result[S, str]:
        """Observe step that processes tool execution results."""
        observation = f"Observation: {result.content}"
        next_state = state.with_context(observation)
        next_state = self._append_scratchpad(next_state, observation)
        next_state = next_state.with_evidence("observation", info={"tool_result_id": result.id})
        return Result(next_state, value=observation, control=Control.Continue())

    def _append_scratchpad(self, state: S, line: str) -> S:
        """Helper method to append a line to the scratchpad."""
        if state.scratchpad:
            return state.with_scratchpad(f"{state.scratchpad}\n{line}")
        return state.with_scratchpad(line)

    def build(
        self,
        task: str,
        schema: type[ReActOutput] = ReActOutput,
    ) -> Agent[S, str]:
        """Build one ReAct round: prompt → think → structured → decide → act → observe.

        Returns an Agent that executes a single iteration of the ReAct loop.
        Use repeat() to execute multiple rounds.

        Args:
            task: The initial task string
            schema: Pydantic model for parsing LLM output (default: ReActOutput)

        Note: State flows from agent.run(state, env), not from build.
        """
        return (
            Agent.lift_value(task)
            .then(self.prompt)
            .then(self.think)
            .then(structured(schema))
            .then(self.decide)
            .then(self.act)
            .then(self.observe)
        )
