from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Self, TypeVar

from pydantic import BaseModel, ValidationError

from ..core import Agent, Control, Env, Result, ToolResult, ToolUse
from ..core.memory import Memory, SimpleMemory
from .evidence import Evidence
from .protocols import ReActStateProtocol

S = TypeVar("S")


@dataclass(frozen=True)
class ReActState:
    """Default opinionated state that combines task information, history, and evidence for agents."""

    history: Memory = field(default_factory=SimpleMemory)
    scratchpad: str = field(default="")
    evidence: Evidence = field(default_factory=lambda: Evidence(action="start"))  # type: ignore

    def with_history(self, entry: str) -> Self:
        new_history = self.history.append(entry)
        return replace(self, history=new_history)

    def with_scratchpad(self, text: str) -> Self:
        return replace(self, scratchpad=text)

    def with_evidence(
        self,
        action: str,
        input_data: Any = None,
        output_data: Any = None,
        info: dict[str, Any] | None = None,
        **kwargs,
    ) -> Self:
        new_evidence = self.evidence.child(
            action,
            info=info or {},
        )

        # Create new state with the new evidence
        return replace(self, evidence=new_evidence)


class ReActOutput(BaseModel):
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    final: str | None = None


@dataclass(frozen=True)
class ReActConfig:
    max_steps: int = 10


def _append_scratchpad(state: ReActStateProtocol[S], line: str) -> S:
    """Helper function to append a line to the scratchpad."""
    if state.scratchpad:
        return state.with_scratchpad(f"{state.scratchpad}\n{line}")
    return state.with_scratchpad(line)


def _clean_json_output(text: str) -> str:
    """Clean markdown code blocks from JSON output."""
    # Remove ```json and ``` wrappers
    text = text.strip()
    if text.startswith("```"):
        # Find the end of the opening code fence
        lines = text.split("\n")
        # Skip the first line (code fence)
        if len(lines) >= 2:
            # Remove the first line (```json or ```)
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # Remove the last line if it's a code fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
    return text


async def react_decide(state: ReActStateProtocol[S], model_output: str, env: Env) -> Result[S, ToolUse | str]:
    """Decide step that parses model output and decides on action or final answer."""
    # Clean markdown code blocks from model output
    cleaned_output = _clean_json_output(model_output)

    try:
        parsed = ReActOutput.model_validate_json(cleaned_output)
    except ValidationError as exc:
        next_state = state.with_evidence("decide", info={"error": "invalid_json"})
        return Result(next_state, control=Control.Error(exc))

    next_state = state
    thought_line = f"Thought: {parsed.thought}"
    next_state = next_state.with_history(thought_line)
    next_state = _append_scratchpad(next_state, thought_line)

    action_input = parsed.action_input or {}
    action_line = ""
    if parsed.action:
        action_line = f"Action: {parsed.action} {action_input}"
        next_state = next_state.with_history(action_line)
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


class ReActPolicy(Generic[S]):
    """ReAct policy that orchestrates the pipeline, independent of provider.
    
    Args:
        config: ReAct configuration
    """
    
    def __init__(self, config: ReActConfig):
        self.config = config
    
    async def prompt(self, state: S, task: str, env: Env) -> Result[S, str]:
        """Prompt step that formats history and scratchpad into a prompt."""
        history_entries = state.history.snapshot()
        history_block = "\n".join(history_entries)

        # Include task in the first prompt
        task_context = f"\nTask: {task}" if task else ""

        prompt = (
            "You are a ReAct agent. Respond only with valid JSON using keys "
            '"thought", "action", "action_input", "final". '
            "Set exactly one of action or final.\n"
            '"action_input" must be an object/dictionary, not a string.\n\n'
            f"History:\n{history_block}\n\n"
            f"Scratchpad:\n{state.scratchpad}{task_context}\n"
        )
        next_state = state.with_evidence("prompt", info={})
        return Result(next_state, value=prompt, control=Control.Continue())

    async def think(self, state: S, prompt: str, env: Env) -> Result[S, str]:
        """Think step that calls the model to generate a response."""
        if env.runtime_context:
            response = await env.model.stream_complete(prompt, env.runtime_context)
        else:
            response = await env.model.complete(prompt)
        next_state = state.with_evidence("think")
        return Result(next_state, value=response, control=Control.Continue())

    async def decide(self, state: S, model_output: str, env: Env) -> Result[S, ToolUse | str]:
        """Decide step that parses model output and decides on action or final answer."""
        return await react_decide(state, model_output, env)

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
        next_state = state.with_history(observation)
        next_state = self._append_scratchpad(next_state, observation)
        next_state = next_state.with_evidence("observation", info={"tool_result_id": result.id})
        return Result(next_state, value=observation, control=Control.Continue())
    
    def _append_scratchpad(self, state: S, line: str) -> S:
        """Helper method to append a line to the scratchpad."""
        if state.scratchpad:
            return state.with_scratchpad(f"{state.scratchpad}\n{line}")
        return state.with_scratchpad(line)
    
    def run(self, initial_state: S, task: str = "") -> Agent[S, str]:
        """Run ReAct agent with optimized recursive loop.

        Args:
            initial_state: The initial state for the agent
            task: The task to perform (included in the prompt)
        """
        def loop(state: S, step_index: int) -> Agent[S, str]:
            """Tail-recursive loop function that creates a new Agent for each step."""
            async def continue_or_halt(
                next_state: S, value: str, env: Env
            ) -> Result[S, str]:
                next_step = step_index + 1
                if next_step >= self.config.max_steps:
                    return Result(next_state, value=value, control=Control.Halt())
                # Tail-recursive call - creates new Agent instance
                result = await loop(next_state, next_step).run(env)
                return result

            return (
                Agent.start(state, task)
                .then(self.prompt)
                .then(self.think)
                .then(self.decide)
                .then(self.act)
                .then(self.observe)
                .then(continue_or_halt)
            )

        return loop(initial_state, 0)


def run_react_agent(initial_state: S) -> Agent[S, str]:
    """Run ReAct agent with optimized recursive loop."""
    config = ReActConfig()
    policy = ReActPolicy(config)
    return policy.run(initial_state)
