from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar, Protocol, Self

from pydantic import BaseModel, ValidationError

from ..core import Control, Result
from ..core import Agent
from ..core import Env
from ..core import ToolCall, ToolResult
from ..core.memory import Memory, SimpleMemory
from .evidence import Evidence

S = TypeVar("S")


class ReActStateProtocol(Protocol):
    """Protocol for states that can be used with ReAct."""
    history: Memory
    scratchpad: str
    evidence: Any
    
    def with_history(self, entry: str) -> S:
        ...
    
    def with_scratchpad(self, text: str) -> S:
        ...
    
    def with_evidence(self, action: str, **kwargs) -> S:
        ...


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


async def react_prompt(state: ReActStateProtocol[S], _: str, env: Env) -> Result[S, str]:
    history_entries = state.history.snapshot()
    history_block = "\n".join(history_entries)
    prompt = (
        "You are a ReAct agent. Respond only with valid JSON using keys "
        '"thought", "action", "action_input", "final". '
        "Set exactly one of action or final.\n\n"
        f"History:\n{history_block}\n\n"
        f"Scratchpad:\n{state.scratchpad}\n"
    )
    next_state = state.with_evidence("prompt", info={})
    return Result(next_state, control=Control.Continue(prompt))


async def react_think(state: ReActStateProtocol[S], prompt: str, env: Env) -> Result[S, str]:
    response = await env.model.complete(prompt)
    next_state = state.with_evidence("think")
    return Result(next_state, control=Control.Continue(response))


async def react_decide(state: ReActStateProtocol[S], model_output: str, env: Env) -> Result[S, ToolCall | str]:
    try:
        parsed = ReActOutput.model_validate_json(model_output)
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
        return Result(next_state, control=Control.Halt(parsed.final))

    if parsed.action:
        tool_call = ToolCall(
            id=f"tool-{int(time.time() * 1000)}",
            name=parsed.action,
            args=action_input,
        )
        return Result(next_state, control=Control.Continue(tool_call))

    return Result(next_state, control=Control.Error("Missing action or final"))


async def react_act(state: ReActStateProtocol[S], call: ToolCall | str, env: Env) -> Result[S, ToolResult]:
    if not isinstance(call, ToolCall):
        return Result(state, control=Control.Error("No tool call to execute"))
    next_state = state.with_evidence("tool_call", info={"name": call.name, "args": call.args})
    try:
        content = await env.tools.call(call.name, call.args)
        result = ToolResult.success(call.id, str(content))
        return Result(next_state, control=Control.Continue(result))
    except Exception as e:
        result = ToolResult.failure(call.id, str(e))
        return Result(next_state, control=Control.Error(f"Tool execution failed: {e}"))


async def react_observe(state: ReActStateProtocol[S], result: ToolResult, env: Env) -> Result[S, str]:
    observation = f"Observation: {result.content}"
    next_state = state.with_history(observation)
    next_state = _append_scratchpad(next_state, observation)
    next_state = next_state.with_evidence("observation", info={"tool_result_id": result.id})
    return Result(next_state, control=Control.Continue(observation))


def _append_scratchpad(state: ReActStateProtocol[S], line: str) -> S:
    if state.scratchpad:
        return state.with_scratchpad(f"{state.scratchpad}\n{line}")
    return state.with_scratchpad(line)


def run_react_agent(initial_state: S) -> Agent[S, str]:
    """Run ReAct agent with optimized recursive loop."""
    config = ReActConfig()

    def loop(state: S, step_index: int) -> Agent[S, str]:
        """Tail-recursive loop function that creates a new Agent for each step."""
        async def continue_or_halt(
            next_state: S, value: str, env: Env
        ) -> Result[S, str]:
            next_step = step_index + 1
            if next_step >= config.max_steps:
                return Result(next_state, control=Control.Halt(value))
            # Tail-recursive call - creates new Agent instance
            result = await loop(next_state, next_step).run(env)
            return result

        return (
            Agent.start(state, "")
            .then(react_prompt)
            .then(react_think)
            .then(react_decide)
            .then(react_act)
            .then(react_observe)
            .then(continue_or_halt)
        )

    return loop(initial_state, 0)
