from __future__ import annotations

from typing import Any, Protocol, TypeVar

from ..core.memory import Memory

S = TypeVar("S")


class ReActStateProtocol(Protocol[S]):
    """Protocol for states that can be used with ReAct."""
    history: Memory
    scratchpad: str
    evidence: Any
    formatted_anthropic_messages: list[dict]
    formatted_messages: list[dict]
    
    def with_history(self, entry: str) -> S:
        ...
    
    def with_scratchpad(self, text: str) -> S:
        ...
    
    def with_evidence(self, action: str, **kwargs) -> S:
        ...
    
    def with_formatted_anthropic_messages(self, messages: list[dict]) -> S:
        ...
    
    def with_formatted_messages(self, messages: list[dict]) -> S:
        ...
