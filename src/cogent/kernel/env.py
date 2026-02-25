"""Environment and Context for Cogent - default implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from cogent.kernel.ports import MemoryPort, ModelPort, SinkPort, ToolPort
from cogent.kernel.trace import Trace


class TrimPolicy(Protocol):
    """Policy for trimming context entries."""

    def __call__(self, entries: list[Any]) -> list[Any]:
        ...


class Context(ABC):
    """
    Execution-local working context.
    Part of domain state.
    Must be pure and persistent-style.
    """

    @abstractmethod
    def append(self, entry: Any) -> Context:
        """Add an entry to the context."""
        pass

    @abstractmethod
    def query(self, predicate: Callable[[Any], bool]) -> Iterable[Any]:
        """Query the context by predicate."""
        pass

    @abstractmethod
    def snapshot(self) -> tuple[Any, ...]:
        """Get a snapshot of the context."""
        pass

    @abstractmethod
    def trim(self, policy: TrimPolicy) -> Context:
        """Trim the context according to policy."""
        pass


@dataclass(frozen=True)
class InMemoryContext(Context):
    """
    In-memory implementation of Context.
    Immutable - all operations return new instances.
    """

    _entries: tuple[Any, ...] = ()

    def append(self, entry: Any) -> InMemoryContext:
        """Add an entry immutably."""
        return InMemoryContext(_entries=self._entries + (entry,))

    def query(self, predicate: Callable[[Any], bool]) -> Iterable[Any]:
        """Query entries matching predicate."""
        return (entry for entry in self._entries if predicate(entry))

    def snapshot(self) -> tuple[Any, ...]:
        """Get a snapshot of all entries."""
        return self._entries

    def trim(self, policy: TrimPolicy) -> InMemoryContext:
        """Trim entries according to policy immutably."""
        current_list = list(self._entries)
        trimmed_list = policy(current_list)
        return InMemoryContext(_entries=tuple(trimmed_list))


@dataclass
class Env:
    """Environment aggregation - combines all ports."""

    model: ModelPort
    tools: ToolPort | None = None
    memory: MemoryPort | None = None
    trace: Trace | None = None
    sink: SinkPort | None = None
