from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol


class TrimPolicy(Protocol):
    def __call__(self, entries: list[Any]) -> list[Any]: ...


class Memory(ABC):
    """
    Memory 抽象类，提供内存操作接口
    """
    
    @abstractmethod
    def append(self, entry: Any) -> Memory:
        """
        添加内存条目
        
        Args:
            entry: 要添加的内存条目
            
        Returns:
            新的 Memory 实例
        """
        pass
    
    @abstractmethod
    def query(self, predicate: Callable[[Any], bool]) -> Iterable[Any]:
        """
        根据谓词查询内存
        
        Args:
            predicate: 查询谓词函数
            
        Returns:
            匹配的内存条目
        """
        pass
    
    @abstractmethod
    def snapshot(self) -> list[Any]:
        """
        获取内存快照
        
        Returns:
            内存条目的快照
        """
        pass
    
    @abstractmethod
    def trim(self, policy: TrimPolicy) -> Memory:
        """
        根据策略裁剪内存
        
        Args:
            policy: 裁剪策略
            
        Returns:
            裁剪后的新 Memory 实例
        """
        pass


@dataclass(frozen=True)
class SimpleMemory(Memory):
    """
    简单的 Memory 实现，使用分层机制提高性能
    """
    
    _base: Memory | None = None
    _current: tuple[Any, ...] = ()
    
    def append(self, entry: Any) -> Memory:
        new_current = self._current + (entry,)
        return SimpleMemory(_base=self._base, _current=new_current)
    
    def query(self, predicate: Callable[[Any], bool]) -> Iterable[Any]:
        if self._base:
            yield from self._base.query(predicate)
        yield from (entry for entry in self._current if predicate(entry))
    
    def snapshot(self) -> list[Any]:
        result = []
        if self._base:
            result.extend(self._base.snapshot())
        result.extend(self._current)
        return result
    
    def trim(self, policy: TrimPolicy) -> Memory:
        all_entries = self.snapshot()
        trimmed_entries = policy(all_entries)
        return SimpleMemory(_current=tuple(trimmed_entries))