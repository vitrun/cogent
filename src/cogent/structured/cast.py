"""Type casting step for Agent value transformation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from cogent.core.env import Env
from cogent.core.result import Control, Result
from cogent.structured.errors import CastError
from cogent.structured.parser import parse_json_if_needed
from cogent.structured.schema import OutputSchema

S = TypeVar("S")
V = TypeVar("V")
T = TypeVar("T")


def make_cast_step(schema: OutputSchema[T]) -> Callable[[S, V, Env], Awaitable[Result[S, T]]]:
    """Create a type casting step for the agent.

    This step validates and transforms the value using the provided schema.
    On failure, it returns RetryDirty to trigger retry with adaptation.

    Args:
        schema: The output schema to validate against

    Returns:
        A step function that validates and transforms the value
    """
    async def step(state: S, value: V, env: Env) -> Result[S, T]:
        try:
            # Parse JSON if the value is a string
            parsed = parse_json_if_needed(value)
            # Validate against the schema
            validated: T = schema.validate(parsed)
            return Result(state, control=Control.Continue(validated))
        except CastError as e:
            # CastError triggers retry_dirty for automatic retry
            return Result(state, control=Control.RetryDirty(reason=str(e)))  # type: ignore[return-value]
        except Exception as e:
            # Other validation errors also trigger retry_dirty
            return Result(state, control=Control.RetryDirty(reason=str(e)))  # type: ignore[return-value]

    return step
