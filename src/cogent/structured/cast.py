"""Type casting step for Agent value transformation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from cogent.kernel.result import Control, Result
from cogent.ports.env import Env
from cogent.structured.errors import CastError
from cogent.structured.parser import parse_json_if_needed
from cogent.structured.schema import OutputSchema

S = TypeVar("S")
V = TypeVar("V")
T = TypeVar("T")


def make_cast_step(schema: OutputSchema[T], max_retries: int = 3) -> Callable[[S, V, Env], Awaitable[Result[S, T]]]:
    """Create a type casting step for the agent.

    This step validates and transforms the value using the provided schema.
    It handles retry internally - up to max_retries attempts.

    Args:
        schema: The output schema to validate against
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        A step function that validates and transforms the value
    """
    async def step(state: S, value: V, env: Env) -> Result[S, T]:
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                # Parse JSON if the value is a string
                parsed = parse_json_if_needed(value)
                # Validate against the schema
                validated: T = schema.validate(parsed)
                return Result(state, value=validated, control=Control.Continue())
            except CastError as e:
                last_error = e
                # Continue to retry
            except Exception as e:
                last_error = e
                # Continue to retry

        # All retries exhausted
        return Result(state, control=Control.Error(str(last_error)))

    return step
