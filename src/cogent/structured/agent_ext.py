"""Agent extensions for structured value transformations."""

from typing import TypeVar

from cogent.kernel.agent import Agent
from cogent.structured.cast import make_cast_step
from cogent.structured.schema import OutputSchema

T = TypeVar("T")


def cast(self: Agent, schema: OutputSchema[T]) -> Agent:
    """Cast the agent's value to a new type using a schema.

    This method validates and transforms the agent's output value using
    the provided schema.

    Args:
        schema: The output schema to validate against

    Returns:
        New agent with the value type transformed to T

    Example:
        >>> from cogent.structured import CallableSchema
        >>> def parse_user(data):
        ...     return UserProfile(**data)
        >>> agent = agent.cast(CallableSchema(parse_user))
    """
    return self.then(make_cast_step(schema))


# Register the cast operation
Agent.register_op("cast", cast)
