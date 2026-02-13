"""Tests for structured value casting (Agent.cast)."""

import asyncio
from dataclasses import dataclass

from cogent import Agent, Result, Control
from cogent.structured import (
    CastError,
    CallableSchema,
    DictSchema,
    parse_json_if_needed,
    make_cast_step,
)
from fakes import make_fake_env


# Test schema implementations
def parse_user_profile(data: dict) -> "UserProfile":
    """Parse a user profile from a dict."""
    if not isinstance(data, dict):
        raise ValueError("Expected dict")
    required = ["name", "email"]
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    return UserProfile(name=data["name"], email=data["email"])


@dataclass(frozen=True)
class UserProfile:
    name: str
    email: str


def test_parse_json_if_needed_with_string() -> None:
    """Test parsing JSON string."""
    result = parse_json_if_needed('{"name": "test", "email": "test@example.com"}')
    assert result == {"name": "test", "email": "test@example.com"}


def test_parse_json_if_needed_with_dict() -> None:
    """Test passing dict through unchanged."""
    data = {"name": "test", "email": "test@example.com"}
    result = parse_json_if_needed(data)
    assert result is data  # Same object, not copied


def test_parse_json_if_needed_with_invalid_json() -> None:
    """Test invalid JSON raises CastError."""
    try:
        parse_json_if_needed('{"name": "test", invalid}')
        assert False, "Expected CastError"
    except CastError as e:
        assert "Invalid JSON" in str(e)
        assert e.raw_value == '{"name": "test", invalid}'


def test_callable_schema_validate_success() -> None:
    """Test CallableSchema validation success."""
    schema = CallableSchema(parse_user_profile)
    data = {"name": "John", "email": "john@example.com"}
    result = schema.validate(data)
    assert isinstance(result, UserProfile)
    assert result.name == "John"
    assert result.email == "john@example.com"


def test_callable_schema_validate_failure() -> None:
    """Test CallableSchema validation failure raises."""
    schema = CallableSchema(parse_user_profile)
    data = {"name": "John"}  # Missing email

    try:
        schema.validate(data)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "email" in str(e)


def test_dict_schema_validate_success() -> None:
    """Test DictSchema validation success."""
    schema = DictSchema(required_fields={"name": str, "age": int})
    data = {"name": "John", "age": 30}
    result = schema.validate(data)
    assert result == data


def test_dict_schema_validate_missing_field() -> None:
    """Test DictSchema missing required field."""
    schema = DictSchema(required_fields={"name": str, "age": int})
    data = {"name": "John"}  # Missing age

    try:
        schema.validate(data)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "age" in str(e)


def test_dict_schema_validate_wrong_type() -> None:
    """Test DictSchema wrong type."""
    schema = DictSchema(required_fields={"name": str, "age": int})
    data = {"name": "John", "age": "30"}  # age should be int

    try:
        schema.validate(data)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "int" in str(e)


def test_dict_schema_no_required_fields() -> None:
    """Test DictSchema with no required fields."""
    schema = DictSchema()
    data = {"any": "thing", "goes": True}
    result = schema.validate(data)
    assert result == data


def test_cast_step_with_json_string() -> None:
    """Test cast step with JSON string input."""
    async def run_flow():
        schema = CallableSchema(parse_user_profile)
        step = make_cast_step(schema)

        state = "initial"
        value = '{"name": "Alice", "email": "alice@example.com"}'

        result = await step(state, value, None)
        return result

    result = asyncio.run(run_flow())
    assert result.control.kind == "continue"
    assert isinstance(result.value, UserProfile)
    assert result.value.name == "Alice"


def test_cast_step_with_dict() -> None:
    """Test cast step with dict input."""
    async def run_flow():
        schema = CallableSchema(parse_user_profile)
        step = make_cast_step(schema)

        state = "initial"
        value = {"name": "Bob", "email": "bob@example.com"}

        result = await step(state, value, None)
        return result

    result = asyncio.run(run_flow())
    assert result.control.kind == "continue"
    assert isinstance(result.value, UserProfile)
    assert result.value.name == "Bob"


def test_cast_step_with_invalid_json() -> None:
    """Test cast step with invalid JSON triggers retry_dirty."""
    async def run_flow():
        schema = CallableSchema(parse_user_profile)
        step = make_cast_step(schema)

        state = "initial"
        value = "not valid json"

        result = await step(state, value, None)
        return result

    result = asyncio.run(run_flow())
    assert result.control.kind == "error"
    assert "Invalid JSON" in str(result.control.reason)


def test_cast_step_with_validation_error() -> None:
    """Test cast step with validation error triggers retry_dirty."""
    async def run_flow():
        schema = CallableSchema(parse_user_profile)
        step = make_cast_step(schema)

        state = "initial"
        value = {"name": "Charlie"}  # Missing email

        result = await step(state, value, None)
        return result

    result = asyncio.run(run_flow())
    assert result.control.kind == "error"
    assert "email" in str(result.control.reason)


def test_agent_cast_with_json_string() -> None:
    """Test Agent.cast with JSON string."""
    async def run_flow():
        schema = CallableSchema(parse_user_profile)
        flow = Agent.start("state", '{"name": "Dave", "email": "dave@example.com"}')
        flow = flow.cast(schema)
        return await flow.run(make_fake_env())

    result = asyncio.run(run_flow())
    assert result.control.kind == "continue"
    assert isinstance(result.value, UserProfile)
    assert result.value.name == "Dave"


def test_agent_cast_with_dict() -> None:
    """Test Agent.cast with dict."""
    async def run_flow():
        schema = CallableSchema(parse_user_profile)
        flow = Agent.start("state", {"name": "Eve", "email": "eve@example.com"})
        flow = flow.cast(schema)
        return await flow.run(make_fake_env())

    result = asyncio.run(run_flow())
    assert result.control.kind == "continue"
    assert isinstance(result.value, UserProfile)
    assert result.value.name == "Eve"


def test_agent_cast_chain_multiple() -> None:
    """Test multiple chained casts."""
    async def run_flow():
        def double(x: int) -> int:
            return x * 2

        flow = Agent.start("state", 5)
        flow = flow.cast(CallableSchema(double))
        flow = flow.cast(CallableSchema(double))
        return await flow.run(make_fake_env())

    result = asyncio.run(run_flow())
    assert result.control.kind == "continue"
    assert result.value == 20  # 5 * 2 * 2


def test_agent_cast_insert_after_step() -> None:
    """Test cast can be inserted after a tool step."""
    async def run_flow():
        from cogent.core.result import Control, Result

        async def tool_step(s, v, env):
            _ = (s, env)
            return Result(s, value='{"name": "Frank", "email": "frank@example.com"}', control=Control.Continue())

        schema = CallableSchema(parse_user_profile)
        flow = Agent.start("state", "initial")
        flow = flow.then(tool_step)  # First step produces JSON string
        flow = flow.cast(schema)  # Then cast to UserProfile
        return await flow.run(make_fake_env())

    result = asyncio.run(run_flow())
    assert result.control.kind == "continue"
    assert isinstance(result.value, UserProfile)
    assert result.value.name == "Frank"


def test_cast_error_preserves_raw_value() -> None:
    """Test CastError preserves the raw value."""
    try:
        parse_json_if_needed("invalid json")
    except CastError as e:
        assert e.raw_value == "invalid json"


def test_agent_cast_type_propagation() -> None:
    """Test that type propagation works correctly."""
    # This is a compile-time check - if it compiles, type propagation works
    schema = CallableSchema(parse_user_profile)

    # Create an agent with a specific value type
    agent: Agent[str, str] = Agent.start("state", "value")

    # Cast transforms the value type
    cast_agent: Agent[str, UserProfile] = agent.cast(schema)

    # Verify the agent has the right structure (type checking done by mypy)
    assert cast_agent is not None
