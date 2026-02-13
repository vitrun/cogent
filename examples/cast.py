#!/usr/bin/env python3
"""
Agent.cast() example - Type-safe value transformation.

This example demonstrates using Agent.cast() to validate and transform
agent output values to structured types.

Key concepts:
- cast() validates and transforms the agent's value type
- On validation failure, retries with adaptation (retry_dirty)
- Works with JSON strings or dicts
- Type-safe: propagates the new type through the agent chain
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from cogent import Agent, Env
from cogent.structured import CallableSchema, DictSchema


# ==================== Data Models ====================

@dataclass(frozen=True)
class UserProfile:
    """User profile with validated fields."""
    name: str
    email: str
    age: int | None = None


@dataclass(frozen=True)
class Order:
    """Order with item and quantity."""
    item: str
    quantity: int


# ==================== Schemas ====================

def parse_user_profile(data: dict) -> UserProfile:
    """Parse and validate a user profile from dict."""
    if not isinstance(data, dict):
        raise ValueError("Expected dict")

    required = ["name", "email"]
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    return UserProfile(
        name=data["name"],
        email=data["email"],
        age=data.get("age"),
    )


def parse_order(data: dict) -> Order:
    """Parse and validate an order from dict."""
    if not isinstance(data, dict):
        raise ValueError("Expected dict")

    required = ["item", "quantity"]
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(data["quantity"], int):
        raise ValueError("quantity must be an integer")

    return Order(item=data["item"], quantity=data["quantity"])


# ==================== Example Usage ====================

async def example_basic_cast():
    """Basic example: cast JSON string to typed object."""
    print("\n=== Basic Cast ===")

    # JSON string from some LLM output
    json_value = '{"name": "Alice", "email": "alice@example.com", "age": 30}'

    # Create agent and cast to UserProfile
    user_schema = CallableSchema(parse_user_profile)
    agent: Agent[str, UserProfile] = Agent.start("state", json_value).cast(user_schema)

    # Run (using fake env for demonstration)
    from tests.fakes import make_fake_env
    result = await agent.run(make_fake_env())

    print(f"Input: {json_value}")
    print(f"Output type: {type(result.control.value).__name__}")
    print(f"Output value: {result.control.value}")
    print(f"Control kind: {result.control.kind}")


async def example_chain_cast():
    """Example: multiple chained casts."""
    print("\n=== Chained Casts ===")

    # Start with integer, double it, then cast to string
    def double(x: int) -> int:
        return x * 2

    def to_str(x: int) -> str:
        return f"Result: {x}"

    agent = Agent.start("state", 5)
    agent = agent.cast(CallableSchema(double))
    agent = agent.cast(CallableSchema(to_str))

    from tests.fakes import make_fake_env
    result = await agent.run(make_fake_env())

    print(f"Input: 5")
    print(f"After double: 10")
    print(f"Final output: {result.control.value}")


async def example_dict_schema():
    """Example: using DictSchema for simple validation."""
    print("\n=== Dict Schema ===")

    # DictSchema for simple field validation
    order_schema = DictSchema(required_fields={"item": str, "quantity": int})

    # Dict input (not JSON string)
    dict_value = {"item": "Widget", "quantity": 3}

    agent: Agent[str, dict] = Agent.start("state", dict_value).cast(order_schema)

    from tests.fakes import make_fake_env
    result = await agent.run(make_fake_env())

    print(f"Input: {dict_value}")
    print(f"Validated: {result.control.value}")
    print(f"Control: {result.control.kind}")


async def example_insert_after_tool():
    """Example: cast inserted after a tool step."""
    print("\n=== Cast After Tool Step ===")

    from cogent.core.result import Control, Result

    # Step that simulates tool returning JSON
    async def tool_step(s, v, env):
        return Result(s, control=Control.Continue('{"name": "Bob", "email": "bob@test.com"}'))

    # Cast the JSON output to UserProfile
    user_schema = CallableSchema(parse_user_profile)
    agent = Agent.start("state", "initial").then(tool_step).cast(user_schema)

    from tests.fakes import make_fake_env
    result = await agent.run(make_fake_env())

    print(f"Tool output: {{'name': 'Bob', 'email': 'bob@test.com'}}")
    print(f"Cast to: {result.control.value}")
    print(f"Type: {type(result.control.value).__name__}")


async def example_type_propagation():
    """Example: type propagation through cast."""
    print("\n=== Type Propagation ===")

    user_schema = CallableSchema(parse_user_profile)

    # Type is automatically propagated
    agent: Agent[str, UserProfile] = Agent.start("state", '{"name": "Test", "email": "test@test.com"}')
    agent = agent.cast(user_schema)

    # The type annotation above would cause mypy error if cast didn't work
    print(f"Agent type: Agent[S, UserProfile]")
    print(f"Type propagation: Works correctly!")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Agent.cast() Examples")
    print("=" * 60)

    await example_basic_cast()
    await example_chain_cast()
    await example_dict_schema()
    await example_insert_after_tool()
    await example_type_propagation()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
