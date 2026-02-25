# Cogent

**Build production-ready AI agents without rewriting your stack.**

Composable. Deterministic. Inspectable.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-green.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2.0)


## The Problem

Most agent frameworks are:

- Easy to prototype, hard to maintain  
- Powerful, but structurally chaotic  
- Difficult to debug  
- Opaque in production  

As systems grow, orchestration becomes fragile.

## What Cogent Does Differently

Cogent gives you:

- ✅ Structured agent orchestration  
- ✅ Deterministic execution model  
- ✅ First-class trace & audit  
- ✅ Composable multi-agent workflows  
- ✅ Built-in LiteLLM support (100+ models)

You can start simple — and scale without rewriting everything.

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick Examples

### 1. Simple ReAct Agent with Tools
```python
from cogent.agents.react import ReactAgent
from cogent.kernel import ToolPort, ToolCall, Result

class CalculatorTools(ToolPort):
    async def call(self, state, call: ToolCall) -> Result:
        if call.name == "add":
            return Result(state, value=str(sum(call.args.values())))
        return Result(state, value=f"Unknown tool: {call.name}")

agent = ReactAgent(model="anthropic/claude-sonnet-4.6", tools=CalculatorTools())
result = await agent.run("What's 2 + 2?")
print(result.value)
```

### 2. Streaming Responses
```python
agent = ReactAgent(model="anthropic/claude-sonnet-4.6")
async for chunk in agent.stream("Explain quantum computing simply"):
    print(chunk, end="")
```

### 3. Multi-Agent Composition
Cogent supports:
- Agent routing
- Handoff
- Parallel execution
- Concurrent orchestration

See [examples/multi_agent.py](./examples/multi_agent.py) for a complete multi-agent workflow with handoff, routing, and concurrent execution.

### 4. Structured Outputs
```python
from dataclasses import dataclass
from cogent import Agent
from cogent.structured import CallableSchema, PydanticSchema

@dataclass
class UserProfile:
    name: str
    email: str

def parse_profile(data: dict) -> UserProfile:
    return UserProfile(name=data["name"], email=data["email"])

agent = Agent.start('{"name": "Alice", "email": "alice@example.com"}')
agent = agent.cast(CallableSchema(parse_profile))
result = await agent.run("state", env)
assert isinstance(result.value, UserProfile)
```

### 5. Trace & Evidence (Debugging & Audit)
```python
from cogent.agents.react import ReactAgent, ReActState

agent = ReactAgent(model="anthropic/claude-sonnet-4.6", trace=True)
result = await agent.run("Analyze this market data")

# Inspect the trace
if result.trace:
    for event in result.trace._events:
        print(f"{event.action}: {event.info}")
```

### 6. Compositional Agent Building
```python
from cogent import Agent
from cogent.kernel import Result, Control

async def step1(state, value, env):
    return Result(state, value=value + " processed", control=Control.Continue())

async def step2(state, value, env):
    return Result(state, value=value + " finalized", control=Control.Continue())

workflow = Agent.start("initial").then(step1).then(step2)
result = await workflow.run("state", env)
print(result.value)  # "initial processed finalized"
```

## LLM Providers

Cogent has **built-in LiteLLM integration** supporting 100+ models (OpenAI, Anthropic, OpenRouter, etc.). Set any supported API key:

```bash
export OPENROUTER_API_KEY="your-key"  # OpenRouter
export ANTHROPIC_API_KEY="your-key"   # Anthropic
export OPENAI_API_KEY="your-key"      # OpenAI
```

### Extensible Provider System
Cogent is designed to be extensible. Implement your own provider by subclassing `FormatterBase`:

```python
from cogent.providers.base import FormatterBase
from cogent.model import Message

class CustomProvider(FormatterBase):
    support_tools_api = True
    support_vision = False
    
    async def format(self, messages: list[Message]) -> list[dict]:
        # Convert Cogent messages to your provider's format
        return [{"role": msg.role, "content": msg.content} for msg in messages]
```

## Tooling

- Type checking: `pyright`
- Linting: `ruff`
- Tests: `pytest`

```bash
pyright
ruff check src tests
pytest
```

## Design Principles

For a complete understanding of the architectural design and principles, read [DESIGN.md](./DESIGN.md).

