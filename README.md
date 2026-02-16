# Cogent

![Python 3.12+](https://img.shields.io/badge/python-3.12+-green.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Cogent is a compositional agent kernel with product-level ergonomics. It provides a principled architecture for AI agent orchestration that treats workflows as composable computations in a shared context.

This repository provides a Python implementation, organized into strict architectural layers. For detailed design principles, see [DESIGN.md](./DESIGN.md).

## Core Architecture

Cogent is divided into layers with strict dependency boundaries:

- **kernel**: Minimal compositional algebra (most stable)
- **combinators**: Higher-order agent composition
- **agents**: Product-facing strategies (e.g., ReActAgent)
- **providers**: Model adapters (litellm integration)
- **runtime**: Default implementations
- **structured**: Typed output extension
- **model**: Model blocks

## Project Layout

- `src/cogent/kernel/`: Core compositional algebra
- `src/cogent/agents/`: Agent implementations (ReAct, etc.)
- `src/cogent/combinators/`: Agent composition operators
- `src/cogent/providers/`: LLM provider adapters (litellm)
- `src/cogent/runtime/`: Runtime implementations and tool registry
- `src/cogent/structured/`: Structured output parsing
- `examples/`: Runnable examples (minimal, ReAct, multi-agent, etc.)
- `tests/`: Test suite

## Quickstart (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Example Usage

See the `examples/` directory for more comprehensive examples.

### Minimal Example

```python
from cogent.agents.react import ReActAgent
from cogent.providers.litellm import LiteLLMProvider

# Create agent and run
provider = LiteLLMProvider(model="gpt-4o-mini")
agent = ReActAgent(provider=provider)
result = agent.run("What is a Monad?")

if result.valid:
    print(result.value)
else:
    print("Failure:", result.error)
```

## LLM Providers

Cogent uses LiteLLM for model integration. Set your API key:

```bash
export OPENAI_API_KEY="your-key"
```

Or use other providers supported by LiteLLM:

```bash
export ANTHROPIC_API_KEY="your-key"
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
