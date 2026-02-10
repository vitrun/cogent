# Cogent

![Python 3.12+](https://img.shields.io/badge/python-3.12+-green.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Cogent is a principled architecture for AI agent orchestration that treats workflows as composable computations in a shared context. It formalizes how state, errors, and side effects propagate through agent steps, using the algebraic structures of Functors, Applicatives, and Monads.

This repository provides a Python implementation, including:

- `AgentMonad`: sequential, stateful, fallible computation chains.
- `AsyncAgentMonad`: async flows with Applicative parallelism via `gather`.
- Pydantic models for structured state and tool calls/results.

## Core Idea

Cogent models an agent workflow as a single container that carries:

- State (memory/history)
- A value (current step output)
- A success/failure signal

The `.then()` operator composes steps. If any step fails, the chain short-circuits, preserving the error and state at the point of failure.

## Project Layout

- `src/cogent/monads.py`: `AgentMonad` and `AsyncAgentMonad`.
- `src/cogent/models.py`: pydantic models and a tool registry.
- `src/cogent/steps.py`: reference steps from the paper (plan, execute, synthesize, format).
- `src/cogent/__main__.py`: a runnable demo.
- `docs/paper.tex`: the research paper.
- `docs/index.html`: project page.

## Quickstart (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Example Usage

```python
from cogent.steps import run_simple_agent

flow = run_simple_agent("What is a Monad?")
if flow.valid:
    print(flow.value)
    for entry in flow.state.history:
        print("-", entry)
else:
    print("Failure:", flow.error)
```

## OpenRouter 

The default steps are deterministic. To use a live LLM call via OpenRouter:

```bash
export OPENROUTER_API_KEY="your-key"
export OPENROUTER_MODEL="x-ai/grok-4.1-fast"
```

```python
from cogent.steps import run_openrouter_agent

flow = run_openrouter_agent("What is a Monad?")
if flow.valid:
    print(flow.value)
else:
    print("Failure:", flow.error)
```

Optional environment variables:

- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `OPENROUTER_REFERER` (sets `HTTP-Referer`)
- `OPENROUTER_TITLE` (sets `X-Title`)
- `OPENROUTER_TIMEOUT_S` (default: `30`)

## Tooling

- Type checking: `pyright`
- Linting: `ruff`
- Tests: `pytest`

```bash
pyright
ruff check src tests
pytest
```

## Notes

- The implementation follows the conceptual design in the paper, including the `AgentMonad`
  and `AsyncAgentMonad` APIs.
- The default example steps are deterministic and self-contained; use OpenRouter for live LLM calls.
