# CLAUDE.md

## Project Overview

Cogent is a Python library implementing principled AI agent orchestration using monadic patterns, inspired by [Monadic Context Engineering](https://arxiv.org/abs/2512.22431). It formalizes agent workflows as composable computations with proper state management, error handling, and optional parallelism.

## Development Commands

### Setup Development Environment
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Run Tests
```bash
pytest
```

### Run Linting
```bash
ruff check src tests
```

### Run Type Checking
```bash
pyright
```

## Key Design Patterns
1. **Monadic Composition**: Steps compose via `.then()`, maintaining state threading
2. **Registry Pattern**: Tools registered in `ToolRegistry` for dynamic discovery
3. **Error Propagation**: Failures short-circuit with full state preservation

## Important Environment Variables
- `OPENROUTER_API_KEY`: For LLM integration
- `OPENROUTER_MODEL`: Model selection (default: x-ai/grok-4.1-fast)
- `OPENROUTER_BASE_URL`: API endpoint override
- `OPENROUTER_TIMEOUT_S`: Request timeout (default: 30)


## Rationale
**Do more with less**
**Core Insight**: Execution is just state evolution. Trace is just evidence of that evolution. We don't need a separate system to track what we can naturally capture in our state.

## Code Conventions
- Type hints required (Python 3.12+)
- Ruff formatting with 100 char line length
- Pydantic models for all structured data
- Tests alongside implementation
- No external dependencies beyond pydantic (base) and test tools
- **Preference**: Use compositional abstractions over new system complexity