# CLAUDE.md

## Project Overview

Cogent is a Python library implementing principled AI agent orchestration using monadic patterns.

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
see [DESIGN.md](DESIGN.md)

## Important Environment Variables
- `OPENROUTER_API_KEY`: For LLM integration

## Code Conventions
- Type hints required (Python 3.12+)
- Ruff formatting with 100 char line length
- Pydantic models for all structured data
- Tests alongside implementation
- No external dependencies beyond pydantic (base) and test tools
- **Preference**: Use compositional abstractions over new system complexity