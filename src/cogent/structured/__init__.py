"""Structured value casting for Agent type transformation.

This module provides the Agent.cast() primitive for type-level value
convergence in the agent monad.
"""

# Import agent_ext to register capabilities
from . import agent_ext  # noqa: F401
from .cast import make_cast_step
from .errors import CastError
from .parser import parse_json_if_needed
from .schema import (
    CallableSchema,
    DictSchema,
    OutputSchema,
    PydanticSchema,
    try_import_pydantic,
)

__all__ = [
    "CastError",
    "OutputSchema",
    "CallableSchema",
    "DictSchema",
    "PydanticSchema",
    "try_import_pydantic",
    "parse_json_if_needed",
    "make_cast_step",
]
