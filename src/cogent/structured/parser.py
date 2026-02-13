"""JSON parsing utilities for structured value casting."""

from __future__ import annotations

import json
from typing import Any

from .errors import CastError


def parse_json_if_needed(value: str | Any) -> Any:
    """Parse JSON string if the value is a string.

    This function attempts to parse a string as JSON. If the value
    is not a string, it is returned as-is.

    Args:
        value: The value to potentially parse as JSON

    Returns:
        The parsed JSON object, or the original value if not a string

    Raises:
        CastError: If the value is a string but cannot be parsed as JSON
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise CastError(f"Invalid JSON: {e.msg} at position {e.pos}", value) from e
        except Exception as e:
            raise CastError(f"Failed to parse JSON: {str(e)}", value) from e
    return value
