"""Error types for structured value casting."""

from __future__ import annotations


class CastError(Exception):
    """Error raised when value casting/validation fails.

    This error preserves the raw value for retry classification
    and debugging purposes.
    """

    def __init__(self, message: str, raw_value: object) -> None:
        self.raw_value = raw_value
        super().__init__(message)

    def __repr__(self) -> str:
        return f"CastError({super().__repr__()}, raw_value={self.raw_value!r})"
