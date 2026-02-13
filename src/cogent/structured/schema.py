"""Schema abstractions for value validation and casting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class OutputSchema(Protocol[T]):
    """Protocol for output value schemas.

    Schemas validate and transform raw values into structured types.
    """

    def validate(self, value: Any) -> T:
        """Validate and transform the input value.

        Args:
            value: The raw value to validate

        Returns:
            The validated/transformed value

        Raises:
            Exception: If validation fails
        """
        ...

    def describe(self) -> str:
        """Return a human-readable description of this schema.

        Returns:
            Description string for debugging/display purposes
        """
        ...


@dataclass(frozen=True)
class CallableSchema(OutputSchema[T]):
    """Schema that uses a callable for validation.

    The callable should raise an exception if validation fails,
    and return the validated/transformed value on success.
    """

    fn: Callable[[Any], T]
    _description: str | None = None

    def validate(self, value: Any) -> T:
        return self.fn(value)

    def describe(self) -> str:
        if self._description:
            return self._description
        return self.fn.__name__


@dataclass(frozen=True)
class DictSchema(OutputSchema[dict[str, Any]]):
    """Schema for validating dictionary structures.

    Validates that the input is a dict and optionally checks
    field existence and types.
    """

    required_fields: dict[str, type] | None = None

    def validate(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value).__name__}")

        if self.required_fields:
            for field_name, field_type in self.required_fields.items():
                if field_name not in value:
                    raise ValueError(f"Missing required field: {field_name}")
                field_value = value[field_name]
                if not isinstance(field_value, field_type):
                    raise ValueError(
                        f"Field '{field_name}' expected {field_type.__name__}, "
                        f"got {type(field_value).__name__}"
                    )

        return value

    def describe(self) -> str:
        if self.required_fields:
            fields = ", ".join(f"{k}: {v.__name__}" for k, v in self.required_fields.items())
            return f"DictSchema({fields})"
        return "DictSchema"


@dataclass(frozen=True)
class PydanticSchema(OutputSchema[T]):
    """Schema that uses a Pydantic model for validation.

    Only available when pydantic is installed.
    Requires pydantic to be importable.
    """

    model: type[T]

    def __post_init__(self) -> None:
        if not try_import_pydantic():
            raise ImportError("PydanticSchema requires pydantic to be installed")

    def validate(self, value: Any) -> T:
        # Import pydantic at validate time to ensure it's available
        # Use model_validate which is the standard pydantic v2 method
        # The model is guaranteed to have this method since it's a pydantic model
        validate_fn = getattr(self.model, "model_validate", None)
        if validate_fn is None:
            # Fallback for pydantic v1
            validate_fn = getattr(self.model, "parse_obj", None)
        if validate_fn is None:
            raise ValueError(f"Model {self.model} has no validation method")
        return validate_fn(value)  # type: ignore[return-value]

    def describe(self) -> str:
        return f"PydanticSchema({self.model.__name__})"


def try_import_pydantic() -> bool:
    """Check if pydantic is available.

    Returns:
        True if pydantic can be imported, False otherwise
    """
    import importlib.util
    return importlib.util.find_spec("pydantic") is not None
