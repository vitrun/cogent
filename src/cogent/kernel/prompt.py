"""Prompt Template System for Cogent Agent runtime.

This module provides a minimal, production-grade prompt template system that supports:
- Explicit variable declaration
- Deterministic rendering
- Version tracking
- Prompt hashing
- Metadata separation (structure vs rendered text)
- Optional integration with Trace (metadata only)

Architectural constraints:
- Prompt is a structured artifact, not a raw string
- Template rendering is pure and deterministic
- Templates are versioned
- Runtime may record metadata but must NOT modify prompt content
- Templates must NOT introduce control-flow semantics
- Templates must NOT access memory, model, tools, or state
- Rendering must be explicit about required variables
- No implicit variable injection
- No hidden globals
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cogent.kernel.trace import Trace


@dataclass(frozen=True)
class RenderedPrompt:
    """Immutable rendered prompt artifact.

    This is a pure data container representing a fully rendered prompt
    with all variables substituted. It contains:
    - text: The fully rendered prompt string
    - template_name: Name of the template used
    - template_version: Version of the template used
    - variables: The exact dict passed to render()
    - hash: SHA256 hash of the rendered text
    """

    text: str
    template_name: str
    template_version: str
    variables: dict[str, Any]
    hash: str


@dataclass(frozen=True)
class PromptTemplate:
    """Immutable prompt template with explicit variable declaration.

    Templates declare their required variables upfront. Rendering is validated
    against this declaration - missing or extra variables cause errors.

    Constraints:
    - variables explicitly declare required placeholders
    - rendering fails if missing or extra variables exist
    - rendering uses str.format(**values)
    - rendering returns RenderedPrompt
    - rendering is deterministic (no side effects, no external deps)
    """

    name: str
    version: str
    content: str
    variables: set[str]

    def render(self, values: dict[str, Any]) -> RenderedPrompt:
        """Render the template with the given values.

        Args:
            values: Dictionary of variable values to substitute

        Returns:
            RenderedPrompt with substituted text and metadata

        Raises:
            ValueError: If there are missing or extra variables
        """
        provided = set(values.keys())
        declared = self.variables
        missing = declared - provided
        extra = provided - declared

        if missing:
            raise ValueError(f"Missing required variables for template '{self.name}': {missing}")

        if extra:
            raise ValueError(f"Extra variables provided for template '{self.name}': {extra}")

        rendered_text = self.content.format(**values)
        text_hash = hashlib.sha256(rendered_text.encode()).hexdigest()

        return RenderedPrompt(
            text=rendered_text,
            template_name=self.name,
            template_version=self.version,
            variables=dict(values),
            hash=text_hash,
        )


class PromptRegistry:
    """Lightweight in-memory registry for prompt templates.

    Templates are stored by (name, version) tuple. This is a simple
    registration system without dynamic loading or external dependencies.
    """

    def __init__(self) -> None:
        self._templates: dict[tuple[str, str], PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        """Register a template in the registry.

        Args:
            template: The template to register

        Raises:
            ValueError: If a template with the same name and version exists
        """
        key = (template.name, template.version)
        if key in self._templates:
            raise ValueError(
                f"Template '{template.name}' version '{template.version}' is already registered"
            )
        self._templates[key] = template

    def get(self, name: str, version: str) -> PromptTemplate:
        """Retrieve a template by name and version.

        Args:
            name: Template name
            version: Template version

        Returns:
            The registered PromptTemplate

        Raises:
            KeyError: If no template matches the name/version
        """
        key = (name, version)
        if key not in self._templates:
            raise KeyError(f"No template found for '{name}' version '{version}'")
        return self._templates[key]


def record_prompt(
    trace: Trace,
    prompt: RenderedPrompt,
    parent_id: int | None,
) -> int | None:
    """Record prompt rendering metadata to trace.

    This records only metadata about the rendered prompt, not the full content.
    This keeps traces lightweight while providing audit capability.

    Recorded info:
    - template_name: Name of the template
    - template_version: Version of the template
    - hash: SHA256 hash of the rendered text
    - variable_keys: Set of variable keys (NOT values)

    Args:
        trace: The trace context to record to
        prompt: The rendered prompt to record
        parent_id: Parent event ID for tree relationships

    Returns:
        The event ID from trace.record()
    """
    return trace.record(
        action="prompt_rendered",
        info={
            "template_name": prompt.template_name,
            "template_version": prompt.template_version,
            "hash": prompt.hash,
            "variable_keys": list(prompt.variables.keys()),
        },
        parent_id=parent_id,
    )
