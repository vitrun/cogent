"""Tests for prompt template system."""

import hashlib
from dataclasses import FrozenInstanceError

import pytest

from cogent.kernel.prompt import PromptRegistry, PromptTemplate, record_prompt
from cogent.kernel.trace import Trace


class TestRenderedPrompt:
    """Tests for RenderedPrompt dataclass."""

    def test_rendered_prompt_is_immutable(self):
        """Verify RenderedPrompt cannot be mutated after creation."""
        template = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )
        rendered = template.render({"name": "World"})

        with pytest.raises(FrozenInstanceError):
            rendered.text = "mutated"  # type: ignore[reportAttributeAccessIssue]


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass."""

    def test_template_is_immutable(self):
        """Verify PromptTemplate cannot be mutated after creation."""
        template = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )

        with pytest.raises(FrozenInstanceError):
            template.name = "mutated"  # type: ignore[reportAttributeAccessIssue]

    def test_render_with_valid_variables(self):
        """Test rendering with all declared variables."""
        template = PromptTemplate(
            name="greet",
            version="v1",
            content="Hello {name}, your balance is {balance}",
            variables={"name", "balance"},
        )

        rendered = template.render({"name": "Alice", "balance": "100"})

        assert rendered.text == "Hello Alice, your balance is 100"
        assert rendered.template_name == "greet"
        assert rendered.template_version == "v1"

    def test_render_missing_variables_raises_error(self):
        """Test that missing variables raise ValueError."""
        template = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )

        with pytest.raises(ValueError, match="Missing required variables"):
            template.render({})

    def test_render_extra_variables_raises_error(self):
        """Test that extra variables raise ValueError."""
        template = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )

        with pytest.raises(ValueError, match="Extra variables provided"):
            template.render({"name": "Alice", "extra": "value"})

    def test_same_input_produces_same_hash(self):
        """Verify deterministic rendering produces same hash."""
        template = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )

        rendered1 = template.render({"name": "World"})
        rendered2 = template.render({"name": "World"})

        assert rendered1.hash == rendered2.hash

    def test_different_variable_value_produces_different_hash(self):
        """Verify different values produce different hashes."""
        template = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )

        rendered1 = template.render({"name": "Alice"})
        rendered2 = template.render({"name": "Bob"})

        assert rendered1.hash != rendered2.hash

    def test_hash_verification(self):
        """Verify hash is SHA256 of rendered text."""
        template = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )

        rendered = template.render({"name": "World"})
        expected_hash = hashlib.sha256(rendered.text.encode()).hexdigest()

        assert rendered.hash == expected_hash


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    def test_register_and_get(self):
        """Test basic registration and retrieval."""
        template = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )
        registry = PromptRegistry()
        registry.register(template)

        retrieved = registry.get("test", "v1")

        assert retrieved is template

    def test_same_name_different_version_allowed(self):
        """Test that same name with different version can be registered."""
        template1 = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )
        template2 = PromptTemplate(
            name="test",
            version="v2",
            content="Hi {name}",
            variables={"name"},
        )
        registry = PromptRegistry()
        registry.register(template1)
        registry.register(template2)

        assert registry.get("test", "v1") is template1
        assert registry.get("test", "v2") is template2

    def test_overwrite_same_name_version_fails(self):
        """Test that overwriting same name/version raises error."""
        template1 = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )
        template2 = PromptTemplate(
            name="test",
            version="v1",
            content="Different content",
            variables=set(),
        )
        registry = PromptRegistry()
        registry.register(template1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(template2)

    def test_get_nonexistent_raises_key_error(self):
        """Test that getting nonexistent template raises KeyError."""
        registry = PromptRegistry()

        with pytest.raises(KeyError, match="No template found"):
            registry.get("nonexistent", "v1")


class TestRecordPrompt:
    """Tests for record_prompt function."""

    def test_record_prompt_metadata_only(self):
        """Verify only metadata is recorded, not full content."""
        template = PromptTemplate(
            name="test",
            version="v1",
            content="Secret: {password}",
            variables={"password"},
        )
        rendered = template.render({"password": "supersecret"})
        trace = Trace(enabled=True)

        event_id = record_prompt(trace, rendered, parent_id=None)

        assert event_id is not None
        assert event_id >= 0
        events = trace.get_events()
        assert len(events) == 1

        ev = events[0]
        assert ev.action == "prompt_rendered"
        assert ev.info["template_name"] == "test"
        assert ev.info["template_version"] == "v1"
        assert ev.info["hash"] == rendered.hash
        assert ev.info["variable_keys"] == ["password"]
        # Verify full content is NOT recorded
        assert "Secret" not in str(ev.info)
        assert "supersecret" not in str(ev.info)

    def test_record_prompt_with_parent_id(self):
        """Verify parent_id is forwarded to trace."""
        template = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )
        rendered = template.render({"name": "World"})
        trace = Trace(enabled=True)

        event_id = record_prompt(trace, rendered, parent_id=42)

        assert event_id is not None
        assert event_id >= 0
        events = trace.get_events()
        assert events[0].parent_id == 42

    def test_record_prompt_disabled_trace(self):
        """Verify nothing is recorded when trace is disabled."""
        template = PromptTemplate(
            name="test",
            version="v1",
            content="Hello {name}",
            variables={"name"},
        )
        rendered = template.render({"name": "World"})
        trace = Trace(enabled=False)

        event_id = record_prompt(trace, rendered, parent_id=None)

        assert event_id is None
        assert len(trace.get_events()) == 0
