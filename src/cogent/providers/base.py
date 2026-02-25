"""Base classes for provider formatters."""

from abc import ABC, abstractmethod
from typing import Any

from cogent.model import Message


class FormatterBase(ABC):
    """Base class for all message formatters."""

    support_tools_api: bool = False
    """Whether support tools API"""

    support_vision: bool = False
    """Whether support vision data"""

    supported_blocks: list[type] = []
    """The list of supported message blocks"""

    @abstractmethod
    async def format(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages into provider-specific API format.

        Args:
            messages (List[Message]):
                The list of message objects to format.

        Returns:
            List[Dict[str, Any]]:
                The formatted messages as a list of dictionaries.
        """
        pass

    def assert_list_of_messages(self, messages: list[Message]) -> None:
        """Assert that the input is a list of Message objects.

        Args:
            messages (List[Message]):
                The list of message objects to check.

        Raises:
            TypeError:
                If the input is not a list of Message objects.
        """
        if not isinstance(messages, list):
            raise TypeError(f"Expected list of Message objects, got {type(messages)}")

        for msg in messages:
            if not isinstance(msg, Message):
                raise TypeError(f"Expected Message object, got {type(msg)}")
