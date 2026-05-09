"""Sliding-window conversation memory for short-term context."""

from __future__ import annotations

from collections import deque
from typing import Any

from loguru import logger

from app.api.config import get


class ShortTermMemory:
    """Maintains a fixed-size rolling buffer of conversation messages.

    Args:
        window: Maximum number of messages to retain. Oldest messages
            are dropped when the buffer is full.
    """

    def __init__(self, window: int | None = None) -> None:
        size = window or int(get("memory.short_term_window", 10))
        self._buffer: deque[dict[str, str]] = deque(maxlen=size)
        logger.debug(f"ShortTermMemory initialized with window={size}")

    def add(self, role: str, content: str) -> None:
        """Append a message to the buffer.

        Args:
            role: ``"user"`` or ``"assistant"``.
            content: Message text.
        """
        self._buffer.append({"role": role, "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        """Return all buffered messages in chronological order.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        return list(self._buffer)

    def clear(self) -> None:
        """Clear all buffered messages."""
        self._buffer.clear()
        logger.debug("ShortTermMemory cleared")

    def __len__(self) -> int:
        return len(self._buffer)
