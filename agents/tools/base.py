"""Abstract base class for all FinEdge agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Contract that every agent tool must fulfil.

    Subclasses declare their ``name``, ``description``, and ``input_schema``
    as class attributes, then implement ``execute()``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case tool identifier used in ReAct Action lines."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description shown to the LLM in the tools prompt."""

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema dict describing the tool's input parameters."""

    @abstractmethod
    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run the tool with the given arguments.

        Args:
            args: Dict of input parameters matching ``input_schema``.

        Returns:
            Dict containing the result. Must always include a ``"result"`` key
            with a human-readable summary string.
        """
