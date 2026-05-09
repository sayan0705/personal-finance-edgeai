"""Tool registry — central store for all available agent tools."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from agents.tools.base import BaseTool


class ToolRegistry:
    """Manages registration and execution of agent tools.

    Usage::

        registry = ToolRegistry()
        registry.register(TaxCalculator())
        registry.register(SIPCalculator())

        result = registry.execute("tax_calculator", {"annual_income": 1000000, "regime": "new"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.

        Args:
            tool: A BaseTool instance to register.
        """
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> BaseTool | None:
        """Return the tool with the given name, or None if not found.

        Args:
            name: Tool name as returned by the LLM.

        Returns:
            BaseTool instance or None.
        """
        return self._tools.get(name)

    def list_all(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """Execute a named tool and return a formatted observation string.

        Args:
            name: Tool name.
            args: Input arguments dict.

        Returns:
            JSON-formatted result string, or an error message.
        """
        tool = self.get(name)
        if tool is None:
            logger.warning(f"ToolRegistry: unknown tool '{name}'")
            return f"Error: tool '{name}' not found. Available: {', '.join(self._tools)}"

        try:
            result = tool.execute(args)
            logger.debug(f"Tool '{name}' executed successfully")
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"Tool '{name}' raised exception: {exc}")
            return f"Error executing {name}: {exc}"

    def to_prompt_block(self) -> str:
        """Format all tools as a prompt block for the LLM system message.

        Returns:
            Multi-line string describing all tools with their schemas.
        """
        lines = ["Available tools:\n"]
        for tool in self._tools.values():
            params = tool.input_schema.get("properties", {})
            param_desc = ", ".join(
                f"{k} ({v.get('type', 'any')})" for k, v in params.items()
            )
            lines.append(f"- **{tool.name}**: {tool.description}")
            if param_desc:
                lines.append(f"  Parameters: {param_desc}")
        return "\n".join(lines)
