"""Output guardrail — filters LLM responses before returning to the user."""

from __future__ import annotations

from loguru import logger

from app.api.config import get


class OutputGuardrail:
    """Post-processes LLM output to remove leakage and enforce length limits.

    Rules (from ``guardrails.output`` config section):
    - Strip system-prompt leak phrases
    - Cap output at max_length characters
    """

    def __init__(self) -> None:
        cfg = get("guardrails.output", {})
        self._max_length: int = int(cfg.get("max_length", 8000))
        self._leak_patterns: list[str] = [
            p.lower() for p in cfg.get("system_leak_patterns", [])
        ]

    def process(self, response: str) -> str:
        """Apply all output filters to an LLM response.

        Args:
            response: Raw LLM output text.

        Returns:
            Filtered response string.
        """
        lower = response.lower()
        for pattern in self._leak_patterns:
            if pattern in lower:
                logger.warning(f"OutputGuardrail: system leak detected — '{pattern}'")
                idx = lower.find(pattern)
                response = response[:idx].rstrip()

        if len(response) > self._max_length:
            logger.warning(f"OutputGuardrail: truncating response ({len(response)} chars)")
            response = response[: self._max_length] + "\n\n*(Response truncated)*"

        return response
