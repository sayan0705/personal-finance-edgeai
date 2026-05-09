"""Input guardrail — validates user messages before they reach the LLM."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from app.api.config import get


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""


class InputGuardrail:
    """Validates incoming user messages against configurable rules.

    Rules (from ``guardrails.input`` config section):
    - Minimum / maximum message length
    - Prompt injection pattern detection
    """

    def __init__(self) -> None:
        cfg = get("guardrails.input", {})
        self._max_length: int = int(cfg.get("max_length", 4000))
        self._min_length: int = int(cfg.get("min_length", 2))
        self._injection_patterns: list[str] = [
            p.lower() for p in cfg.get("injection_patterns", [])
        ]

    def validate(self, user_message: str) -> GuardrailResult:
        """Check a user message against all input rules.

        Args:
            user_message: The raw user text to validate.

        Returns:
            GuardrailResult with ``allowed=True`` if safe, else ``allowed=False``
            and a human-readable ``reason``.
        """
        text = user_message.strip()

        if len(text) < self._min_length:
            logger.warning("InputGuardrail: message too short")
            return GuardrailResult(allowed=False, reason="Message is too short.")

        if len(text) > self._max_length:
            logger.warning(f"InputGuardrail: message too long ({len(text)} chars)")
            return GuardrailResult(
                allowed=False,
                reason=f"Message exceeds maximum length of {self._max_length} characters.",
            )

        lower = text.lower()
        for pattern in self._injection_patterns:
            if pattern in lower:
                logger.warning(f"InputGuardrail: injection pattern detected — '{pattern}'")
                return GuardrailResult(
                    allowed=False,
                    reason="Your message contains content that cannot be processed.",
                )

        return GuardrailResult(allowed=True)
