"""Quality validation for synthetic and real Q&A training samples."""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

_REQUIRED_ROLES = {"system", "user", "assistant"}
_PLACEHOLDER_PATTERN = re.compile(r"\{[A-Z_]+\}|\[[A-Z][A-Z_]+\]")
_PII_PATTERN = re.compile(
    r"\b\d{10}\b"          # mobile numbers
    r"|\b[A-Z]{5}\d{4}[A-Z]\b"   # PAN card
    r"|\b\d{12}\b"         # Aadhaar
    r"|\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",  # email
    re.IGNORECASE,
)


class DataQualityValidator:
    """Validates that a ChatML sample meets quality requirements.

    Checks performed:
      - Correct ChatML structure (3 messages with system/user/assistant roles)
      - Minimum and maximum answer length
      - No unfilled template placeholders
      - No accidental PII inclusion (can be skipped for known-synthetic data)
      - Non-empty question and answer text

    Args:
        min_answer_length: Minimum character count for the assistant turn.
        max_answer_length: Maximum character count for the assistant turn.
        skip_pii: Skip PII detection when True (use for synthetic generators
            that intentionally include PAN/phone number patterns as training examples).
    """

    def __init__(
        self,
        min_answer_length: int = 80,
        max_answer_length: int = 2000,
        skip_pii: bool = False,
    ) -> None:
        self._min_ans = min_answer_length
        self._max_ans = max_answer_length
        self._skip_pii = skip_pii

    def validate(self, sample: dict[str, Any]) -> list[str]:
        """Run all checks and return a list of error strings (empty = valid).

        Args:
            sample: A ChatML dict with a ``messages`` key.

        Returns:
            List of validation error messages. Empty list means the sample is valid.
        """
        errors: list[str] = []
        errors.extend(self._check_structure(sample))
        if errors:
            return errors  # Can't safely run further checks without valid structure

        messages: list[dict] = sample["messages"]
        roles = {m["role"]: m["content"] for m in messages}

        errors.extend(self._check_lengths(roles))
        errors.extend(self._check_placeholders(roles))
        if not self._skip_pii:
            errors.extend(self._check_pii(roles))

        return errors

    def validate_batch(self, samples: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
        """Split a batch into valid and invalid samples.

        Args:
            samples: List of ChatML dicts to validate.

        Returns:
            Tuple of (valid_samples, invalid_samples).
        """
        valid, invalid = [], []
        for sample in samples:
            if self.validate(sample):
                invalid.append(sample)
            else:
                valid.append(sample)

        if invalid:
            logger.warning(f"Validation: {len(invalid)}/{len(samples)} samples failed quality checks")
        return valid, invalid

    # ── private checks ────────────────────────────────────────────────────────

    @staticmethod
    def _check_structure(sample: dict) -> list[str]:
        errors: list[str] = []
        if "messages" not in sample:
            return ["Missing 'messages' key"]

        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) != 3:
            errors.append(f"Expected 3 messages, got {len(messages) if isinstance(messages, list) else 'non-list'}")
            return errors

        roles = {m.get("role") for m in messages}
        missing = _REQUIRED_ROLES - roles
        if missing:
            errors.append(f"Missing roles: {missing}")

        for msg in messages:
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                errors.append(f"Empty or non-string content for role '{msg.get('role')}'")

        return errors

    def _check_lengths(self, roles: dict[str, str]) -> list[str]:
        errors: list[str] = []
        answer = roles.get("assistant", "")

        if len(answer) < self._min_ans:
            errors.append(
                f"Answer too short: {len(answer)} chars (minimum {self._min_ans})"
            )
        if len(answer) > self._max_ans:
            errors.append(
                f"Answer too long: {len(answer)} chars (maximum {self._max_ans})"
            )

        question = roles.get("user", "")
        if len(question.strip()) < 10:
            errors.append(f"Question too short: '{question[:50]}'")

        return errors

    @staticmethod
    def _check_placeholders(roles: dict[str, str]) -> list[str]:
        errors: list[str] = []
        for role, content in roles.items():
            matches = _PLACEHOLDER_PATTERN.findall(content)
            if matches:
                errors.append(f"Unfilled placeholder in '{role}': {matches[:3]}")
        return errors

    @staticmethod
    def _check_pii(roles: dict[str, str]) -> list[str]:
        errors: list[str] = []
        for role, content in roles.items():
            match = _PII_PATTERN.search(content)
            if match:
                errors.append(f"Potential PII found in '{role}': '{match.group()[:20]}...'")
        return errors
