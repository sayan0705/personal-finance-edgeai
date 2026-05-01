"""Text cleaning utilities for raw ingested content."""

from __future__ import annotations

import re
import unicodedata


class TextCleaner:
    """Normalises raw text extracted from PDFs and HTML.

    Handles common artefacts from PDF extraction (hyphenation, ligatures,
    control characters) while preserving Unicode including Devanagari script.

    Args:
        min_line_length: Lines shorter than this are stripped as noise.
        max_consecutive_newlines: Collapse longer blank-line runs to this limit.
    """

    def __init__(
        self,
        min_line_length: int = 3,
        max_consecutive_newlines: int = 2,
    ) -> None:
        self._min_line = min_line_length
        self._max_newlines = max_consecutive_newlines

    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline on *text*.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text string.
        """
        text = self._remove_control_chars(text)
        text = self._fix_pdf_hyphenation(text)
        text = self._normalise_whitespace(text)
        text = self._remove_noise_lines(text)
        text = self._collapse_blank_lines(text)
        return text.strip()

    # ── private steps ─────────────────────────────────────────────────────────

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        """Remove ASCII control characters except newline and tab."""
        return "".join(
            ch for ch in text
            if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
        )

    @staticmethod
    def _fix_pdf_hyphenation(text: str) -> str:
        """Rejoin words split by soft-hyphen at end of line."""
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """Replace multiple spaces/tabs with a single space, preserve newlines."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" \n", "\n", text)
        return text

    def _remove_noise_lines(self, text: str) -> str:
        """Drop lines that are too short to be meaningful content."""
        lines = text.splitlines()
        cleaned = [line for line in lines if len(line.strip()) >= self._min_line or not line.strip()]
        return "\n".join(cleaned)

    def _collapse_blank_lines(self, text: str) -> str:
        """Collapse runs of blank lines to at most max_consecutive_newlines."""
        pattern = r"\n{" + str(self._max_newlines + 1) + r",}"
        return re.sub(pattern, "\n" * self._max_newlines, text)
