"""Generic HTML → text loader using BeautifulSoup."""

from __future__ import annotations

from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from loguru import logger

from .base import BaseLoader, Document

_NOISE_TAGS = frozenset(
    ["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "iframe"]
)
_MAIN_SELECTORS = ["main", "article", '[role="main"]', "#content", "#main-content", ".content"]


class GenericHTMLLoader(BaseLoader):
    """Parse HTML content and extract the meaningful body text.

    Removes boilerplate tags (nav, footer, scripts), then prefers a ``<main>``
    or ``<article>`` element before falling back to the full ``<body>``.

    Args:
        content: Raw HTML string or path to an HTML file.
        metadata: Extra key/value pairs merged into the returned Document.
        base_url: Used to populate the ``source`` metadata field.
        parser: BeautifulSoup parser backend. ``lxml`` is fastest when installed.
    """

    def __init__(
        self,
        content: str | Path,
        metadata: dict | None = None,
        base_url: str = "",
        parser: str = "lxml",
    ) -> None:
        if isinstance(content, Path):
            if not content.exists():
                raise FileNotFoundError(f"HTML file not found: {content}")
            self._html = content.read_text(encoding="utf-8", errors="replace")
        else:
            self._html = content
        self.metadata = metadata or {}
        self.base_url = base_url
        self.parser = parser

    def load(self) -> List[Document]:
        """Parse HTML and return a single Document.

        Returns:
            A single-element list, or empty list if no meaningful text found.
        """
        soup = BeautifulSoup(self._html, self.parser)

        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        title = self._extract_title(soup)
        body = self._extract_body(soup)

        if not body.strip():
            logger.warning(f"No text extracted from {self.base_url or 'HTML content'}")
            return []

        meta = {
            "source": self.base_url,
            "title": title,
            "format": "html",
            **self.metadata,
        }
        return [Document(page_content=body.strip(), metadata=meta)]

    # ── private helpers ───────────────────────────────────────────────────────

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for selector in ["title", "h1"]:
            tag = soup.find(selector)
            if tag:
                return tag.get_text(strip=True)
        return ""

    def _extract_body(self, soup: BeautifulSoup) -> str:
        for selector in _MAIN_SELECTORS:
            main = soup.select_one(selector)
            if main:
                return main.get_text(separator="\n", strip=True)
        body = soup.find("body")
        target = body or soup
        return target.get_text(separator="\n", strip=True)
