"""Generic PDF → text loader with pypdf primary and pdfminer fallback."""

from __future__ import annotations

from pathlib import Path
from typing import List

from loguru import logger

from .base import BaseLoader, Document

_MIN_MEANINGFUL_CHARS = 50


class GenericPDFLoader(BaseLoader):
    """Extract text from a single PDF file.

    Attempts extraction with pypdf first (fast). If the result is empty or
    suspiciously short it falls back to pdfminer.six which handles complex
    layouts and multi-column PDFs better.

    Args:
        file_path: Path to the PDF file.
        metadata: Extra key/value pairs merged into every returned Document.
    """

    def __init__(self, file_path: str | Path, metadata: dict | None = None) -> None:
        self.file_path = Path(file_path)
        self.metadata = metadata or {}

    def load(self) -> List[Document]:
        """Extract text from the PDF.

        Returns:
            A single-element list containing the extracted Document, or an
            empty list when no text can be recovered.

        Raises:
            FileNotFoundError: If the PDF does not exist on disk.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.file_path}")

        text = self._extract_pypdf()
        if len(text.strip()) < _MIN_MEANINGFUL_CHARS:
            logger.debug(
                f"{self.file_path.name}: pypdf returned short text "
                f"({len(text)} chars), falling back to pdfminer"
            )
            text = self._extract_pdfminer()

        if not text.strip():
            logger.warning(f"No text extracted from {self.file_path}")
            return []

        meta = {
            "source": str(self.file_path),
            "file_name": self.file_path.name,
            "format": "pdf",
            **self.metadata,
        }
        return [Document(page_content=text.strip(), metadata=meta)]

    # ── private helpers ───────────────────────────────────────────────────────

    def _extract_pypdf(self) -> str:
        try:
            from pypdf import PdfReader  # lazy import — optional dependency

            reader = PdfReader(str(self.file_path))
            pages: list[str] = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception as exc:
                    logger.debug(f"pypdf: skipping page in {self.file_path.name}: {exc}")
            return "\n".join(pages)
        except Exception as exc:
            logger.debug(f"pypdf failed on {self.file_path.name}: {exc}")
            return ""

    def _extract_pdfminer(self) -> str:
        try:
            from pdfminer.high_level import extract_text  # lazy import

            return extract_text(str(self.file_path)) or ""
        except Exception as exc:
            logger.debug(f"pdfminer failed on {self.file_path.name}: {exc}")
            return ""
