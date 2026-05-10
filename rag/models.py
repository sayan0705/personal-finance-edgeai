"""Shared data models for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A retrieved document chunk with content, metadata, and relevance score.

    Attributes:
        id: Unique identifier (usually source_path + chunk index).
        content: The raw text of the chunk.
        metadata: Arbitrary key-value pairs (source, title, page, date, …).
        score: Relevance score assigned by the retriever or reranker.
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def source(self) -> str:
        """Human-readable source label from metadata."""
        return self.metadata.get("source", "unknown")

    @property
    def title(self) -> str:
        """Document title from metadata."""
        return self.metadata.get("title", "")

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, score={self.score:.4f}, source={self.source!r})"
