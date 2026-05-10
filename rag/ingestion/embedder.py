"""Offline embedder: encodes document chunks and upserts them into ChromaDB."""

from __future__ import annotations

import hashlib
from typing import Any

from loguru import logger

from rag.models import Document
from rag.vector_store.chroma_store import ChromaVectorStore


class EmbeddingIndexer:
    """Computes embeddings for document chunks and indexes them in ChromaDB.

    Designed for **offline** use — run this script separately before starting
    the API server. The API retriever then queries the pre-built index at
    query time without recomputing embeddings.

    Embedding is delegated to ChromaDB's configured embedding function
    (``SentenceTransformerEmbeddingFunction``) so the same model is used
    for both indexing and retrieval, guaranteeing vector-space consistency.

    Args:
        vector_store: Initialised ChromaVectorStore.
        batch_size: Number of chunks to upsert per ChromaDB call.
            Larger batches are faster but use more memory.
    """

    def __init__(self, vector_store: ChromaVectorStore, batch_size: int = 128) -> None:
        self._store = vector_store
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, chunks: list[Document]) -> int:
        """Embed and upsert chunks into ChromaDB.

        Skips chunks with empty content. Uses a deterministic content-hash
        as the ID when the chunk's ``id`` field is missing or empty.

        Args:
            chunks: Chunked Documents from SemanticChunker.

        Returns:
            Number of chunks successfully upserted.
        """
        valid = [c for c in chunks if c.content.strip()]
        if not valid:
            logger.warning("No non-empty chunks to index")
            return 0

        total_upserted = 0
        for batch_start in range(0, len(valid), self._batch_size):
            batch = valid[batch_start : batch_start + self._batch_size]

            ids = [_stable_id(c) for c in batch]
            documents = [c.content for c in batch]
            metadatas = [_sanitise_metadata(c.metadata) for c in batch]

            self._store.add(ids=ids, documents=documents, metadatas=metadatas)
            total_upserted += len(batch)
            logger.info(
                f"Indexed batch {batch_start // self._batch_size + 1}: "
                f"{total_upserted}/{len(valid)} chunks"
            )

        logger.info(f"Indexing complete — {total_upserted} chunks in ChromaDB")
        return total_upserted

    def clear_and_reindex(self, chunks: list[Document]) -> int:
        """Drop the collection and rebuild it from scratch.

        Args:
            chunks: All chunks to index after clearing.

        Returns:
            Number of chunks upserted.
        """
        logger.warning("Clearing ChromaDB collection before reindexing")
        self._store.delete_collection()
        return self.index(chunks)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _stable_id(doc: Document) -> str:
    """Return a stable unique ID for a chunk, falling back to content hash."""
    if doc.id:
        return doc.id
    return "sha256:" + hashlib.sha256(doc.content.encode()).hexdigest()[:16]


def _sanitise_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Ensure all metadata values are ChromaDB-compatible scalar types."""
    clean: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean
