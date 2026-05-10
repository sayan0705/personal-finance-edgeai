"""Hybrid retriever: ChromaDB vector search + BM25 fused via Reciprocal Rank Fusion."""

from __future__ import annotations

import threading
from typing import Any

from loguru import logger
from rank_bm25 import BM25Okapi

from rag.models import Document
from rag.vector_store.chroma_store import ChromaVectorStore


class HybridRetriever:
    """Combines dense vector search and sparse BM25 keyword search with RRF fusion.

    Workflow:
        1. ``warm_up()`` — fetches all document texts from ChromaDB and builds
           an in-memory BM25 index. Call once after the pipeline initialises.
        2. ``retrieve(query, k)`` — runs both searches in parallel, merges with
           Reciprocal Rank Fusion, and returns the top-k documents.

    The BM25 index is refreshed automatically when ``warm_up()`` is called
    again (e.g. after new documents are ingested offline).

    Args:
        vector_store: Initialised ChromaVectorStore instance.
        bm25_enabled: When False, falls back to pure vector search.
        rrf_k: RRF constant (higher → flatter score distribution). Default 60.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_enabled: bool = True,
        rrf_k: int = 60,
    ) -> None:
        self._store = vector_store
        self._bm25_enabled = bm25_enabled
        self._rrf_k = rrf_k

        # BM25 state — guarded by a lock for thread-safe warm_up/retrieve
        self._bm25_lock = threading.RLock()
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_texts: list[str] = []
        self._bm25_metas: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """Build the in-memory BM25 index from all ChromaDB documents.

        Safe to call multiple times (rebuilds the index on each call).
        No-op when ``bm25_enabled`` is False.
        """
        if not self._bm25_enabled:
            logger.info("BM25 disabled — skipping warm-up")
            return

        logger.info("Building BM25 index from ChromaDB corpus…")
        try:
            ids, texts, metas = self._store.get_all()
        except Exception as exc:
            logger.error(f"BM25 warm-up failed (ChromaDB unreachable?): {exc}")
            return

        if not texts:
            logger.warning("ChromaDB collection is empty — BM25 index not built")
            return

        tokenized = [_tokenize(t) for t in texts]
        with self._bm25_lock:
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_ids = ids
            self._bm25_texts = texts
            self._bm25_metas = metas

        logger.info(f"BM25 index ready with {len(texts)} documents")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 10) -> list[Document]:
        """Retrieve the top-k most relevant documents for a query.

        Runs vector search and (if enabled) BM25 search, then fuses results
        via Reciprocal Rank Fusion.

        Args:
            query: Natural-language query string.
            k: Number of documents to return.

        Returns:
            List of Documents sorted by descending RRF score.
        """
        logger.debug(f"HybridRetriever.retrieve(k={k}): {query[:80]!r}")

        vector_results = self._vector_search(query, k)

        if self._bm25_enabled:
            bm25_results = self._bm25_search(query, k)
            if bm25_results:
                fused = self._rrf([vector_results, bm25_results])
                logger.debug(f"RRF fusion: vector={len(vector_results)}, bm25={len(bm25_results)} → {len(fused)}")
                return fused[:k]

        return vector_results[:k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, k: int) -> list[Document]:
        try:
            return self._store.query(query, n_results=k)
        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            return []

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        with self._bm25_lock:
            if self._bm25 is None:
                logger.debug("BM25 index not ready — skipping keyword search")
                return []

            tokens = _tokenize(query)
            scores = self._bm25.get_scores(tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

            results: list[Document] = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    break
                results.append(
                    Document(
                        id=self._bm25_ids[idx],
                        content=self._bm25_texts[idx],
                        metadata=self._bm25_metas[idx],
                        score=float(scores[idx]),
                    )
                )
            return results

    def _rrf(self, rankings: list[list[Document]]) -> list[Document]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            rankings: Each sub-list is an ordered list of Documents.

        Returns:
            Single merged list ordered by descending RRF score.
        """
        k = self._rrf_k
        doc_map: dict[str, Document] = {}
        rrf_scores: dict[str, float] = {}

        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                doc_map[doc.id] = doc
                rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)

        merged: list[Document] = []
        for doc_id in sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True):
            result_doc = doc_map[doc_id]
            result_doc.score = rrf_scores[doc_id]
            merged.append(result_doc)

        return merged


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer for BM25."""
    return text.lower().split()
