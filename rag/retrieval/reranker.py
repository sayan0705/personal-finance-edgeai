"""Cross-encoder reranker for RAG post-retrieval scoring."""

from __future__ import annotations

from loguru import logger

from rag.models import Document


class CrossEncoderReranker:
    """Reranks a candidate set of documents using a cross-encoder model.

    The model is loaded lazily on the first call to ``rerank()`` to avoid
    increasing cold-start time when reranking is optional.

    Cross-encoders jointly encode the query and each document and produce
    a single relevance score, making them significantly more accurate than
    bi-encoder cosine similarity but more expensive (O(n) forward passes).

    Args:
        model_name: HuggingFace cross-encoder model identifier.
            Default ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~80 MB).
        max_length: Maximum token length for (query, document) pairs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._model = None  # loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(self, query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
        """Rerank documents by cross-encoder relevance score.

        Args:
            query: The original user query.
            docs: Candidate documents from the retriever (any order).
            top_n: Number of top-scoring documents to return.

        Returns:
            Up to ``top_n`` Documents sorted by descending cross-encoder score.
            Returns the input list sliced to ``top_n`` if it is empty or the
            model fails to load (graceful degradation).
        """
        if not docs:
            return []

        if top_n >= len(docs):
            # Nothing to rerank — return all sorted by existing score
            return sorted(docs, key=lambda d: d.score, reverse=True)[:top_n]

        try:
            self._load()
        except Exception as exc:
            logger.error(f"Cross-encoder load failed — falling back to retriever scores: {exc}")
            return sorted(docs, key=lambda d: d.score, reverse=True)[:top_n]

        pairs = [(query, doc.content) for doc in docs]
        try:
            raw_scores: list[float] = self._model.predict(pairs).tolist()  # type: ignore[union-attr]
        except Exception as exc:
            logger.error(f"Cross-encoder predict failed: {exc}")
            return sorted(docs, key=lambda d: d.score, reverse=True)[:top_n]

        scored = sorted(zip(docs, raw_scores), key=lambda x: x[1], reverse=True)
        result: list[Document] = []
        for doc, score in scored[:top_n]:
            doc.score = float(score)
            result.append(doc)

        logger.debug(
            f"Reranker top-{top_n} scores: {[round(d.score, 3) for d in result]}"
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading cross-encoder model: {self._model_name}")
        # Import deferred so the module loads even without sentence-transformers installed
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        self._model = CrossEncoder(self._model_name, max_length=self._max_length)
        logger.info("Cross-encoder ready")
