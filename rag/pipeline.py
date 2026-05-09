"""RAG pipeline stub — returns empty context until the pipeline is initialised."""

from __future__ import annotations

from loguru import logger

from app.api.config import get


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    Currently a stub. When ``rag.enabled`` is ``false`` in config (default),
    ``query()`` returns an empty string and logs a warning.

    To activate: set ``rag.enabled: true`` in ``configs/app_config.yaml``
    and implement the retriever/reranker in ``rag/retrieval/``.
    """

    def __init__(self) -> None:
        self._enabled: bool = str(get("rag.enabled", "false")).lower() == "true"
        if not self._enabled:
            logger.info("RAGPipeline: disabled — set rag.enabled=true in config to activate")

    def query(self, user_query: str) -> str:
        """Retrieve relevant context for a user query.

        Args:
            user_query: The user's question.

        Returns:
            Formatted context string, or empty string if RAG is disabled.
        """
        if not self._enabled:
            logger.debug("RAGPipeline: not initialized — returning empty context")
            return ""

        # TODO: implement retriever → reranker → format pipeline
        logger.warning("RAGPipeline: enabled but not implemented — returning empty context")
        return ""
