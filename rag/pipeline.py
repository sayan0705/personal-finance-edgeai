"""RAG pipeline: ChromaDB vector store → hybrid retrieval → cross-encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from app.api.config import get
from rag.models import Document
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.retriever import HybridRetriever
from rag.vector_store.chroma_store import ChromaVectorStore


@dataclass
class RAGResult:
    """Output of a RAG query.

    Attributes:
        query: The original user query.
        context: Formatted context string ready to inject into the LLM prompt.
        sources: Retrieved and reranked Document objects for citation.
    """

    query: str
    context: str
    sources: list[Document] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        return bool(self.context.strip())


class RAGPipeline:
    """End-to-end RAG retrieval pipeline for FinEdge.

    Components (all lazy-initialised on first query):
        1. ``ChromaVectorStore`` — connects to the ChromaDB HTTP server.
        2. ``HybridRetriever`` — dense (vector) + sparse (BM25) search fused
           via Reciprocal Rank Fusion.
        3. ``CrossEncoderReranker`` — re-scores top candidates with a
           cross-encoder model for higher precision.

    Configuration is read from ``configs/app_config.yaml`` (``rag.*`` keys)
    and the ChromaDB host/port can be overridden via env vars ``CHROMA_HOST``
    and ``CHROMA_PORT``.

    Usage::

        pipeline = RAGPipeline()
        result = pipeline.query_with_sources("How do I save tax under 80C?")
        # result.context → inject into LLM prompt
        # result.sources → display citations in UI
    """

    def __init__(self) -> None:
        self._enabled: bool = str(get("rag.enabled", "false")).lower() == "true"
        self._store: ChromaVectorStore | None = None
        self._retriever: HybridRetriever | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._ready: bool = False

        if not self._enabled:
            logger.info("RAGPipeline: disabled — set rag.enabled=true in config to activate")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, user_query: str) -> str:
        """Return formatted context string for injection into an LLM prompt.

        Args:
            user_query: The user's natural-language question.

        Returns:
            Multi-paragraph context block, or ``""`` if RAG is disabled /
            no relevant documents are found.
        """
        return self.query_with_sources(user_query).context

    def query_with_sources(self, user_query: str) -> RAGResult:
        """Retrieve context and expose the source Documents for citation.

        Args:
            user_query: The user's natural-language question.

        Returns:
            ``RAGResult`` containing the formatted context and source documents.
        """
        if not self._enabled:
            return RAGResult(query=user_query, context="")

        if not self._ensure_ready():
            return RAGResult(query=user_query, context="")

        k: int = int(get("rag.retrieval_k", 10))
        top_n: int = int(get("rag.rerank_top_n", 3))

        try:
            docs = self._retriever.retrieve(user_query, k=k)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error(f"Retrieval failed for query {user_query[:60]!r}: {exc}")
            return RAGResult(query=user_query, context="")

        if not docs:
            logger.debug("Retriever returned 0 documents")
            return RAGResult(query=user_query, context="")

        if self._reranker:
            try:
                docs = self._reranker.rerank(user_query, docs, top_n=top_n)
            except Exception as exc:
                logger.error(f"Reranking failed: {exc} — using retriever ranking")
                docs = docs[:top_n]
        else:
            docs = docs[:top_n]

        context = _format_context(docs)
        logger.debug(f"RAG context: {len(docs)} sources, {len(context)} chars")
        return RAGResult(query=user_query, context=context, sources=docs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """Pre-initialise all components and build the BM25 index.

        Call this at application startup (e.g. FastAPI ``lifespan``) to move
        all model-loading and network I/O out of the first request path.
        No-op when RAG is disabled.
        """
        if not self._enabled:
            return
        self._ensure_ready()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> bool:
        """Initialise components on first call. Returns True on success."""
        if self._ready:
            return True
        try:
            self._init_components()
            self._ready = True
            return True
        except Exception as exc:
            logger.error(f"RAGPipeline initialisation failed: {exc}")
            return False

    def _init_components(self) -> None:
        logger.info("Initialising RAG pipeline components…")

        chroma_host: str = get("rag.chroma.host", "chromadb")
        chroma_port: int = int(get("rag.chroma.port", 8000))
        collection: str = get("rag.chroma.collection", "finedge_finance_docs")
        embedding_model: str = get(
            "rag.embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        bm25_enabled: bool = str(get("rag.bm25.enabled", "true")).lower() == "true"
        reranker_enabled: bool = str(get("rag.reranker.enabled", "true")).lower() == "true"
        reranker_model: str = get(
            "rag.reranker.model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self._store = ChromaVectorStore(
            host=chroma_host,
            port=chroma_port,
            collection_name=collection,
            embedding_model=embedding_model,
        )

        self._retriever = HybridRetriever(
            vector_store=self._store,
            bm25_enabled=bm25_enabled,
        )
        self._retriever.warm_up()

        if reranker_enabled:
            self._reranker = CrossEncoderReranker(model_name=reranker_model)

        logger.info(
            f"RAGPipeline ready — "
            f"chroma={chroma_host}:{chroma_port}/{collection}, "
            f"bm25={bm25_enabled}, reranker={reranker_enabled}"
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _format_context(docs: list[Document]) -> str:
    """Render retrieved documents into an LLM-friendly context block.

    Args:
        docs: Reranked source documents.

    Returns:
        Multi-section text block with numbered citations.
    """
    if not docs:
        return ""

    parts: list[str] = ["### Relevant context from knowledge base\n"]
    for i, doc in enumerate(docs, start=1):
        title = doc.title
        source = doc.source
        header = f"**[{i}] {title}** — *{source}*" if title else f"**[{i}]** *{source}*"
        parts.append(f"{header}\n\n{doc.content.strip()}")

    return "\n\n---\n\n".join(parts)
