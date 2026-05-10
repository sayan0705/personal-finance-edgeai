"""FinEdge RAG package.

Public surface:
    Document      — shared chunk dataclass used throughout the pipeline.
    RAGPipeline   — query-time retrieval pipeline (import from rag.pipeline).
    RAGResult     — structured result with context + source citations.

RAGPipeline and RAGResult are NOT imported here to avoid pulling in
retrieval-stack dependencies (rank_bm25, sentence-transformers, chromadb)
into ingestion-only scripts. Import them directly when needed:

    from rag.pipeline import RAGPipeline, RAGResult
"""

from rag.models import Document

__all__ = ["Document"]
