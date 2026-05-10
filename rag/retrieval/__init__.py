"""Retrieval components: hybrid retriever and cross-encoder reranker.

Import directly to avoid requiring rank_bm25 / sentence-transformers at
package import time:

    from rag.retrieval.retriever import HybridRetriever
    from rag.retrieval.reranker import CrossEncoderReranker
"""

__all__ = ["HybridRetriever", "CrossEncoderReranker"]
