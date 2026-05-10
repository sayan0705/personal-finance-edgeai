"""Document ingestion pipeline for offline ChromaDB population.

Import submodules directly to avoid pulling in heavy optional deps at
package import time:

    from rag.ingestion.loader import DocumentLoader
    from rag.ingestion.chunker import SemanticChunker
    from rag.ingestion.embedder import EmbeddingIndexer   # needs chromadb
"""

__all__ = ["DocumentLoader", "SemanticChunker", "EmbeddingIndexer"]
