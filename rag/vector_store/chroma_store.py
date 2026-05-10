"""ChromaDB HTTP vector store wrapper."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger

from rag.models import Document


class ChromaVectorStore:
    """Thin wrapper around a ChromaDB HTTP client for a single collection.

    Connects lazily on first use so the object is cheap to construct and
    retries are handled externally (e.g. Docker healthcheck on the service).

    Args:
        host: ChromaDB server hostname (default ``chromadb`` inside Docker).
        port: ChromaDB server port (default ``8000``).
        collection_name: ChromaDB collection to use.
        embedding_model: HuggingFace sentence-transformer model name used
            as the collection's embedding function.
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._client: chromadb.HttpClient | None = None
        self._collection: Any | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish connection to ChromaDB server and bind collection."""
        if self._client is not None:
            return

        logger.info(f"Connecting to ChromaDB at {self._host}:{self._port}")
        self._client = chromadb.HttpClient(host=self._host, port=int(self._port))

        ef = SentenceTransformerEmbeddingFunction(
            model_name=self._embedding_model,
            normalize_embeddings=True,
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        count = self._collection.count()
        logger.info(
            f"ChromaDB ready — collection '{self._collection_name}' "
            f"contains {count} chunks"
        )

    @property
    def _col(self) -> Any:
        self._connect()
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return number of chunks stored in the collection."""
        return self._col.count()

    def query(self, query_text: str, n_results: int = 10) -> list[Document]:
        """Vector-similarity search.

        Args:
            query_text: The query string to embed and search.
            n_results: Maximum number of results to return.

        Returns:
            Documents sorted by descending cosine similarity.

        Raises:
            RuntimeError: If ChromaDB returns an unexpected response shape.
        """
        total = self._col.count()
        if total == 0:
            logger.warning("ChromaDB collection is empty — returning no results")
            return []

        effective_n = min(n_results, total)
        try:
            results = self._col.query(
                query_texts=[query_text],
                n_results=effective_n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error(f"ChromaDB query error: {exc}")
            raise

        docs: list[Document] = []
        ids = results["ids"][0]
        texts = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc_id, content, meta, dist in zip(ids, texts, metas, dists):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            similarity = 1.0 - dist
            docs.append(
                Document(id=doc_id, content=content, metadata=meta or {}, score=similarity)
            )

        logger.debug(f"Vector search returned {len(docs)} docs (top score={docs[0].score:.4f})" if docs else "Vector search returned 0 docs")
        return docs

    def get_all(self) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Fetch every document in the collection for BM25 index construction.

        Returns:
            Tuple of (ids, texts, metadatas).
        """
        try:
            result = self._col.get(include=["documents", "metadatas"])
            ids: list[str] = result["ids"]
            texts: list[str] = result["documents"] or []
            metas: list[dict[str, Any]] = result["metadatas"] or []
            logger.debug(f"Fetched {len(ids)} documents from ChromaDB for BM25 index")
            return ids, texts, metas
        except Exception as exc:
            logger.error(f"ChromaDB get_all error: {exc}")
            raise

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Upsert documents into the collection.

        Args:
            ids: Unique chunk identifiers.
            documents: Raw text of each chunk.
            metadatas: Per-chunk metadata dicts.
            embeddings: Pre-computed embeddings (optional; ChromaDB will
                compute them using the collection's embedding function if omitted).
        """
        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        try:
            self._col.upsert(**kwargs)
            logger.info(f"Upserted {len(ids)} chunks into collection '{self._collection_name}'")
        except Exception as exc:
            logger.error(f"ChromaDB upsert error: {exc}")
            raise

    def delete_collection(self) -> None:
        """Drop and recreate the collection (destructive — use offline only)."""
        if self._client is None:
            self._connect()
        logger.warning(f"Deleting collection '{self._collection_name}'")
        self._client.delete_collection(self._collection_name)  # type: ignore[union-attr]
        self._collection = None
        self._connect()
