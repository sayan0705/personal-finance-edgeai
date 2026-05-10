"""Text chunker that splits Documents into overlapping chunks for indexing."""

from __future__ import annotations

from loguru import logger

from rag.models import Document


class SemanticChunker:
    """Splits Documents into overlapping chunks using LangChain's recursive splitter.

    Preserves all metadata from the parent Document and appends ``chunk_index``
    and ``total_chunks`` fields so the origin of each chunk is traceable.

    Args:
        chunk_size: Target size in characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
        separators: Ordered list of separators tried by the recursive splitter.
            Defaults to paragraph → newline → sentence → word → character.
    """

    _DEFAULT_SEPARATORS = ["\n\n", "\n", "। ", ". ", " ", ""]  # includes Devanagari danda

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or self._DEFAULT_SEPARATORS
        self._splitter = None  # lazy import

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split a list of Documents into smaller chunks.

        Args:
            documents: Source documents (e.g. from DocumentLoader).

        Returns:
            List of chunk Documents, each with inherited + updated metadata.
        """
        splitter = self._get_splitter()
        chunks: list[Document] = []

        for doc in documents:
            raw_chunks = splitter.split_text(doc.content)
            total = len(raw_chunks)
            for idx, text in enumerate(raw_chunks):
                if not text.strip():
                    continue
                meta = {**doc.metadata, "chunk_index": idx, "total_chunks": total}
                chunk_id = f"{doc.id}::c{idx}"
                chunks.append(Document(id=chunk_id, content=text, metadata=meta))

        logger.info(
            f"Chunked {len(documents)} documents → {len(chunks)} chunks "
            f"(size={self._chunk_size}, overlap={self._chunk_overlap})"
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_splitter(self):
        if self._splitter is not None:
            return self._splitter
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415
        except ImportError:
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "langchain-text-splitters is required: pip install langchain-text-splitters"
                ) from exc

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=self._separators,
            length_function=len,
        )
        return self._splitter
