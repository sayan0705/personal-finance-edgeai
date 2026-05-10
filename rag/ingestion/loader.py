"""Document loader supporting PDF, HTML, plain text, JSONL, and HTTP URLs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from rag.models import Document

_SUPPORTED_SUFFIXES = {".pdf", ".html", ".htm", ".txt", ".jsonl"}


class DocumentLoader:
    """Loads raw documents from disk or HTTP into a list of Document objects.

    Each Document produced here has the full text of the source file and
    metadata describing its origin. The chunker then splits them.

    Supported sources:
        - Local ``.pdf`` files — via ``pypdf``
        - Local ``.html`` / ``.htm`` files — via ``beautifulsoup4``
        - Local ``.jsonl`` files — one JSON object per line
        - Local ``.txt`` files — raw UTF-8 text
        - HTTP/HTTPS URLs pointing to PDFs — streamed to a temp file
        - Local directories — recursively loads all supported files
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> list[Document]:
        """Load documents from a file, directory, or HTTP URL.

        Args:
            path: Local file path, directory path, or ``http(s)://`` URL.
                  Directories are recursively scanned for supported files.

        Returns:
            List of Document objects (pre-chunking, one per page/record).
        """
        path_str = str(path)
        if path_str.startswith("http://") or path_str.startswith("https://"):
            return self.load_url(path_str)

        p = Path(path)
        if p.is_dir():
            docs: list[Document] = []
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in _SUPPORTED_SUFFIXES:
                    docs.extend(self._load_file(child))
            logger.info(f"Loaded {len(docs)} documents from directory {p}")
            return docs
        return self._load_file(p)

    def load_url(
        self,
        url: str,
        timeout: int = 30,
        retries: int = 3,
    ) -> list[Document]:
        """Download a PDF from an HTTP/HTTPS URL and return its pages as Documents.

        The PDF is streamed to a temporary file (deleted after parsing).
        A retry adapter with exponential backoff handles transient network errors.

        Args:
            url: Full HTTP or HTTPS URL to a PDF document.
            timeout: Per-request timeout in seconds.
            retries: Maximum number of retry attempts on network failure.

        Returns:
            List of Documents — one per page — with ``url`` stored as ``source``.

        Raises:
            ValueError: If the URL scheme is not http or https.
            requests.HTTPError: If the server returns a non-2xx status.
        """
        try:
            import requests  # noqa: PLC0415
            from requests.adapters import HTTPAdapter  # noqa: PLC0415
            from urllib3.util.retry import Retry  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("requests is required for URL loading: pip install requests") from exc

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"URL must start with http:// or https://, got: {url!r}")

        # Derive a human-readable filename from the URL path
        url_filename = Path(parsed.path).name or "document"
        if not url_filename.lower().endswith(".pdf"):
            url_filename += ".pdf"

        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=2,
            status_forcelist={429, 500, 502, 503, 504},
            allowed_methods={"GET"},
        )
        session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        session.mount("http://", HTTPAdapter(max_retries=retry_strategy))

        logger.info(f"Downloading PDF: {url}")
        response = session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            logger.warning(
                f"Content-Type is '{content_type}' (expected application/pdf) — "
                f"attempting to parse anyway: {url}"
            )

        total_bytes = int(response.headers.get("Content-Length", 0))

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                downloaded = 0
                chunk_size = 65_536  # 64 KB
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        tmp.write(chunk)
                        downloaded += len(chunk)

            size_kb = downloaded / 1024
            logger.info(
                f"Downloaded {size_kb:.1f} KB"
                + (f" / {total_bytes / 1024:.1f} KB" if total_bytes else "")
                + f" → {tmp_path.name}"
            )

            # Parse, then override source metadata with the original URL
            docs = self.load_pdf(tmp_path)
            for doc in docs:
                doc.metadata["source"] = url
                doc.metadata["url"] = url
                doc.metadata["filename"] = url_filename
                # Replace temp path in id with url-based id
                doc.id = doc.id.replace(str(tmp_path), url)

            logger.info(f"Parsed {len(docs)} pages from {url}")
            return docs

        except Exception as exc:
            logger.error(f"Failed to load URL {url}: {exc}")
            raise
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
                logger.debug(f"Cleaned up temp file {tmp_path.name}")

    # ------------------------------------------------------------------
    # Format-specific loaders
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                return self.load_pdf(path)
            if suffix in {".html", ".htm"}:
                return self.load_html(path)
            if suffix == ".jsonl":
                return self.load_jsonl(path)
            if suffix == ".txt":
                return self.load_txt(path)
            logger.warning(f"Unsupported file type '{suffix}' — skipping {path}")
            return []
        except Exception as exc:
            logger.error(f"Failed to load {path}: {exc}")
            return []

    def load_pdf(self, path: Path) -> list[Document]:
        """Load a PDF file page by page.

        Args:
            path: Path to the PDF file.

        Returns:
            One Document per page with ``page`` metadata.
        """
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("pypdf is required for PDF loading: pip install pypdf") from exc

        reader = PdfReader(str(path))
        docs: list[Document] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    id=f"{path}::p{page_num}",
                    content=text,
                    metadata={
                        "source": str(path),
                        "title": path.stem,
                        "page": page_num,
                        "format": "pdf",
                    },
                )
            )
        logger.debug(f"Loaded {len(docs)} pages from {path}")
        return docs

    def load_html(self, path: Path) -> list[Document]:
        """Extract visible text from an HTML file.

        Args:
            path: Path to the HTML file.

        Returns:
            Single Document with all visible text content.
        """
        try:
            from bs4 import BeautifulSoup  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4") from exc

        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")

        # Remove script / style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.string if soup.title else path.stem

        logger.debug(f"Loaded HTML: {path} ({len(text)} chars)")
        return [
            Document(
                id=str(path),
                content=text,
                metadata={"source": str(path), "title": title, "format": "html"},
            )
        ]

    def load_jsonl(self, path: Path) -> list[Document]:
        """Load a JSONL file where each line is a JSON object.

        Expected schema per line:
            ``{"text": "...", "metadata": {...}}``
            or ``{"content": "...", "metadata": {...}}``

        Args:
            path: Path to the JSONL file.

        Returns:
            One Document per non-empty line.
        """
        docs: list[Document] = []
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(f"{path}:{line_num}: JSON parse error — {exc}")
                    continue

                text = obj.get("text") or obj.get("content") or ""
                if not text:
                    logger.warning(f"{path}:{line_num}: missing 'text'/'content' field — skipping")
                    continue

                meta: dict[str, Any] = obj.get("metadata", {})
                meta.setdefault("source", str(path))
                meta.setdefault("format", "jsonl")

                docs.append(
                    Document(
                        id=f"{path}::{line_num}",
                        content=str(text),
                        metadata=meta,
                    )
                )

        logger.debug(f"Loaded {len(docs)} records from {path}")
        return docs

    def load_txt(self, path: Path) -> list[Document]:
        """Load a plain text file as a single Document.

        Args:
            path: Path to the text file.

        Returns:
            Single Document.
        """
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        logger.debug(f"Loaded TXT: {path} ({len(text)} chars)")
        return [
            Document(
                id=str(path),
                content=text,
                metadata={"source": str(path), "title": path.stem, "format": "txt"},
            )
        ]
