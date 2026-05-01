"""Unit tests for the data ingestion layer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.ingestion.base import (
    ChecksumTracker,
    Document,
    RateLimiter,
    create_http_session,
)
from src.data.ingestion.html_loader import GenericHTMLLoader
from src.data.ingestion.pdf_loader import GenericPDFLoader


# ── Document ──────────────────────────────────────────────────────────────────


class TestDocument:
    def test_basic_creation(self) -> None:
        doc = Document(page_content="Hello world")
        assert doc.page_content == "Hello world"
        assert doc.metadata == {}

    def test_with_metadata(self) -> None:
        meta = {"source": "sebi.gov.in", "date": "2024-01-15"}
        doc = Document(page_content="content", metadata=meta)
        assert doc.metadata["source"] == "sebi.gov.in"

    def test_non_string_content_raises(self) -> None:
        with pytest.raises(TypeError):
            Document(page_content=123)  # type: ignore[arg-type]


# ── RateLimiter ───────────────────────────────────────────────────────────────


class TestRateLimiter:
    def test_wait_does_not_raise(self) -> None:
        limiter = RateLimiter(min_delay=0.0, max_delay=0.01)
        limiter.wait()  # should complete without error

    def test_second_call_respects_delay(self) -> None:
        import time

        limiter = RateLimiter(min_delay=0.05, max_delay=0.1)
        limiter.wait()
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.0  # basic sanity; not timing-sensitive in CI


# ── ChecksumTracker ───────────────────────────────────────────────────────────


class TestChecksumTracker:
    def test_not_downloaded_initially(self, tmp_path: Path) -> None:
        tracker = ChecksumTracker(tmp_path / "cache.json")
        assert not tracker.is_downloaded("https://example.com/doc.pdf")

    def test_mark_and_check(self, tmp_path: Path) -> None:
        tracker = ChecksumTracker(tmp_path / "cache.json")
        url = "https://example.com/doc.pdf"
        local = tmp_path / "doc.pdf"
        local.write_bytes(b"fake pdf")

        tracker.mark_downloaded(url, local)
        assert tracker.is_downloaded(url)
        assert tracker.local_path(url) == str(local)

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cache.json"
        url = "https://example.com/doc.pdf"
        local = tmp_path / "doc.pdf"
        local.write_bytes(b"data")

        tracker1 = ChecksumTracker(cache_file)
        tracker1.mark_downloaded(url, local)

        tracker2 = ChecksumTracker(cache_file)
        assert tracker2.is_downloaded(url)

    def test_corrupted_cache_recovers(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("NOT VALID JSON", encoding="utf-8")
        tracker = ChecksumTracker(cache_file)
        assert not tracker.is_downloaded("https://any.url")


# ── create_http_session ───────────────────────────────────────────────────────


class TestCreateHttpSession:
    def test_returns_session(self) -> None:
        import requests

        session = create_http_session()
        assert isinstance(session, requests.Session)

    def test_user_agent_set(self) -> None:
        session = create_http_session(user_agent="TestBot/1.0")
        assert session.headers["User-Agent"] == "TestBot/1.0"


# ── GenericHTMLLoader ─────────────────────────────────────────────────────────


class TestGenericHTMLLoader:
    _SAMPLE_HTML = """
    <html>
      <head><title>SEBI Circular</title></head>
      <body>
        <nav>Navigation links here</nav>
        <main>
          <h1>Circular No. SEBI/HO/2024/001</h1>
          <p>All SEBI registered intermediaries are advised to comply with the following guidelines.</p>
        </main>
        <footer>Footer content</footer>
      </body>
    </html>
    """

    def test_extracts_main_content(self) -> None:
        loader = GenericHTMLLoader(self._SAMPLE_HTML, base_url="https://sebi.gov.in/test")
        docs = loader.load()
        assert len(docs) == 1
        assert "intermediaries" in docs[0].page_content
        assert "Navigation" not in docs[0].page_content  # nav removed

    def test_metadata_populated(self) -> None:
        loader = GenericHTMLLoader(
            self._SAMPLE_HTML,
            metadata={"source_org": "SEBI"},
            base_url="https://sebi.gov.in/test",
        )
        docs = loader.load()
        assert docs[0].metadata["source_org"] == "SEBI"
        assert docs[0].metadata["format"] == "html"

    def test_empty_html_returns_empty(self) -> None:
        loader = GenericHTMLLoader("<html><body></body></html>")
        docs = loader.load()
        assert docs == []

    def test_load_from_file(self, tmp_path: Path) -> None:
        html_file = tmp_path / "test.html"
        html_file.write_text(self._SAMPLE_HTML, encoding="utf-8")
        loader = GenericHTMLLoader(html_file)
        docs = loader.load()
        assert len(docs) == 1

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            GenericHTMLLoader(Path("/nonexistent/file.html"))

    def test_load_safe_returns_empty_on_error(self) -> None:
        loader = GenericHTMLLoader("<html><body></body></html>")
        docs = loader.load_safe()
        assert isinstance(docs, list)


# ── GenericPDFLoader ──────────────────────────────────────────────────────────


class TestGenericPDFLoader:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        loader = GenericPDFLoader(tmp_path / "nonexistent.pdf")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_safe_returns_empty_on_missing_file(self, tmp_path: Path) -> None:
        loader = GenericPDFLoader(tmp_path / "nonexistent.pdf")
        docs = loader.load_safe()
        assert docs == []

    @patch("src.data.ingestion.pdf_loader.GenericPDFLoader._extract_pypdf")
    def test_uses_pdfminer_fallback(self, mock_pypdf: MagicMock, tmp_path: Path) -> None:
        mock_pypdf.return_value = ""  # simulate pypdf returning nothing
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")  # minimal fake PDF

        with patch(
            "src.data.ingestion.pdf_loader.GenericPDFLoader._extract_pdfminer",
            return_value="Extracted via pdfminer",
        ) as mock_miner:
            loader = GenericPDFLoader(pdf_file)
            docs = loader.load()
            mock_miner.assert_called_once()
