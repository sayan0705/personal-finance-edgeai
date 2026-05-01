"""Scraper for SEBI (Securities and Exchange Board of India) circulars.

Target: https://www.sebi.gov.in/legal/circulars/
Each circular is a PDF linked from the listing page. The scraper:
  1. Fetches the listing page and parses circular metadata.
  2. Downloads PDFs that have not been fetched before (checksum cache).
  3. Extracts text via GenericPDFLoader.
  4. Returns Documents tagged with SEBI-specific metadata.

Note: SEBI's website uses dynamic pagination. If JavaScript rendering is
required for newer pages, consider adding a Selenium-based fetch fallback.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

from ..base import BaseLoader, ChecksumTracker, Document, RateLimiter, create_http_session
from ..pdf_loader import GenericPDFLoader


class SEBILoader(BaseLoader):
    """Download and parse SEBI circulars from sebi.gov.in.

    Args:
        config: Section from ``dataset_config.yaml`` under ``sources.public.sebi``.
        session: Optional pre-built requests.Session (useful for testing).
    """

    def __init__(self, config: dict, session: requests.Session | None = None) -> None:
        self._cfg = config
        self._output = Path(config["output_path"])
        self._output.mkdir(parents=True, exist_ok=True)

        http_cfg = config.get("http", {})
        self._session = session or create_http_session(**http_cfg)
        self._timeout = http_cfg.get("timeout", 30)

        self._limiter = RateLimiter(**config.get("rate_limit", {}))
        self._tracker = ChecksumTracker(config.get("checksum_cache", "data/.download_cache.json"))
        self._max_docs = config.get("max_documents", 500)
        self._max_pages = config.get("max_pages", 10)
        self._base_url = config["base_url"]
        self._circulars_url = config["circulars_url"]

    # ── public API ────────────────────────────────────────────────────────────

    def load(self) -> List[Document]:
        """Fetch SEBI circular listing and download PDFs.

        Returns:
            List of Documents, one per successfully parsed circular.
        """
        circular_meta = self._fetch_listing()
        logger.info(f"SEBI: found {len(circular_meta)} circulars; downloading up to {self._max_docs}")

        docs: list[Document] = []
        with tqdm(circular_meta[: self._max_docs], desc="SEBI circulars", unit="doc") as bar:
            for meta in bar:
                result = self._download_and_parse(meta)
                docs.extend(result)
                bar.set_postfix(loaded=len(docs))

        logger.info(f"SEBI: loaded {len(docs)} documents")
        return docs

    # ── private helpers ───────────────────────────────────────────────────────

    def _fetch_listing(self) -> list[dict]:
        """Scrape circular listing pages and return metadata dicts."""
        results: list[dict] = []
        page = 1

        while page <= self._max_pages and len(results) < self._max_docs:
            url = self._circulars_url if page == 1 else f"{self._circulars_url}?page={page}"
            self._limiter.wait()
            try:
                resp = self._session.get(url, timeout=self._timeout)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.error(f"SEBI listing page {page} failed: {exc}")
                break

            new_items = self._parse_listing_page(resp.text, resp.url)
            if not new_items:
                break
            results.extend(new_items)
            page += 1

        return results

    def _parse_listing_page(self, html: str, page_url: str) -> list[dict]:
        """Extract circular metadata from a single listing page."""
        soup = BeautifulSoup(html, "lxml")
        items: list[dict] = []

        # SEBI renders circulars inside <table> rows or <ul> lists; try both.
        for link in soup.find_all("a", href=re.compile(r"\.pdf", re.I)):
            href = link.get("href", "")
            pdf_url = urljoin(self._base_url, href)

            # Walk up to find date/subject in surrounding text
            parent = link.find_parent(["tr", "li", "div", "p"])
            text = parent.get_text(" ", strip=True) if parent else link.get_text(strip=True)

            date = self._extract_date(text)
            circular_no = self._extract_circular_no(text)

            items.append(
                {
                    "pdf_url": pdf_url,
                    "subject": link.get_text(strip=True),
                    "date": date,
                    "circular_no": circular_no,
                    "listing_page": page_url,
                    "source_org": "SEBI",
                }
            )

        return items

    def _download_and_parse(self, meta: dict) -> list[Document]:
        """Download a PDF and extract its text as a Document."""
        url = meta["pdf_url"]
        filename = self._safe_filename(url)
        local_path = self._output / filename

        if self._tracker.is_downloaded(url) and local_path.exists():
            logger.debug(f"SEBI: cache hit {filename}")
        else:
            self._limiter.wait()
            if not self._download_pdf(url, local_path):
                return []
            self._tracker.mark_downloaded(url, local_path)

        loader = GenericPDFLoader(
            local_path,
            metadata={
                "url": url,
                "date": meta.get("date", ""),
                "circular_no": meta.get("circular_no", ""),
                "source_org": "SEBI",
                "subject": meta.get("subject", ""),
                "data_type": "regulatory_circular",
            },
        )
        return loader.load_safe()

    def _download_pdf(self, url: str, dest: Path) -> bool:
        """Stream a PDF from *url* to *dest*. Returns True on success."""
        try:
            with self._session.get(url, stream=True, timeout=self._timeout) as resp:
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            logger.debug(f"SEBI: downloaded {dest.name}")
            return True
        except requests.RequestException as exc:
            logger.error(f"SEBI: download failed for {url}: {exc}")
            return False

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _safe_filename(url: str) -> str:
        name = url.split("/")[-1].split("?")[0]
        return re.sub(r"[^\w.\-]", "_", name) or "sebi_circular.pdf"

    @staticmethod
    def _extract_date(text: str) -> str:
        match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},?\s*\d{4})\b", text)
        return match.group(0) if match else ""

    @staticmethod
    def _extract_circular_no(text: str) -> str:
        match = re.search(r"(SEBI/[A-Z]+/[A-Z]+/\d+/\d+|Circular No\.?\s*[\w/\-]+)", text, re.I)
        return match.group(0).strip() if match else ""
