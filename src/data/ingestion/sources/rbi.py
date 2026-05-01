"""Scraper for RBI (Reserve Bank of India) master circulars and directions.

Targets:
  - Master Circulars: https://www.rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx
  - Master Directions: https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx

Each page lists documents that may link to PDFs or HTML pages. The scraper
normalises both formats into Documents tagged with RBI-specific metadata.
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
from ..html_loader import GenericHTMLLoader
from ..pdf_loader import GenericPDFLoader


class RBILoader(BaseLoader):
    """Download and parse RBI master circulars and master directions.

    Args:
        config: Section from ``dataset_config.yaml`` under ``sources.public.rbi``.
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
        self._max_docs = config.get("max_documents", 300)
        self._base_url = config["base_url"]

        self._listing_urls = [
            config.get("master_circulars_url", ""),
            config.get("master_directions_url", ""),
        ]

    # ── public API ────────────────────────────────────────────────────────────

    def load(self) -> List[Document]:
        """Fetch RBI documents from all configured listing URLs.

        Returns:
            List of Documents, one per successfully parsed document.
        """
        all_meta: list[dict] = []
        for listing_url in self._listing_urls:
            if not listing_url:
                continue
            all_meta.extend(self._fetch_listing(listing_url))

        logger.info(f"RBI: found {len(all_meta)} entries; loading up to {self._max_docs}")

        docs: list[Document] = []
        with tqdm(all_meta[: self._max_docs], desc="RBI documents", unit="doc") as bar:
            for meta in bar:
                result = self._load_document(meta)
                docs.extend(result)
                bar.set_postfix(loaded=len(docs))

        logger.info(f"RBI: loaded {len(docs)} documents")
        return docs

    # ── private helpers ───────────────────────────────────────────────────────

    def _fetch_listing(self, url: str) -> list[dict]:
        """Scrape a RBI listing page and return document metadata."""
        self._limiter.wait()
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"RBI: listing fetch failed for {url}: {exc}")
            return []

        return self._parse_listing_page(resp.text, url)

    def _parse_listing_page(self, html: str, page_url: str) -> list[dict]:
        """Extract document links and metadata from a listing page HTML."""
        soup = BeautifulSoup(html, "lxml")
        items: list[dict] = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if not href or href.startswith("#"):
                continue

            full_url = urljoin(self._base_url, href)
            is_pdf = re.search(r"\.pdf", href, re.I)
            is_html_doc = re.search(r"\.(aspx|html|htm)", href, re.I)

            if not (is_pdf or is_html_doc):
                continue

            parent = link.find_parent(["tr", "li", "div", "p"]) or link
            context = parent.get_text(" ", strip=True)

            items.append(
                {
                    "doc_url": full_url,
                    "is_pdf": bool(is_pdf),
                    "subject": link.get_text(strip=True),
                    "date": self._extract_date(context),
                    "doc_no": self._extract_doc_no(context),
                    "listing_page": page_url,
                    "source_org": "RBI",
                }
            )

        return items

    def _load_document(self, meta: dict) -> list[Document]:
        """Fetch and parse a single RBI document (PDF or HTML)."""
        url = meta["doc_url"]

        if meta["is_pdf"]:
            return self._load_pdf(url, meta)
        return self._load_html_page(url, meta)

    def _load_pdf(self, url: str, meta: dict) -> list[Document]:
        filename = self._safe_filename(url, suffix=".pdf")
        local_path = self._output / filename

        if not (self._tracker.is_downloaded(url) and local_path.exists()):
            self._limiter.wait()
            if not self._download_file(url, local_path):
                return []
            self._tracker.mark_downloaded(url, local_path)

        loader = GenericPDFLoader(local_path, metadata=self._doc_metadata(url, meta))
        return loader.load_safe()

    def _load_html_page(self, url: str, meta: dict) -> list[Document]:
        if self._tracker.is_downloaded(url):
            filename = self._safe_filename(url, suffix=".html")
            local_path = self._output / filename
            if local_path.exists():
                loader = GenericHTMLLoader(
                    local_path, metadata=self._doc_metadata(url, meta), base_url=url
                )
                return loader.load_safe()

        self._limiter.wait()
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"RBI: HTML fetch failed for {url}: {exc}")
            return []

        # Save raw HTML for reproducibility
        filename = self._safe_filename(url, suffix=".html")
        local_path = self._output / filename
        local_path.write_text(resp.text, encoding="utf-8")
        self._tracker.mark_downloaded(url, local_path)

        loader = GenericHTMLLoader(
            resp.text,
            metadata=self._doc_metadata(url, meta),
            base_url=url,
        )
        return loader.load_safe()

    def _download_file(self, url: str, dest: Path) -> bool:
        try:
            with self._session.get(url, stream=True, timeout=self._timeout) as resp:
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            logger.debug(f"RBI: downloaded {dest.name}")
            return True
        except requests.RequestException as exc:
            logger.error(f"RBI: download failed for {url}: {exc}")
            return False

    @staticmethod
    def _doc_metadata(url: str, meta: dict) -> dict:
        return {
            "url": url,
            "date": meta.get("date", ""),
            "doc_no": meta.get("doc_no", ""),
            "source_org": "RBI",
            "subject": meta.get("subject", ""),
            "data_type": "regulatory_guideline",
        }

    @staticmethod
    def _safe_filename(url: str, suffix: str = "") -> str:
        name = url.rstrip("/").split("/")[-1].split("?")[0]
        name = re.sub(r"[^\w.\-]", "_", name)
        if suffix and not name.endswith(suffix):
            name += suffix
        return name or f"rbi_doc{suffix}"

    @staticmethod
    def _extract_date(text: str) -> str:
        match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},?\s*\d{4})\b", text)
        return match.group(0) if match else ""

    @staticmethod
    def _extract_doc_no(text: str) -> str:
        match = re.search(
            r"(RBI/\d{4}-\d{2,4}/\d+|DBOD\.No\.\S+|Master Circular\s+[\w/\-]+)", text, re.I
        )
        return match.group(0).strip() if match else ""
