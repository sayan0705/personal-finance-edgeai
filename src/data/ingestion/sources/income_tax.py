"""Scraper for Income Tax India content.

Targets:
  - IT Act sections: https://www.incometaxindia.gov.in/pages/acts/income-tax-act.aspx
  - Circulars:       https://www.incometaxindia.gov.in/pages/communications/circulars.aspx

The Income Tax India website serves most content as HTML, with some circulars
linked as PDFs. The scraper fetches both and returns Documents tagged with
section numbers and regulatory metadata.
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


class IncomeTaxLoader(BaseLoader):
    """Download and parse Income Tax India Act sections and circulars.

    Args:
        config: Section from ``dataset_config.yaml`` under ``sources.public.income_tax``.
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
        self._max_docs = config.get("max_documents", 200)
        self._base_url = config["base_url"]

        self._entry_urls = [
            config.get("act_url", ""),
            config.get("circulars_url", ""),
        ]

    # ── public API ────────────────────────────────────────────────────────────

    def load(self) -> List[Document]:
        """Fetch IT Act content and circulars.

        Returns:
            List of Documents tagged with section/circular metadata.
        """
        all_meta: list[dict] = []
        for entry_url in self._entry_urls:
            if entry_url:
                all_meta.extend(self._fetch_listing(entry_url))

        logger.info(
            f"IncomeTax: found {len(all_meta)} entries; loading up to {self._max_docs}"
        )

        docs: list[Document] = []
        with tqdm(all_meta[: self._max_docs], desc="IncomeTax docs", unit="doc") as bar:
            for meta in bar:
                result = self._load_document(meta)
                docs.extend(result)
                bar.set_postfix(loaded=len(docs))

        logger.info(f"IncomeTax: loaded {len(docs)} documents")
        return docs

    # ── private helpers ───────────────────────────────────────────────────────

    def _fetch_listing(self, url: str) -> list[dict]:
        """Fetch one entry/listing page and extract document links."""
        self._limiter.wait()
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"IncomeTax: listing fetch failed for {url}: {exc}")
            return []

        return self._parse_listing(resp.text, url)

    def _parse_listing(self, html: str, page_url: str) -> list[dict]:
        """Extract document links from a listing or act-index page."""
        soup = BeautifulSoup(html, "lxml")
        items: list[dict] = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if not href or href.startswith("#") or href.startswith("mailto"):
                continue

            full_url = urljoin(self._base_url, href)
            is_pdf = bool(re.search(r"\.pdf", href, re.I))
            is_html = bool(re.search(r"\.(aspx|html|htm)", href, re.I))

            if not (is_pdf or is_html):
                continue

            text = link.get_text(strip=True)
            if len(text) < 3:
                continue

            parent = link.find_parent(["tr", "li", "div", "p"]) or link
            context = parent.get_text(" ", strip=True)

            items.append(
                {
                    "doc_url": full_url,
                    "is_pdf": is_pdf,
                    "subject": text,
                    "section_no": self._extract_section_no(context),
                    "circular_no": self._extract_circular_no(context),
                    "date": self._extract_date(context),
                    "listing_page": page_url,
                    "source_org": "IncomeTaxIndia",
                }
            )

        return items

    def _load_document(self, meta: dict) -> list[Document]:
        url = meta["doc_url"]
        if meta["is_pdf"]:
            return self._load_pdf(url, meta)
        return self._load_html_page(url, meta)

    def _load_pdf(self, url: str, meta: dict) -> list[Document]:
        filename = self._safe_filename(url, ".pdf")
        local_path = self._output / filename

        if not (self._tracker.is_downloaded(url) and local_path.exists()):
            self._limiter.wait()
            if not self._download_file(url, local_path):
                return []
            self._tracker.mark_downloaded(url, local_path)

        loader = GenericPDFLoader(local_path, metadata=self._doc_metadata(url, meta))
        return loader.load_safe()

    def _load_html_page(self, url: str, meta: dict) -> list[Document]:
        filename = self._safe_filename(url, ".html")
        local_path = self._output / filename

        if self._tracker.is_downloaded(url) and local_path.exists():
            loader = GenericHTMLLoader(
                local_path, metadata=self._doc_metadata(url, meta), base_url=url
            )
            return loader.load_safe()

        self._limiter.wait()
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"IncomeTax: fetch failed for {url}: {exc}")
            return []

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
            logger.debug(f"IncomeTax: downloaded {dest.name}")
            return True
        except requests.RequestException as exc:
            logger.error(f"IncomeTax: download failed for {url}: {exc}")
            return False

    @staticmethod
    def _doc_metadata(url: str, meta: dict) -> dict:
        return {
            "url": url,
            "date": meta.get("date", ""),
            "section_no": meta.get("section_no", ""),
            "circular_no": meta.get("circular_no", ""),
            "source_org": "IncomeTaxIndia",
            "subject": meta.get("subject", ""),
            "data_type": "tax_regulation",
        }

    @staticmethod
    def _safe_filename(url: str, suffix: str = "") -> str:
        name = url.rstrip("/").split("/")[-1].split("?")[0]
        name = re.sub(r"[^\w.\-]", "_", name)
        if suffix and not name.endswith(suffix):
            name += suffix
        return name or f"it_doc{suffix}"

    @staticmethod
    def _extract_date(text: str) -> str:
        match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},?\s*\d{4})\b", text)
        return match.group(0) if match else ""

    @staticmethod
    def _extract_section_no(text: str) -> str:
        match = re.search(r"\bSection\s+(\d+[A-Z]*(?:\(\w+\))*)\b", text, re.I)
        return f"Section {match.group(1)}" if match else ""

    @staticmethod
    def _extract_circular_no(text: str) -> str:
        match = re.search(r"\bCircular\s+No\.?\s*(\d+/\d{4}|\d+)\b", text, re.I)
        return f"Circular No. {match.group(1)}" if match else ""
