"""Core abstractions for the data ingestion system."""

from __future__ import annotations

import hashlib
import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class Document:
    """A single document with extracted text and associated metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.page_content, str):
            raise TypeError("page_content must be a string")


class RateLimiter:
    """Adds polite, jittered delays between outbound HTTP requests.

    Args:
        min_delay: Minimum seconds to wait between requests.
        max_delay: Maximum seconds to wait between requests.
    """

    def __init__(self, min_delay: float = 1.0, max_delay: float = 3.0) -> None:
        self._min = min_delay
        self._max = max_delay
        self._last_call: float = 0.0

    def wait(self) -> None:
        """Block until the minimum inter-request interval has elapsed."""
        elapsed = time.monotonic() - self._last_call
        delay = random.uniform(self._min, self._max)
        remaining = delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


class ChecksumTracker:
    """Persists a URL→local-path mapping to avoid redundant downloads.

    Args:
        cache_file: Path to the JSON file used as the download cache.
    """

    def __init__(self, cache_file: str | Path) -> None:
        self._cache_file = Path(cache_file)
        self._cache: dict[str, str] = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict[str, str]:
        if self._cache_file.exists():
            try:
                return json.loads(self._cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Could not read download cache ({exc}); starting fresh")
        return {}

    def _save(self) -> None:
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache_file.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── public API ────────────────────────────────────────────────────────────

    def is_downloaded(self, url: str) -> bool:
        """Return True if *url* has already been fetched."""
        return url in self._cache

    def mark_downloaded(self, url: str, local_path: Path) -> None:
        """Record that *url* was saved to *local_path*."""
        self._cache[url] = str(local_path)
        self._save()

    def local_path(self, url: str) -> Optional[str]:
        """Return the cached local path for *url*, or None."""
        return self._cache.get(url)


def create_http_session(
    timeout: int = 30,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    user_agent: str = "FinEdge-DataBot/1.0",
) -> requests.Session:
    """Build a requests.Session with exponential-backoff retry logic.

    Args:
        timeout: Socket timeout in seconds (stored on the session for callers).
        max_retries: Total retry attempts on transient HTTP errors.
        backoff_factor: Multiplier for exponential backoff between retries.
        user_agent: User-Agent header sent with every request.

    Returns:
        Configured requests.Session instance.
    """
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = user_agent
    session.timeout = timeout  # type: ignore[attr-defined]
    return session


class BaseLoader(ABC):
    """Abstract base class that all document loaders must implement."""

    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents and return them as a list.

        Returns:
            List of Document instances. May be empty on failure.
        """

    def load_safe(self) -> List[Document]:
        """Wrapper around load() that catches all exceptions.

        Returns:
            Documents on success, empty list on any error.
        """
        try:
            return self.load()
        except Exception as exc:
            logger.error(f"{self.__class__.__name__}.load() failed: {exc}")
            return []
