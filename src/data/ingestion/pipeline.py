"""Top-level orchestrator for all public data ingestion sources."""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from loguru import logger

from .base import Document
from .sources.income_tax import IncomeTaxLoader
from .sources.rbi import RBILoader
from .sources.sebi import SEBILoader

_SOURCE_REGISTRY: dict[str, type] = {
    "sebi": SEBILoader,
    "rbi": RBILoader,
    "income_tax": IncomeTaxLoader,
}


class DataIngestionPipeline:
    """Orchestrates loading from all configured public data sources.

    Example::

        pipeline = DataIngestionPipeline.from_config("configs/dataset_config.yaml")
        docs = pipeline.run(sources=["sebi", "rbi"])

    Args:
        source_configs: Mapping of source name → source-level config dict.
        http_config: Shared HTTP settings (timeout, retries, user_agent).
        checksum_cache: Path to the shared download-cache JSON file.
    """

    def __init__(
        self,
        source_configs: dict[str, dict],
        http_config: dict | None = None,
        checksum_cache: str = "data/.download_cache.json",
    ) -> None:
        self._source_configs = source_configs
        self._http = http_config or {}
        self._cache = checksum_cache

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DataIngestionPipeline":
        """Instantiate from a dataset_config.yaml file.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Configured DataIngestionPipeline instance.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        return cls(
            source_configs=cfg.get("sources", {}).get("public", {}),
            http_config=cfg.get("http", {}),
            checksum_cache=cfg.get("checksum_cache", "data/.download_cache.json"),
        )

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, sources: list[str] | None = None) -> List[Document]:
        """Run ingestion for the specified sources (or all if None).

        Args:
            sources: Names of sources to run (e.g. ``["sebi", "rbi"]``).
                     Defaults to all sources in the config.

        Returns:
            Combined list of Documents from all sources.
        """
        enabled = sources or list(_SOURCE_REGISTRY.keys())
        all_docs: list[Document] = []

        for name in enabled:
            if name not in _SOURCE_REGISTRY:
                logger.warning(f"Unknown source '{name}'; skipping")
                continue
            if name not in self._source_configs:
                logger.warning(f"No config found for source '{name}'; skipping")
                continue

            loader_cls = _SOURCE_REGISTRY[name]
            cfg = {
                **self._source_configs[name],
                "http": self._http,
                "checksum_cache": self._cache,
            }

            logger.info(f"Starting ingestion: {name}")
            loader = loader_cls(cfg)
            docs = loader.load_safe()
            logger.info(f"{name}: ingested {len(docs)} documents")
            all_docs.extend(docs)

        logger.info(f"Total documents ingested: {len(all_docs)}")
        return all_docs

    def save(self, docs: List[Document], output_dir: str | Path) -> None:
        """Persist Documents as individual text files for debugging/inspection.

        Args:
            docs: Documents to save.
            output_dir: Directory where .txt files are written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, doc in enumerate(docs):
            source_org = doc.metadata.get("source_org", "unknown")
            fname = f"{source_org}_{i:04d}.txt"
            (output_dir / fname).write_text(doc.page_content, encoding="utf-8")

        logger.info(f"Saved {len(docs)} documents to {output_dir}")
