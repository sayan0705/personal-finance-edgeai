"""Main pipeline class — orchestrates extraction, tagging, quality gates, and export."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from .extractor import assign_layer, extract_dataset, run_quality_gates
from .registry import DATASET_REGISTRY

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert financial analyst and Indian personal finance advisor. "
    "Provide accurate, well-reasoned responses with relevant calculations and "
    "regulatory references where applicable."
)


class HFDatasetPipeline:
    """End-to-end pipeline to extract, curate, and export real financial datasets.

    Pulls random samples from 13 open-source HuggingFace financial datasets,
    normalises them into the finance-alpaca ChatML format, applies quality gates
    (dedup, length, safety, diversity), and saves the result as JSONL.

    Example::

        pipeline = HFDatasetPipeline.from_config("configs/dataset_config.yaml")
        pipeline.run()

    Args:
        output_dir: Directory where JSONL and report files are written.
        samples_per_dataset: Target sample count for regular-sized datasets.
        max_samples_large: Target sample count for very large datasets (500K+).
        diversity_cap_pct: Max fraction any single source may occupy.
        seed: Random seed for reproducible sampling and shuffling.
        system_prompt: Default system-role content injected into each record.
        enabled_datasets: Optional subset of DATASET_REGISTRY keys to run.
                          Defaults to all registered datasets.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/raw/hf",
        samples_per_dataset: int = 500,
        max_samples_large: int = 2000,
        diversity_cap_pct: float = 0.30,
        seed: int = 42,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        enabled_datasets: Optional[list[str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.samples_per_dataset = samples_per_dataset
        self.max_samples_large = max_samples_large
        self.diversity_cap_pct = diversity_cap_pct
        self.seed = seed
        self.system_prompt = system_prompt
        self.enabled_datasets = enabled_datasets or list(DATASET_REGISTRY.keys())

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path) -> "HFDatasetPipeline":
        """Instantiate from a dataset_config.yaml file.

        Reads the ``huggingface`` top-level key and falls back to defaults for
        any missing fields.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Configured HFDatasetPipeline instance.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        hf_cfg = cfg.get("huggingface", {})

        return cls(
            output_dir=hf_cfg.get("output_path", "data/raw/hf"),
            samples_per_dataset=hf_cfg.get("samples_per_dataset", 500),
            max_samples_large=hf_cfg.get("max_samples_large", 2000),
            diversity_cap_pct=hf_cfg.get("diversity_cap_pct", 0.30),
            seed=hf_cfg.get("random_seed", 42),
            system_prompt=hf_cfg.get("system_prompt", _DEFAULT_SYSTEM_PROMPT),
            enabled_datasets=hf_cfg.get("enabled_datasets", None),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Path:
        """Execute the full extraction pipeline.

        Steps:
            1. Extract and convert samples from all enabled datasets.
            2. Assign layer tags (L1–L4).
            3. Run quality gates (dedup, length, safety, diversity).
            4. Export curated JSONL + metadata report.

        Returns:
            Path to the exported JSONL file.
        """
        logger.info(
            f"HFDatasetPipeline starting — "
            f"{len(self.enabled_datasets)} datasets, "
            f"~{self._planned_total():,} planned samples"
        )

        # Step 1: Extract
        all_samples, extraction_log = self._extract_all()

        if not all_samples:
            logger.error("No samples extracted — aborting pipeline")
            raise RuntimeError("Extraction produced zero samples")

        # Step 2: Layer tagging
        logger.info("Assigning layer tags…")
        for s in all_samples:
            s["layer"] = assign_layer(s)
        self._log_layer_distribution(all_samples)

        # Step 3: Quality gates
        final_samples, gate_stats = run_quality_gates(
            all_samples, diversity_cap_pct=self.diversity_cap_pct, seed=self.seed
        )

        # Step 4: Export
        output_path = self._export(final_samples, extraction_log, gate_stats, len(all_samples))

        logger.info(f"Pipeline complete — {len(final_samples):,} curated samples → {output_path}")
        return output_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _planned_total(self) -> int:
        is_large = lambda k: "500k" in k.lower() or "sujet" in k.lower()
        return sum(
            self.max_samples_large if is_large(k) else self.samples_per_dataset
            for k in self.enabled_datasets
        )

    def _extract_all(self) -> tuple[list[dict[str, Any]], dict[str, dict]]:
        all_samples: list[dict[str, Any]] = []
        extraction_log: dict[str, dict] = {}

        for ds_key in self.enabled_datasets:
            if ds_key not in DATASET_REGISTRY:
                logger.warning(f"Unknown dataset key '{ds_key}' — skipping")
                continue

            samples, log_entry = extract_dataset(
                ds_key=ds_key,
                system_prompt=self.system_prompt,
                samples_per_dataset=self.samples_per_dataset,
                max_samples_large=self.max_samples_large,
                seed=self.seed,
            )
            all_samples.extend(samples)
            extraction_log[ds_key] = log_entry

        logger.info(f"Total extracted across all sources: {len(all_samples):,}")
        return all_samples, extraction_log

    def _log_layer_distribution(self, samples: list[dict[str, Any]]) -> None:
        dist = Counter(s["layer"] for s in samples)
        for layer, count in sorted(dist.items()):
            logger.info(f"  {layer:<35} {count:>6,}  ({count/len(samples)*100:.1f}%)")

    def _export(
        self,
        final_samples: list[dict[str, Any]],
        extraction_log: dict[str, dict],
        gate_stats: dict[str, int],
        raw_total: int,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = self.output_dir / "indian_finance_real_curated.jsonl"
        report_path = self.output_dir / "real_dataset_report.json"

        # Export JSONL
        with jsonl_path.open("w", encoding="utf-8") as f:
            for s in final_samples:
                f.write(json.dumps(self._clean_for_export(s), ensure_ascii=False) + "\n")

        size_mb = jsonl_path.stat().st_size / (1024 * 1024)
        logger.info(f"Exported JSONL: {jsonl_path} ({size_mb:.2f} MB)")

        # Export metadata report
        report = {
            "dataset_name": "IndianFinanceSLM-RealCurated",
            "version": "0.1.0",
            "created": datetime.now().isoformat(),
            "total_samples": len(final_samples),
            "sources_attempted": len(extraction_log),
            "sources_successful": sum(1 for v in extraction_log.values() if v.get("converted", 0) > 0),
            "layer_distribution": dict(Counter(s["layer"] for s in final_samples)),
            "task_distribution": dict(Counter(s["task_type"] for s in final_samples)),
            "source_distribution": dict(Counter(s["source_dataset"] for s in final_samples)),
            "quality_gates": gate_stats,
            "extraction_log": extraction_log,
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported report: {report_path}")
        return jsonl_path

    @staticmethod
    def _clean_for_export(sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": sample["id"],
            "source_dataset": sample["source_dataset"],
            "task_type": sample["task_type"],
            "layer": sample["layer"],
            "language": sample["language"],
            "messages": sample["messages"],
        }

    # ── Optional: merge with synthetic dataset ────────────────────────────────

    def merge_with_synthetic(self, synthetic_path: str | Path) -> Optional[Path]:
        """Merge the curated real dataset with a synthetic JSONL file (if it exists).

        The merged file is written next to the real dataset JSONL.

        Args:
            synthetic_path: Path to the synthetic JSONL file produced by the
                            synthetic generator notebook/script.

        Returns:
            Path to the merged JSONL file, or ``None`` if synthetic_path does
            not exist.
        """
        import random as _random

        synthetic_path = Path(synthetic_path)
        real_path = self.output_dir / "indian_finance_real_curated.jsonl"

        if not synthetic_path.exists():
            logger.info(f"Synthetic dataset not found at {synthetic_path} — skipping merge")
            return None

        if not real_path.exists():
            logger.error("Real curated JSONL not found — run pipeline.run() first")
            return None

        real_samples: list[dict] = []
        with real_path.open(encoding="utf-8") as f:
            for line in f:
                real_samples.append(json.loads(line))

        synthetic_samples: list[dict] = []
        with synthetic_path.open(encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                s.setdefault("layer", "L3_synthetic")
                s.setdefault("source_dataset", "synthetic")
                s.setdefault("task_type", "instruction")
                s.setdefault("language", "en")
                synthetic_samples.append(s)

        merged = real_samples + [
            {
                "id": s.get("id", f"syn_{i:06d}"),
                "source_dataset": s["source_dataset"],
                "task_type": s["task_type"],
                "layer": s["layer"],
                "language": s["language"],
                "messages": s["messages"],
            }
            for i, s in enumerate(synthetic_samples)
        ]
        _random.Random(self.seed).shuffle(merged)

        merged_path = self.output_dir / "indian_finance_hybrid_merged.jsonl"
        with merged_path.open("w", encoding="utf-8") as f:
            for s in merged:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger.info(
            f"Merged {len(real_samples):,} real + {len(synthetic_samples):,} synthetic "
            f"→ {len(merged):,} samples at {merged_path}"
        )
        return merged_path
