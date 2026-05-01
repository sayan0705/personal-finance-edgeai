"""GapFillerPipeline — orchestrates all 12 gap-filling synthetic generators."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from tqdm import tqdm

from .gap_generators import (
    BankStatementGenerator,
    CASStatementGenerator,
    HinglishGenerator,
    HRAExemptionGenerator,
    InsuranceAdvisoryGenerator,
    RegulatoryQAGenerator,
    RetirementSchemesGenerator,
    Section80CGenerator,
    TaxRegimeComparisonGenerator,
    UPICategorizerGenerator,
    XIRRPortfolioGenerator,
)
from .validators import DataQualityValidator

# Ordered list of (key, GeneratorClass) — keys are used in enabled_generators config
_ALL_GENERATORS: list[tuple[str, type]] = [
    ("bank_stmt", BankStatementGenerator),
    ("tax_regime", TaxRegimeComparisonGenerator),
    ("hra_exemption", HRAExemptionGenerator),
    ("section_80c", Section80CGenerator),
    ("xirr_portfolio", XIRRPortfolioGenerator),
    ("cas_statement", CASStatementGenerator),
    ("hinglish", HinglishGenerator),
    ("upi_categorizer", UPICategorizerGenerator),
    ("regulatory_qa", RegulatoryQAGenerator),
    ("insurance_advisory", InsuranceAdvisoryGenerator),
    ("retirement_schemes", RetirementSchemesGenerator),
]


class GapFillerPipeline:
    """End-to-end pipeline that runs all gap generators, applies quality gates,
    and exports a curated JSONL file ready to merge with the real HF dataset.

    Args:
        output_dir: Directory where the output JSONL and report are written.
        samples_per_gap: Number of samples each generator should produce.
        seed: Global random seed for reproducibility.
        enabled_generators: Subset of generator keys to run; ``None`` means all.

    Example::

        pipeline = GapFillerPipeline.from_config("configs/dataset_config.yaml")
        output_path = pipeline.run()
        merged = pipeline.merge_with_real("data/raw/hf/indian_finance_real_curated.jsonl")
    """

    DEFAULT_OUTPUT = "data/raw/synthetic"
    DEFAULT_SAMPLES = 300
    DEFAULT_SEED = 42

    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT,
        samples_per_gap: int = DEFAULT_SAMPLES,
        seed: int = DEFAULT_SEED,
        enabled_generators: list[str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.samples_per_gap = samples_per_gap
        self.seed = seed
        self.enabled_generators: list[str] = (
            enabled_generators if enabled_generators is not None
            else [key for key, _ in _ALL_GENERATORS]
        )

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path) -> "GapFillerPipeline":
        """Instantiate from a ``gap_filler:`` section in ``dataset_config.yaml``.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Configured GapFillerPipeline.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        gf = cfg.get("gap_filler", {})
        return cls(
            output_dir=gf.get("output_path", cls.DEFAULT_OUTPUT),
            samples_per_gap=gf.get("samples_per_gap", cls.DEFAULT_SAMPLES),
            seed=gf.get("random_seed", cls.DEFAULT_SEED),
            enabled_generators=gf.get("enabled_generators"),
        )

    # ── public API ────────────────────────────────────────────────────────────

    def run(self) -> Path:
        """Execute all enabled generators, apply quality gates, and export JSONL.

        Returns:
            Path to the exported JSONL file.
        """
        rng = random.Random(self.seed)
        validator = DataQualityValidator(
            min_answer_length=50,
            max_answer_length=4000,
            skip_pii=True,
        )

        enabled_set = set(self.enabled_generators)
        gen_instances = [
            (key, cls(self.samples_per_gap, rng))
            for key, cls in _ALL_GENERATORS
            if key in enabled_set
        ]

        logger.info(
            f"GapFillerPipeline: {len(gen_instances)} generators × "
            f"{self.samples_per_gap} samples = "
            f"~{len(gen_instances) * self.samples_per_gap} raw samples"
        )

        all_raw: list[dict[str, Any]] = []
        with tqdm(total=len(gen_instances), desc="Gap generators", unit="gen") as bar:
            for key, gen in gen_instances:
                samples = gen.generate()
                logger.info(f"  {key}: {len(samples)} samples")
                all_raw.extend(samples)
                bar.update(1)

        # Quality gates
        final, stats = self._quality_gates(all_raw, validator)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "indian_finance_synthetic_gaps.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for s in final:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

        report = self._build_report(final, stats, gen_instances)
        report_path = self.output_dir / "gap_filler_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Gap filler done: {len(final):,} samples → {out_path} ({size_mb:.2f} MB)"
        )
        logger.info(f"Report: {report_path}")
        return out_path

    def merge_with_real(self, real_path: str | Path) -> Path | None:
        """Merge the gap-filler output with the real HF curated dataset.

        Args:
            real_path: Path to ``indian_finance_real_curated.jsonl``.

        Returns:
            Path to the merged JSONL file, or ``None`` if either file is missing.
        """
        real_path = Path(real_path)
        gap_path = self.output_dir / "indian_finance_synthetic_gaps.jsonl"

        if not real_path.exists():
            logger.error(f"Real dataset not found: {real_path}")
            return None
        if not gap_path.exists():
            logger.error(f"Gap-filler output not found — run pipeline first")
            return None

        merged_path = self.output_dir / "indian_finance_hybrid_merged.jsonl"
        total = 0
        with merged_path.open("w", encoding="utf-8") as out:
            for src in (real_path, gap_path):
                for line in src.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        total += 1

        size_mb = merged_path.stat().st_size / (1024 * 1024)
        logger.info(f"Merged dataset: {total:,} samples → {merged_path} ({size_mb:.2f} MB)")
        return merged_path

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _quality_gates(
        samples: list[dict[str, Any]], validator: DataQualityValidator
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Apply dedup + length + structure quality gates.

        Args:
            samples: Raw samples from all generators.
            validator: Configured DataQualityValidator instance.

        Returns:
            Tuple of (filtered_samples, stats_dict).
        """
        seen: set[str] = set()
        dedup_removed = 0
        after_dedup: list[dict[str, Any]] = []
        for s in samples:
            content = s["messages"][1]["content"][:300]
            h = hashlib.md5(content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                after_dedup.append(s)
            else:
                dedup_removed += 1

        valid: list[dict[str, Any]] = []
        invalid_count = 0
        for s in after_dedup:
            if not validator.validate(s):
                valid.append(s)
            else:
                invalid_count += 1

        rng = random.Random(42)
        rng.shuffle(valid)

        stats = {
            "raw": len(samples),
            "dedup_removed": dedup_removed,
            "invalid_removed": invalid_count,
            "final": len(valid),
        }
        logger.info(
            f"Quality gates: raw={stats['raw']}, dedup_removed={dedup_removed}, "
            f"invalid_removed={invalid_count}, final={len(valid)}"
        )
        return valid, stats

    @staticmethod
    def _build_report(
        samples: list[dict[str, Any]],
        stats: dict[str, int],
        gen_instances: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        task_dist = dict(Counter(s.get("task_type", "") for s in samples))
        layer_dist = dict(Counter(s.get("layer", "") for s in samples))
        lang_dist = dict(Counter(s.get("language", "") for s in samples))
        source_dist = dict(Counter(s.get("source_dataset", "") for s in samples))
        return {
            "dataset": "IndianFinanceSLM-SyntheticGapFiller-v2",
            "created": datetime.now().isoformat(),
            "total_samples": len(samples),
            "generators_run": [key for key, _ in gen_instances],
            "quality_gates": stats,
            "task_distribution": task_dist,
            "layer_distribution": layer_dist,
            "language_distribution": lang_dist,
            "source_distribution": source_dist,
        }
