"""Main synthetic data generator for FinEdge training corpus."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from tqdm import tqdm

from .constants import SYSTEM_PROMPT
from .templates import InsuranceTemplates, InvestmentTemplates, LoanTemplates, TaxTemplates
from .validators import DataQualityValidator

_TOPIC_TEMPLATE_MAP = {
    "tax": TaxTemplates,
    "investment": InvestmentTemplates,
    "loan": LoanTemplates,
    "insurance": InsuranceTemplates,
}


class SyntheticDataGenerator:
    """Generates synthetic Indian personal finance Q&A training data.

    Produces ChatML-formatted samples that are factually grounded in FY 2024-25
    tax rules and verified financial calculations. All samples are deterministic
    given the same seed.

    Args:
        config: The ``synthetic`` section from ``dataset_config.yaml``.
        seed: Random seed for reproducibility.

    Example::

        generator = SyntheticDataGenerator.from_config("configs/dataset_config.yaml")
        samples = generator.generate()
        generator.save(samples, "data/raw/custom/qa-pairs/synthetic.jsonl")
    """

    def __init__(self, config: dict, seed: int | None = None) -> None:
        self._cfg = config
        self._seed = seed if seed is not None else config.get("random_seed", 42)
        self._rng = random.Random(self._seed)
        self._validator = DataQualityValidator(
            min_answer_length=config.get("min_answer_length", 80),
            max_answer_length=config.get("max_answer_length", 2000),
        )
        self._topic_counts: dict[str, int] = config.get(
            "topic_counts",
            {"tax": 500, "investment": 500, "loan": 300, "insurance": 200},
        )

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SyntheticDataGenerator":
        """Instantiate from a dataset_config.yaml file.

        Args:
            config_path: Path to the YAML config.

        Returns:
            Configured SyntheticDataGenerator.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        synthetic_cfg = cfg.get("synthetic", {})
        return cls(synthetic_cfg)

    # ── public API ────────────────────────────────────────────────────────────

    def generate(self) -> list[dict]:
        """Generate all configured samples across all topics.

        Returns:
            List of ChatML-formatted dicts ready for JSONL serialisation.
        """
        all_samples: list[dict] = []
        total = sum(self._topic_counts.values())
        logger.info(
            f"Generating {total} synthetic samples "
            f"(seed={self._seed}, topics={list(self._topic_counts.keys())})"
        )

        with tqdm(total=total, desc="Synthetic samples", unit="sample") as bar:
            for topic, count in self._topic_counts.items():
                topic_samples = self._generate_topic(topic, count, bar)
                all_samples.extend(topic_samples)

        logger.info(
            f"Generated {len(all_samples)} valid samples "
            f"(dropped {total - len(all_samples)} that failed validation)"
        )
        return all_samples

    def save(self, samples: list[dict], output_path: str | Path) -> None:
        """Write samples to a JSONL file (UTF-8, one JSON object per line).

        Args:
            samples: ChatML dicts to serialize.
            output_path: Destination file path (created if needed).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as fh:
            for sample in samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(samples)} samples to {output_path}")

    # ── private helpers ───────────────────────────────────────────────────────

    def _generate_topic(
        self, topic: str, count: int, bar: tqdm  # type: ignore[type-arg]
    ) -> list[dict]:
        template_cls = _TOPIC_TEMPLATE_MAP.get(topic)
        if template_cls is None:
            logger.warning(f"Unknown topic '{topic}'; skipping")
            return []

        template = template_cls(self._rng)
        samples: list[dict] = []

        for _ in range(count):
            try:
                raw = template.generate_sample()
                chatml = self._to_chatml(raw)
                errors = self._validator.validate(chatml)
                if not errors:
                    samples.append(chatml)
            except Exception as exc:
                logger.debug(f"Sample generation failed for topic '{topic}': {exc}")
            bar.update(1)

        logger.debug(f"Topic '{topic}': generated {len(samples)}/{count} valid samples")
        return samples

    @staticmethod
    def _to_chatml(raw: dict[str, str]) -> dict[str, Any]:
        """Wrap a {question, answer} dict in the ChatML messages format."""
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw["question"]},
                {"role": "assistant", "content": raw["answer"]},
            ]
        }
