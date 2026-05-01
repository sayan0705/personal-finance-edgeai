"""Abstract base class for all gap-filling synthetic data generators."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from ..constants import SYSTEM_PROMPT


class BaseGapGenerator(ABC):
    """Abstract base for all 12 gap-filling generators.

    Each subclass declares class-level ``gap_name``, ``task_type``, ``layer``,
    and ``source_dataset`` attributes, then implements ``_generate_one``.

    Output format matches the HF pipeline's unified schema so gap-filler samples
    can be merged directly with real dataset samples.

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance for deterministic generation.
    """

    gap_name: str = ""
    task_type: str = "instruction"
    layer: str = "L3_personal_finance"
    source_dataset: str = ""

    def __init__(self, samples_per_gap: int, rng: random.Random) -> None:
        self.samples_per_gap = samples_per_gap
        self.rng = rng

    def generate(self) -> list[dict[str, Any]]:
        """Generate all samples for this gap.

        Returns:
            List of ChatML dicts with id, source_dataset, task_type, layer,
            language, difficulty, and messages keys.
        """
        samples: list[dict[str, Any]] = []
        for i in range(self.samples_per_gap):
            try:
                samples.append(self._generate_one(i))
            except Exception as exc:
                logger.debug(f"{self.__class__.__name__} sample {i} failed: {exc}")
        logger.debug(
            f"{self.__class__.__name__}: {len(samples)}/{self.samples_per_gap} generated"
        )
        return samples

    @abstractmethod
    def _generate_one(self, idx: int) -> dict[str, Any]:
        """Generate a single sample.

        Args:
            idx: Zero-based index used to build the sample id.

        Returns:
            ChatML-formatted dict.
        """
        ...

    def _make_sample(
        self,
        idx: int,
        q: str,
        a: str,
        lang: str = "en",
        difficulty: str = "intermediate",
        task_type: str | None = None,
        layer: str | None = None,
    ) -> dict[str, Any]:
        """Assemble a sample dict in the project's unified output format.

        Args:
            idx: Sequential index (becomes part of the sample id).
            q: User question text.
            a: Assistant answer text.
            lang: Language tag (e.g. ``en``, ``hinglish``).
            difficulty: ``beginner``, ``intermediate``, or ``advanced``.
            task_type: Override class-level ``task_type``.
            layer: Override class-level ``layer``.

        Returns:
            Dict with id, source_dataset, task_type, layer, language,
            difficulty, and messages.
        """
        return {
            "id": f"{self.source_dataset}_{idx:05d}",
            "source_dataset": self.source_dataset,
            "task_type": task_type or self.task_type,
            "layer": layer or self.layer,
            "language": lang,
            "difficulty": difficulty,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
        }
