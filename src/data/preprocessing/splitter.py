"""Reproducible train/eval/test split with JSONL output."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class DatasetSplitter:
    """Split a list of ChatML samples into train/eval/test sets and save them.

    Args:
        train_ratio: Fraction of data for training (default 0.80).
        eval_ratio: Fraction for evaluation (default 0.10).
        seed: Random seed for reproducible shuffling.

    The test fraction is inferred as ``1 - train - eval``.
    """

    def __init__(
        self,
        train_ratio: float = 0.80,
        eval_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        if not (0 < train_ratio < 1 and 0 < eval_ratio < 1):
            raise ValueError("train_ratio and eval_ratio must be between 0 and 1 exclusive")
        if train_ratio + eval_ratio >= 1.0:
            raise ValueError("train_ratio + eval_ratio must be < 1.0 (leaves room for test)")

        self._train_r = train_ratio
        self._eval_r = eval_ratio
        self._seed = seed

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DatasetSplitter":
        """Instantiate from a dataset_config.yaml file.

        Args:
            config_path: Path to the config YAML.

        Returns:
            Configured DatasetSplitter.
        """
        path = Path(config_path)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        ratios = cfg.get("dataset", {}).get("split_ratios", {})
        seed = cfg.get("dataset", {}).get("random_seed", 42)
        return cls(
            train_ratio=ratios.get("train", 0.80),
            eval_ratio=ratios.get("eval", 0.10),
            seed=seed,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def split(
        self, samples: list[dict[str, Any]]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Shuffle and split samples into train/eval/test.

        Args:
            samples: List of ChatML dicts to split.

        Returns:
            Tuple of (train, eval, test) sample lists.
        """
        shuffled = list(samples)
        random.Random(self._seed).shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * self._train_r)
        n_eval = int(n * self._eval_r)

        train = shuffled[:n_train]
        eval_ = shuffled[n_train : n_train + n_eval]
        test = shuffled[n_train + n_eval :]

        logger.info(
            f"Split {n} samples → train:{len(train)} | eval:{len(eval_)} | test:{len(test)}"
        )
        return train, eval_, test

    def save(
        self,
        samples: list[dict[str, Any]],
        train_dir: str | Path,
        eval_dir: str | Path,
        test_dir: str | Path,
        filename: str = "data.jsonl",
    ) -> None:
        """Split and write the three splits to separate directories.

        Args:
            samples: ChatML dicts to split and save.
            train_dir: Directory for training JSONL.
            eval_dir: Directory for evaluation JSONL.
            test_dir: Directory for test JSONL.
            filename: Name of the JSONL file in each directory.
        """
        train, eval_, test = self.split(samples)

        for split_name, split_data, out_dir in [
            ("train", train, train_dir),
            ("eval", eval_, eval_dir),
            ("test", test, test_dir),
        ]:
            out_path = Path(out_dir) / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_jsonl(split_data, out_path)
            logger.info(f"Saved {split_name} split ({len(split_data)} samples) → {out_path}")

    @staticmethod
    def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
        """Load a JSONL file into a list of dicts.

        Args:
            path: Path to the JSONL file.

        Returns:
            List of parsed JSON objects.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        samples = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(f"Skipping malformed JSON at {path}:{lineno}: {exc}")

        return samples

    @staticmethod
    def _write_jsonl(samples: list[dict], path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for sample in samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
