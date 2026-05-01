"""CLI script — run the synthetic gap-filler pipeline.

Usage::

    python -m src.data.run_gap_filler
    python -m src.data.run_gap_filler --config configs/dataset_config.yaml
    python -m src.data.run_gap_filler --samples 500 --output data/raw/synthetic
    python -m src.data.run_gap_filler --generators bank_stmt tax_regime hra_exemption
    python -m src.data.run_gap_filler --merge-real data/raw/hf/indian_finance_real_curated.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.synthetic.gap_filler_pipeline import GapFillerPipeline, _ALL_GENERATORS

_ALL_KEYS = [key for key, _ in _ALL_GENERATORS]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic Indian finance data to fill the 12 identified gaps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/dataset_config.yaml",
        help="Path to dataset_config.yaml (default: configs/dataset_config.yaml)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Override output directory (e.g. data/raw/synthetic)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override samples_per_gap from config",
    )
    p.add_argument(
        "--generators",
        nargs="*",
        choices=_ALL_KEYS,
        metavar="GEN",
        help=f"Subset of generators to run. Choices: {_ALL_KEYS}",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    p.add_argument(
        "--merge-real",
        metavar="PATH",
        default=None,
        help="After generation, merge with the real HF curated JSONL at PATH",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        pipeline = GapFillerPipeline.from_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config not found at {config_path} — using defaults")
        pipeline = GapFillerPipeline()

    if args.output:
        pipeline.output_dir = Path(args.output)
    if args.samples:
        pipeline.samples_per_gap = args.samples
    if args.generators:
        pipeline.enabled_generators = args.generators
    if args.seed is not None:
        pipeline.seed = args.seed

    logger.info(f"Generators ({len(pipeline.enabled_generators)}): {pipeline.enabled_generators}")
    logger.info(f"Samples per generator: {pipeline.samples_per_gap}")
    logger.info(f"Output directory: {pipeline.output_dir}")

    output_path = pipeline.run()

    if args.merge_real:
        merged = pipeline.merge_with_real(args.merge_real)
        if merged:
            logger.info(f"Hybrid merged dataset: {merged}")

    logger.info(f"Done. Gap-filler output: {output_path}")


if __name__ == "__main__":
    main()
