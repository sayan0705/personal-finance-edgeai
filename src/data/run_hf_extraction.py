"""CLI script — run the HuggingFace real dataset extraction pipeline.

Usage::

    python -m src.data.run_hf_extraction
    python -m src.data.run_hf_extraction --config configs/dataset_config.yaml
    python -m src.data.run_hf_extraction --datasets finance_alpaca indian_itr
    python -m src.data.run_hf_extraction --samples 200 --output data/raw/hf
    python -m src.data.run_hf_extraction --merge-synthetic data/raw/custom/synthetic.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# Ensure the project root is on sys.path when run directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.hf_datasets.pipeline import HFDatasetPipeline
from src.data.hf_datasets.registry import DATASET_REGISTRY


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract real financial datasets from HuggingFace and curate them into "
        "the finance-alpaca ChatML format.",
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
        help="Override output directory from config (e.g. data/raw/hf)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override samples_per_dataset from config",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        choices=list(DATASET_REGISTRY.keys()),
        metavar="DS",
        help=f"Subset of datasets to run. Choices: {list(DATASET_REGISTRY.keys())}",
    )
    p.add_argument(
        "--merge-synthetic",
        metavar="PATH",
        default=None,
        help="After extraction, merge with a synthetic JSONL file at PATH",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Build pipeline — prefer config file, then apply CLI overrides
    config_path = Path(args.config)
    if config_path.exists():
        pipeline = HFDatasetPipeline.from_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config not found at {config_path} — using defaults")
        pipeline = HFDatasetPipeline()

    # CLI overrides
    if args.output:
        pipeline.output_dir = Path(args.output)
    if args.samples:
        pipeline.samples_per_dataset = args.samples
    if args.datasets:
        pipeline.enabled_datasets = args.datasets
    if args.seed is not None:
        pipeline.seed = args.seed

    logger.info(f"Enabled datasets ({len(pipeline.enabled_datasets)}): {pipeline.enabled_datasets}")
    logger.info(f"Output directory: {pipeline.output_dir}")
    logger.info(f"Samples per dataset: {pipeline.samples_per_dataset}")

    output_path = pipeline.run()

    if args.merge_synthetic:
        merged_path = pipeline.merge_with_synthetic(args.merge_synthetic)
        if merged_path:
            logger.info(f"Merged dataset written to {merged_path}")

    logger.info(f"Done. Curated dataset: {output_path}")


if __name__ == "__main__":
    main()
