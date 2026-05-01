"""Extract 1 000 random FinQA samples and merge into the hybrid dataset.

FinQA (https://finqasite.github.io/) contains financial-report QA pairs that
require numerical reasoning over SEC filings.  This script pulls from
``ibm/finqa`` on HuggingFace, converts samples to the project's ChatML format,
applies the standard quality gates, and appends them to the canonical
hybrid-merged JSONL file.

Usage::

    python -m src.data.run_finqa_extraction
    python -m src.data.run_finqa_extraction --samples 500 --seed 99
    python -m src.data.run_finqa_extraction --hybrid-path data/raw/synthetic/indian_finance_hybrid_merged.jsonl
    python -m src.data.run_finqa_extraction --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.hf_datasets.converters import convert_finqa
from src.data.hf_datasets.extractor import (
    assign_layer,
    gate_dedup,
    gate_length,
    gate_safety,
)

_SOURCE_ID = "finqasite/finqa"
_DEFAULT_HYBRID = "data/raw/synthetic/indian_finance_hybrid_merged.jsonl"
_DEFAULT_FINQA_OUT = "data/raw/hf/finqa_curated.jsonl"
_DEFAULT_SAMPLES = 1000

# FinQA JSON files hosted on the official GitHub repository
_FINQA_URLS = {
    "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "dev": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
}


def _fetch_finqa_split(url: str, split_name: str) -> list[dict]:
    """Download a FinQA JSON split from GitHub."""
    import urllib.request

    logger.info(f"Downloading FinQA {split_name} split from {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        logger.info(f"FinQA {split_name}: {len(data):,} examples")
        return data
    except Exception as exc:
        logger.error(f"Failed to download {split_name}: {exc}")
        return []


def _load_finqa(samples: int, seed: int) -> list[dict]:
    """Download FinQA from GitHub and return converted ChatML records."""
    train_rows = _fetch_finqa_split(_FINQA_URLS["train"], "train")
    dev_rows = _fetch_finqa_split(_FINQA_URLS["dev"], "dev")

    all_rows = train_rows + dev_rows
    if not all_rows:
        raise RuntimeError("Could not download any FinQA data")

    total_available = len(all_rows)
    logger.info(f"Total FinQA available: {total_available:,}")

    rng = random.Random(seed)
    sample_n = min(samples, total_available)
    sampled = rng.sample(all_rows, sample_n)
    logger.info(f"Sampled {len(sampled):,} examples (seed={seed})")

    converted: list[dict] = []
    failed = 0
    for idx, row in enumerate(sampled):
        record = convert_finqa(row, _SOURCE_ID, idx)
        if record:
            record["layer"] = assign_layer(record)
            converted.append(record)
        else:
            failed += 1

    rate = len(converted) / max(1, len(sampled)) * 100
    logger.info(f"Converted {len(converted):,}/{len(sampled):,} ({rate:.1f}%), failed: {failed}")
    return converted


def _apply_gates(samples: list[dict]) -> list[dict]:
    """Run dedup, length, and safety gates (skip diversity — single source)."""
    after_dedup, n_dd = gate_dedup(samples)
    after_length, n_len = gate_length(after_dedup)
    after_safety, n_safe = gate_safety(after_length)
    logger.info(
        f"Quality gates: -{n_dd} dedup, -{n_len} length, -{n_safe} safety "
        f"→ {len(after_safety):,} remain"
    )
    return after_safety


def _save_finqa(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_clean(r), ensure_ascii=False) + "\n")
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"FinQA curated JSONL → {out_path} ({size_kb:.1f} KB, {len(records):,} records)")


def _merge_into_hybrid(finqa_records: list[dict], hybrid_path: Path, seed: int) -> None:
    """Append FinQA records to the hybrid JSONL, shuffle, and overwrite."""
    hybrid_path = hybrid_path.resolve()

    existing: list[dict] = []
    if hybrid_path.exists():
        with hybrid_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
        logger.info(f"Loaded {len(existing):,} existing records from {hybrid_path}")
    else:
        logger.warning(f"Hybrid file not found at {hybrid_path} — will create it")

    clean_finqa = [_clean(r) for r in finqa_records]

    # Avoid duplicates by ID
    existing_ids = {r.get("id") for r in existing}
    new_records = [r for r in clean_finqa if r.get("id") not in existing_ids]
    skipped = len(clean_finqa) - len(new_records)
    if skipped:
        logger.info(f"Skipped {skipped} FinQA records already present in hybrid")

    merged = existing + new_records
    random.Random(seed).shuffle(merged)

    hybrid_path.parent.mkdir(parents=True, exist_ok=True)
    with hybrid_path.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    size_mb = hybrid_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Merged hybrid JSONL → {hybrid_path} "
        f"({len(existing):,} existing + {len(new_records):,} FinQA "
        f"= {len(merged):,} total, {size_mb:.2f} MB)"
    )


def _clean(record: dict) -> dict:
    return {
        "id": record["id"],
        "source_dataset": record["source_dataset"],
        "task_type": record["task_type"],
        "layer": record.get("layer", "L1_financial_qa"),
        "language": record["language"],
        "messages": record["messages"],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract FinQA samples and merge into the hybrid fine-tuning dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--samples",
        type=int,
        default=_DEFAULT_SAMPLES,
        help=f"Number of FinQA samples to extract (default: {_DEFAULT_SAMPLES})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    p.add_argument(
        "--finqa-out",
        default=_DEFAULT_FINQA_OUT,
        metavar="PATH",
        help=f"Where to save the FinQA-only curated JSONL (default: {_DEFAULT_FINQA_OUT})",
    )
    p.add_argument(
        "--hybrid-path",
        default=_DEFAULT_HYBRID,
        metavar="PATH",
        help=f"Hybrid merged JSONL to append FinQA samples into (default: {_DEFAULT_HYBRID})",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging into the hybrid JSONL (only save the FinQA curated file)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and convert samples but do not write any files",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logger.info(f"FinQA extraction starting — target {args.samples:,} samples, seed={args.seed}")

    records = _load_finqa(samples=args.samples, seed=args.seed)
    records = _apply_gates(records)

    if not records:
        logger.error("No usable FinQA records after quality gates — aborting")
        sys.exit(1)

    if args.dry_run:
        logger.info(f"Dry-run: would write {len(records):,} records — no files written")
        return

    finqa_out = Path(args.finqa_out)
    _save_finqa(records, finqa_out)

    if not args.no_merge:
        _merge_into_hybrid(records, Path(args.hybrid_path), seed=args.seed)
        logger.info("Done — FinQA samples extracted and merged into hybrid dataset")
    else:
        logger.info("Done — FinQA samples extracted (merge skipped)")


if __name__ == "__main__":
    main()
