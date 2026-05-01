"""Core extraction engine: load → sample → convert → quality-gate."""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from typing import Any, Optional

from loguru import logger

from .converters import detect_and_convert
from .registry import DATASET_REGISTRY, LAYER_KEYWORDS, UNSAFE_PATTERNS


# ── Layer assignment ──────────────────────────────────────────────────────────

def assign_layer(sample: dict[str, Any]) -> str:
    """Assign a sample to the appropriate dataset layer.

    Layers:
        L1_foundation / L1_financial_qa / L1_sentiment_nlp — generic finance
        L2_indian_regulatory — India-specific content
        L3_personal_finance — personal finance advisory
        L4_community_conversational — advisory/conversational style

    Args:
        sample: Normalised record dict with ``source_dataset``, ``task_type``,
                and ``messages`` fields.

    Returns:
        Layer label string.
    """
    source = sample.get("source_dataset", "")
    task = sample.get("task_type", "")
    full_text = " ".join(m["content"].lower() for m in sample.get("messages", []))

    if "agamai" in source.lower() or "indian" in source.lower():
        return "L2_indian_regulatory"

    india_score = sum(1 for kw in LAYER_KEYWORDS["L2_indian_regulatory"] if kw in full_text)
    if india_score >= 2:
        return "L2_indian_regulatory"

    pf_score = sum(1 for kw in LAYER_KEYWORDS["L3_personal_finance"] if kw in full_text)
    if pf_score >= 2:
        return "L3_personal_finance"

    conv_score = sum(1 for kw in LAYER_KEYWORDS["L4_community_conversational"] if kw in full_text)
    if conv_score >= 1 and task in ("instruction", "conversation", "qa"):
        return "L4_community_conversational"

    if task in ("sentiment", "classification", "ner"):
        return "L1_sentiment_nlp"

    if task in ("qa", "qa_context"):
        return "L1_financial_qa"

    return "L1_foundation"


# ── Individual dataset extraction ─────────────────────────────────────────────

def extract_dataset(
    ds_key: str,
    system_prompt: str,
    samples_per_dataset: int,
    max_samples_large: int,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one HuggingFace dataset, sample it, and convert rows to the unified format.

    Args:
        ds_key: Key in DATASET_REGISTRY identifying the dataset.
        system_prompt: Default system-role content for converted records.
        samples_per_dataset: Target sample count for regular-sized datasets.
        max_samples_large: Target sample count for very large datasets (500K+).
        seed: Random seed for reproducible sampling.

    Returns:
        Tuple of (converted_records, log_entry_dict).
    """
    from datasets import load_dataset  # imported here to keep module importable without datasets

    ds_cfg = DATASET_REGISTRY[ds_key]
    hf_id = ds_cfg["hf_id"]
    split = ds_cfg["split"]
    config = ds_cfg["config"]

    # Large datasets get a higher cap
    is_large = "500k" in ds_key.lower() or "177k" in ds_key.lower() or "sujet" in ds_key.lower()
    target_n = max_samples_large if is_large else samples_per_dataset

    logger.info(f"[{ds_key}] Loading {hf_id} (split={split}, config={config or 'default'})")

    try:
        ds = load_dataset(hf_id, config, split=split, trust_remote_code=True) if config else load_dataset(
            hf_id, split=split, trust_remote_code=True
        )
    except Exception as exc:
        logger.error(f"[{ds_key}] Load failed: {exc}")
        return [], {"hf_id": hf_id, "status": "FAILED", "error": str(exc)[:200], "converted": 0}

    total_available = len(ds)
    logger.info(f"[{ds_key}] {total_available:,} rows available | columns: {list(ds.column_names)}")

    sample_n = min(target_n, total_available)
    rng = random.Random(seed)
    indices = rng.sample(range(total_available), sample_n)
    sampled = ds.select(indices)

    converted: list[dict[str, Any]] = []
    failed = 0
    for i, row in enumerate(sampled):
        result = detect_and_convert(dict(row), hf_id, i, ds_key, system_prompt)
        if result:
            converted.append(result)
        else:
            failed += 1

    conversion_rate = len(converted) / max(1, sample_n) * 100
    logger.info(f"[{ds_key}] Converted {len(converted)}/{sample_n} ({conversion_rate:.1f}%)")

    log_entry = {
        "hf_id": hf_id,
        "status": "SUCCESS",
        "total_available": total_available,
        "sampled": sample_n,
        "converted": len(converted),
        "failed": failed,
        "columns": list(ds.column_names),
        "conversion_rate": f"{conversion_rate:.1f}%",
    }
    return converted, log_entry


# ── Quality gates ─────────────────────────────────────────────────────────────

def gate_dedup(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove duplicate samples via MD5 hash of the first user-turn content.

    Args:
        samples: Input sample list.

    Returns:
        Tuple of (deduplicated_samples, num_removed).
    """
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for s in samples:
        user_text = " ".join(m["content"][:200] for m in s["messages"] if m["role"] == "user")
        h = hashlib.md5(user_text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            kept.append(s)
    removed = len(samples) - len(kept)
    logger.info(f"Dedup gate: removed {removed} duplicates, {len(kept):,} remain")
    return kept, removed


def gate_length(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Filter out samples with too-short turns or suspiciously long content.

    Args:
        samples: Input sample list.

    Returns:
        Tuple of (filtered_samples, num_removed).
    """
    kept: list[dict[str, Any]] = []
    for s in samples:
        msgs = s.get("messages", [])
        user_msgs = [m for m in msgs if m["role"] == "user"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        if not user_msgs or len(user_msgs[0]["content"].strip()) < 10:
            continue
        if not asst_msgs or len(asst_msgs[0]["content"].strip()) < 15:
            continue
        if any(len(m["content"]) > 15_000 for m in msgs):
            continue
        kept.append(s)
    removed = len(samples) - len(kept)
    logger.info(f"Length gate: removed {removed}, {len(kept):,} remain")
    return kept, removed


def gate_safety(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Filter samples whose assistant turn contains unsafe financial claims.

    Phrases like "guaranteed returns" or "get rich quick" are removed unless
    they appear in a clearly negative/warning context.

    Args:
        samples: Input sample list.

    Returns:
        Tuple of (filtered_samples, num_removed).
    """
    kept: list[dict[str, Any]] = []
    for s in samples:
        asst_text = " ".join(m["content"].lower() for m in s["messages"] if m["role"] == "assistant")
        unsafe = False
        for pattern in UNSAFE_PATTERNS:
            m = re.search(pattern, asst_text)
            if m:
                start = max(0, m.start() - 50)
                window = asst_text[start : m.start() + 150]
                negations = ("avoid", "never", "don't", "warning", "scam", "beware", "not")
                if not any(neg in window for neg in negations):
                    unsafe = True
                    break
        if not unsafe:
            kept.append(s)
    removed = len(samples) - len(kept)
    logger.info(f"Safety gate: removed {removed}, {len(kept):,} remain")
    return kept, removed


def gate_diversity(
    samples: list[dict[str, Any]], max_pct: float = 0.30, seed: int = 42
) -> tuple[list[dict[str, Any]], int]:
    """Cap any single source to at most ``max_pct`` of the total corpus.

    Args:
        samples: Input sample list (will be shuffled in-place for fairness).
        max_pct: Maximum fraction any one source may occupy.
        seed: Random seed for the shuffle.

    Returns:
        Tuple of (capped_samples, num_removed).
    """
    random.Random(seed).shuffle(samples)
    max_per_source = int(len(samples) * max_pct)
    budget: dict[str, int] = defaultdict(int)
    kept: list[dict[str, Any]] = []
    for s in samples:
        src = s["source_dataset"]
        if budget[src] < max_per_source:
            kept.append(s)
            budget[src] += 1
    removed = len(samples) - len(kept)
    logger.info(f"Diversity gate: capped {removed} samples, {len(kept):,} remain")
    return kept, removed


def run_quality_gates(
    samples: list[dict[str, Any]], diversity_cap_pct: float = 0.30, seed: int = 42
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply all four quality gates in sequence.

    Gates (in order): deduplication → length → safety → diversity cap.

    Args:
        samples: Raw extracted samples.
        diversity_cap_pct: Max fraction any source may contribute.
        seed: Random seed for diversity shuffle.

    Returns:
        Tuple of (final_samples, stats_dict) where stats_dict records how many
        samples each gate removed.
    """
    after_dedup, n_dedup = gate_dedup(samples)
    after_length, n_length = gate_length(after_dedup)
    after_safety, n_safety = gate_safety(after_length)
    final, n_diversity = gate_diversity(after_safety, max_pct=diversity_cap_pct, seed=seed)

    stats = {
        "duplicates_removed": n_dedup,
        "length_filtered": n_length,
        "safety_filtered": n_safety,
        "diversity_capped": n_diversity,
    }
    logger.info(
        f"Quality gates complete: {len(samples):,} → {len(final):,} "
        f"(removed {len(samples)-len(final):,})"
    )
    return final, stats
