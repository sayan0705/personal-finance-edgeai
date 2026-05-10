"""Shared helpers for BanyanTree-style finance tools."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("BANYANTREE_DATA_DIR", project_root() / "data")).resolve()


def financial_kg_root() -> Path:
    return Path(
        os.environ.get("BANYANTREE_FINANCIAL_KG_ROOT", data_dir() / "financial_kg")
    ).resolve()


def seed_docs_path() -> Path:
    return Path(
        os.environ.get(
            "BANYANTREE_SEED_DOCS_PATH",
            financial_kg_root() / "raw_docs" / "seed" / "personal_finance_seed.json",
        )
    ).resolve()


def flattened_docs_path() -> Path:
    return Path(
        os.environ.get(
            "BANYANTREE_PAGEINDEX_FLATTENED_DOCS_PATH",
            financial_kg_root() / "raw_docs" / "pageindex" / "pageindex_flattened_docs.json",
        )
    ).resolve()


def amount_values(text: str) -> list[float]:
    vals: list[float] = []
    pattern = r"(?:rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|thousand)?"
    for value, unit in re.findall(pattern, text or "", re.I):
        amount = float(value.replace(",", ""))
        unit = (unit or "").lower()
        if unit in {"crore", "cr"}:
            amount *= 10_000_000
        elif unit in {"lakh", "lac"}:
            amount *= 100_000
        elif unit in {"k", "thousand"}:
            amount *= 1_000
        if amount >= 500:
            vals.append(amount)
    return vals


def pct(text: str, default: float = 12.0) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text or "")
    return float(match.group(1)) if match else default


def years(text: str, default: float = 10.0) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", text or "", re.I)
    return float(match.group(1)) if match else default


def age(text: str, default: int = 30) -> int:
    match = re.search(r"(?:age\s*[:=]?\s*|i am\s+|i'm\s+)(\d{2})", text or "", re.I)
    return int(match.group(1)) if match else default


def allocation(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for asset in ["equity", "debt", "gold", "cash"]:
        patterns = [
            rf"{asset}\s*(?:is|at|=|:)?\s*(\d{{1,3}})\s*%",
            rf"(\d{{1,3}})\s*%\s*{asset}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text or "", re.I)
            if match:
                out[asset] = int(match.group(1))
                break
    return out


def fmt_inr(value: float) -> str:
    if abs(value) >= 10_000_000:
        return f"Rs {value / 10_000_000:.2f} crore"
    if abs(value) >= 100_000:
        return f"Rs {value / 100_000:.2f} lakh"
    return f"Rs {value:,.0f}"


def sip_future_value(monthly: float, annual_rate_pct: float, duration_years: float) -> float:
    monthly_rate = annual_rate_pct / 100 / 12
    months = max(1, int(round(duration_years * 12)))
    if monthly_rate == 0:
        return monthly * months
    return monthly * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)


def emi_amount(principal: float, annual_rate_pct: float, duration_years: float) -> tuple[float, float]:
    monthly_rate = annual_rate_pct / 100 / 12
    months = max(1, int(round(duration_years * 12)))
    if monthly_rate == 0:
        emi = principal / months
    else:
        emi = principal * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)
    total = emi * months
    return emi, total


def required_sip(target: float, annual_rate_pct: float, duration_years: float) -> float:
    monthly_rate = annual_rate_pct / 100 / 12
    months = max(1, int(round(duration_years * 12)))
    if monthly_rate == 0:
        return target / months
    factor = (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
    return target / factor


def load_financial_docs() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in (seed_docs_path(), flattened_docs_path()):
        if not path.exists():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(loaded, list):
            continue
        for i, doc in enumerate(loaded):
            if not isinstance(doc, dict):
                continue
            content = str(doc.get("content", "")).strip()
            if not content:
                continue
            docs.append(
                {
                    **doc,
                    "title": str(doc.get("title", f"Document {i}")).strip(),
                    "content": content,
                    "source": str(doc.get("source", path.name)),
                }
            )
    return docs


def search_financial_docs(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    terms = [t for t in re.findall(r"\w+", (query or "").lower()) if len(t) > 2]
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in load_financial_docs():
        haystack = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
        score = sum(haystack.count(term) for term in terms)
        if score:
            title_hit = sum(str(doc.get("title", "")).lower().count(term) for term in terms)
            scored.append((score + title_hit * 2, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def extract_symbols(query: str) -> list[str]:
    symbols: list[str] = []
    for token in re.findall(r"\b[A-Z]{2,12}\b", query or ""):
        if token not in {"SIP", "EMI", "RBI", "SEBI", "NAV", "ETF"}:
            symbols.append(token)
    company_map = {
        "tcs": "TCS",
        "infosys": "INFY",
        "infy": "INFY",
        "hdfc bank": "HDFCBANK",
        "reliance": "RELIANCE",
        "wipro": "WIPRO",
        "icici bank": "ICICIBANK",
    }
    lowered = (query or "").lower()
    for name, symbol in company_map.items():
        if name in lowered and symbol not in symbols:
            symbols.append(symbol)
    return symbols[:4]


def simple_risk_score(volatility_hint: float | None = None, allocation_equity: int | None = None) -> str:
    score = 0.0
    if volatility_hint is not None and not math.isnan(volatility_hint):
        score += volatility_hint
    if allocation_equity is not None:
        score += allocation_equity / 100
    if score >= 0.75:
        return "high"
    if score >= 0.35:
        return "moderate"
    return "low"
