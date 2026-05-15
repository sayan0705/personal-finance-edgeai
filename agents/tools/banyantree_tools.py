"""BanyanTree-compatible tool implementations for the container API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from agents.tools.base import BaseTool


def _load_nse_dict() -> dict[str, str]:
    """Load company-name → NSE-ticker mapping from the shared data volume."""
    path = Path(os.environ.get("BANYANTREE_FINANCIAL_KG_ROOT", "/app/data/financial_kg")) / "nse_ticker_dict.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k.lower(): v for k, v in data.get("tickers", {}).items()}
    except Exception:
        return {}

_NSE_DICT: dict[str, str] = {}

def _resolve_nse_ticker(symbol: str) -> str:
    """Resolve a company name or partial ticker to the correct NSE ticker symbol.

    e.g. "HCL" → "HCLTECH", "Infosys" → "INFY", already-correct "TCS" → "TCS".
    Falls back to the original symbol uppercased if no match found.
    """
    global _NSE_DICT
    if not _NSE_DICT:
        _NSE_DICT = _load_nse_dict()
    key = symbol.strip().lower()
    # Exact match
    if key in _NSE_DICT:
        return _NSE_DICT[key]
    # Prefix match (e.g. "hcl tech" matches "hcl technologies")
    for name, ticker in _NSE_DICT.items():
        if key in name or name in key:
            return ticker
    return symbol.upper()
from agents.tools.banyantree_common import (
    age,
    allocation,
    amount_values,
    emi_amount,
    extract_symbols,
    fmt_inr,
    pct,
    required_sip,
    search_financial_docs,
    simple_risk_score,
    sip_future_value,
    years,
)


class AMFINavTool(BaseTool):
    @property
    def name(self) -> str:
        return "amfi_nav"

    @property
    def description(self) -> str:
        return "Fetch AMFI mutual fund NAV rows, optionally filtered by fund name such as ELSS."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"fund_filter": {"type": "string"}},
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        fund_filter = str(args.get("fund_filter") or args.get("query") or "").strip()
        try:
            response = httpx.get("https://www.amfiindia.com/spages/NAVAll.txt", timeout=20)
            response.raise_for_status()
            lines = [line for line in response.text.splitlines() if line.strip()]
            if fund_filter:
                sample = "\n".join(line for line in lines if fund_filter.lower() in line.lower())[:1200]
            else:
                sample = "\n".join(lines[:20])[:1200]
            if not sample:
                sample = "No AMFI rows matched the requested filter."
            return {"result": f"AMFI NAV for {fund_filter or 'all funds'}:\n{sample}"}
        except Exception as exc:
            return {"result": f"AMFI NAV lookup failed: {exc}"}


class SearchRAGTool(BaseTool):
    @property
    def name(self) -> str:
        return "search_rag"

    @property
    def description(self) -> str:
        return "Search BanyanTree personal-finance KG seed docs and offline PDF/Qwen extracted docs."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 4},
            },
            "required": ["query"],
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        top_k = int(args.get("top_k", 4))
        docs = search_financial_docs(query, top_k=top_k)
        if not docs:
            return {"result": "No matching BanyanTree finance KG documents found.", "sources": []}
        parts = []
        sources = []
        for doc in docs:
            title = str(doc.get("title", "Untitled"))
            content = str(doc.get("content", "")).strip()
            parts.append(f"{title}: {content[:500]}")
            sources.append(
                {
                    "title": title,
                    "source": doc.get("source", ""),
                    "section_path": doc.get("section_path", []),
                    "page_start": doc.get("page_start"),
                    "page_end": doc.get("page_end"),
                }
            )
        return {"result": "\n\n".join(parts), "sources": sources}


class SIPQueryCalculator(BaseTool):
    @property
    def name(self) -> str:
        return "sip_calculator"

    @property
    def description(self) -> str:
        return "BanyanTree SIP calculator from a natural language query, e.g. '5000 SIP for 10 years at 12%'."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        amounts = amount_values(query)
        monthly = amounts[0] if amounts else float(args.get("monthly_amount", 5000))
        duration = years(query, float(args.get("years", 10)))
        rate = pct(query, float(args.get("annual_rate_pct", 12)))
        fv = sip_future_value(monthly, rate, duration)
        invested = monthly * duration * 12
        return {
            "monthly": monthly,
            "years": duration,
            "annual_rate_pct": rate,
            "invested": round(invested),
            "estimated_value": round(fv),
            "result": (
                f"SIP plan | Monthly: {fmt_inr(monthly)} | Horizon: {duration:.1f} years | "
                f"Return: {rate:.2f}% | Invested: {fmt_inr(invested)} | "
                f"Estimated value: {fmt_inr(fv)}"
            ),
        }


class EMIQueryCalculator(BaseTool):
    @property
    def name(self) -> str:
        return "emi_calculator"

    @property
    def description(self) -> str:
        return "BanyanTree EMI calculator from a natural language loan query."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        amounts = amount_values(query)
        principal = amounts[0] if amounts else float(args.get("principal", 5_000_000))
        duration = years(query, float(args.get("tenure_years", 20)))
        rate = pct(query, float(args.get("annual_rate_pct", 8.5)))
        emi, total = emi_amount(principal, rate, duration)
        return {
            "emi": round(emi),
            "principal": round(principal),
            "total_interest": round(total - principal),
            "result": (
                f"EMI plan | Loan: {fmt_inr(principal)} | Rate: {rate:.2f}% | "
                f"Tenure: {duration:.1f} years | EMI: {fmt_inr(emi)} | "
                f"Total interest: {fmt_inr(total - principal)}"
            ),
        }


class PortfolioHealthTool(BaseTool):
    @property
    def name(self) -> str:
        return "portfolio_health"

    @property
    def description(self) -> str:
        return "Review portfolio allocation and suggest a BanyanTree model mix."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        investor_age = age(query)
        alloc = allocation(query)
        risk = "aggressive" if investor_age < 35 else "moderate" if investor_age < 50 else "conservative"
        eq = 70 if risk == "aggressive" else 55 if risk == "moderate" else 35
        debt = 15 if risk == "aggressive" else 30 if risk == "moderate" else 45
        current_equity = alloc.get("equity")
        warnings = []
        if current_equity is not None and abs(current_equity - eq) > 20:
            warnings.append("equity allocation differs materially from model mix")
        if sum(alloc.values()) > 110:
            warnings.append("allocation percentages appear inconsistent")
        summary = (
            f"Portfolio review | Age: {investor_age} | Risk: {risk} | Current: {alloc or 'not provided'} | "
            f"Model mix: equity {eq}%, debt {debt}%, gold 10%, cash {max(5, 100 - eq - debt - 10)}% | "
            f"{'; '.join(warnings) if warnings else 'Allocation broadly aligned'}"
        )
        return {"risk_profile": risk, "model_allocation": {"equity": eq, "debt": debt, "gold": 10}, "result": summary}


class GoalPlannerTool(BaseTool):
    @property
    def name(self) -> str:
        return "goal_planner"

    @property
    def description(self) -> str:
        return "Calculate monthly SIP needed for a target corpus."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        amounts = amount_values(query)
        target = amounts[0] if amounts else float(args.get("target_amount", 10_000_000))
        duration = years(query, float(args.get("years", 10)))
        rate = pct(query, float(args.get("annual_rate_pct", 12)))
        sip = required_sip(target, rate, duration)
        return {
            "target": round(target),
            "required_monthly_sip": round(sip),
            "result": (
                f"Goal plan | Target corpus: {fmt_inr(target)} | Horizon: {duration:.1f} years | "
                f"Return: {rate:.2f}% | Required monthly SIP: {fmt_inr(sip)}"
            ),
        }


class ScreenerTool(BaseTool):
    @property
    def name(self) -> str:
        return "screener"

    @property
    def description(self) -> str:
        return "Fetch simple stock quote/fundamental snapshot using yfinance fallback."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        raw = str(args.get("symbol") or "").upper().replace(".NS", "").replace(".BO", "")
        if not raw:
            return {"result": "No symbol provided."}
        # Resolve company name → correct NSE ticker (e.g. HCL → HCLTECH)
        symbol = _resolve_nse_ticker(raw)
        try:
            import yfinance as yf

            # Do NOT pass a custom session — yfinance >=0.2.52 uses curl_cffi
            # internally for TLS fingerprint mimicking; passing requests.Session breaks it.
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "lastPrice", None)
            previous = getattr(info, "previous_close", None) or getattr(info, "previousClose", None)
            if not price:
                full = ticker.info or {}
                price = full.get("currentPrice") or full.get("regularMarketPrice")
                previous = full.get("previousClose") or previous
            change = ""
            if price and previous:
                change = f" | Change: {(float(price) - float(previous)) / float(previous) * 100:.2f}%"
            if not price:
                return {"symbol": symbol, "result": f"{symbol} price unavailable — market may be closed or data delayed."}
            return {
                "symbol": symbol,
                "price": price,
                "previous_close": previous,
                "result": f"{symbol} | NSE | Price: {fmt_inr(float(price))}{change} | Source: yfinance",
            }
        except Exception as exc:
            return {"symbol": symbol, "result": f"{symbol} lookup failed: {exc}"}


class PortfolioMultiAgentTool(BaseTool):
    @property
    def name(self) -> str:
        return "portfolio_multi_agent"

    @property
    def description(self) -> str:
        return "BanyanTree market workflow for comparing listed stocks and summarising risks."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "symbols": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        symbols = [str(s).upper() for s in args.get("symbols", []) if str(s).strip()]
        symbols = symbols or extract_symbols(query)
        if not symbols:
            return {"result": "No stock symbols found. Ask with symbols like TCS, INFY, HDFCBANK."}

        screener = ScreenerTool()
        rows = [screener.execute({"symbol": symbol}) for symbol in symbols[:4]]
        lines = [row.get("result", "") for row in rows]
        risk = simple_risk_score(allocation_equity=70 if "long term" in query.lower() else 45)
        summary = (
            "BanyanTree market workflow | "
            f"Symbols: {', '.join(symbols[:4])} | Risk lens: {risk}. "
            "Use this as a first-pass snapshot, then verify with exchange filings and a SEBI-registered advisor."
        )
        return {
            "symbols": symbols[:4],
            "risk": risk,
            "agent_outputs": {"screener": rows, "risk_guard": risk},
            "result": summary + "\n" + "\n".join(lines),
        }
