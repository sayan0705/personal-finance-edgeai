"""Gaps 5 & 6 — MF portfolio XIRR calculation and CAS statement parsing."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .base import BaseGapGenerator

_FUND_DB: dict[str, dict[str, Any]] = {
    "Parag Parikh Flexi Cap": {"cat": "Flexi Cap", "nav_growth": 0.15},
    "HDFC Mid-Cap Opportunities": {"cat": "Mid Cap", "nav_growth": 0.18},
    "SBI Small Cap": {"cat": "Small Cap", "nav_growth": 0.22},
    "Axis Bluechip": {"cat": "Large Cap", "nav_growth": 0.12},
    "Mirae Asset Large Cap": {"cat": "Large Cap", "nav_growth": 0.13},
    "Kotak Emerging Equity": {"cat": "Mid Cap", "nav_growth": 0.17},
    "Nippon India Small Cap": {"cat": "Small Cap", "nav_growth": 0.20},
    "UTI Nifty 50 Index": {"cat": "Index", "nav_growth": 0.12},
    "Mirae Asset Tax Saver": {"cat": "ELSS", "nav_growth": 0.14},
    "HDFC Short Term Debt": {"cat": "Debt", "nav_growth": 0.07},
}

_SIP_AMOUNTS = [1000, 2000, 3000, 5000, 7500, 10000]
_TODAY = date(2025, 1, 1)  # fixed anchor for reproducibility


def _xirr(cashflows: list[float], dates: list[date], guess: float = 0.10) -> float | None:
    """Newton-Raphson XIRR solver for SIP portfolios.

    Args:
        cashflows: List of signed cash-flow amounts (negative = investment).
        dates: List of corresponding dates.
        guess: Initial rate guess.

    Returns:
        Annualised XIRR as a decimal, or None if it fails to converge.
    """
    if not cashflows or len(cashflows) != len(dates):
        return None
    d0 = min(dates)

    def npv(rate: float) -> float:
        return sum(
            cf / (1 + rate) ** ((d - d0).days / 365.0)
            for cf, d in zip(cashflows, dates)
        )

    def dnpv(rate: float) -> float:
        return sum(
            -cf * ((d - d0).days / 365.0) / (1 + rate) ** ((d - d0).days / 365.0 + 1)
            for cf, d in zip(cashflows, dates)
        )

    rate = guess
    for _ in range(200):
        nv = npv(rate)
        dn = dnpv(rate)
        if abs(dn) < 1e-12:
            break
        rate -= nv / dn
        if abs(nv) < 1e-6:
            break
    return rate if -0.99 < rate < 10.0 else None


class XIRRPortfolioGenerator(BaseGapGenerator):
    """Generates MF portfolio XIRR analysis Q&A pairs (Gap 5).

    Simulates 1–4 SIP portfolios with realistic NAV growth, computes per-fund and
    overall XIRR using Newton-Raphson, and produces actionable commentary.

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance.
    """

    gap_name = "xirr_portfolio"
    task_type = "investment_analysis"
    layer = "L3_personal_finance"
    source_dataset = "synthetic_xirr"

    def _generate_one(self, idx: int) -> dict[str, Any]:
        num_funds = self.rng.randint(1, 4)
        funds = self.rng.sample(list(_FUND_DB.keys()), num_funds)

        portfolio_lines: list[str] = []
        all_cfs: list[float] = []
        all_dates: list[date] = []

        for fname in funds:
            finfo = _FUND_DB[fname]
            sip_amt = self.rng.choice(_SIP_AMOUNTS)
            months = self.rng.randint(12, 60)
            nav_start = round(self.rng.uniform(20, 200), 2)
            start = _TODAY - timedelta(days=months * 30)

            cfs: list[float] = []
            ds: list[date] = []
            units = 0.0
            for m in range(months):
                d = start + timedelta(days=m * 30)
                growth = 1 + finfo["nav_growth"] * self.rng.uniform(0.5, 1.5) / 12
                nav = nav_start * (growth ** m)
                units += sip_amt / nav
                cfs.append(-float(sip_amt))
                ds.append(d)

            current_nav = nav_start * (
                (1 + finfo["nav_growth"] * self.rng.uniform(0.7, 1.3)) ** (months / 12)
            )
            current_val = round(units * current_nav, 0)
            invested = sip_amt * months
            cfs.append(current_val)
            ds.append(_TODAY)

            fund_xirr = _xirr(cfs, ds)
            xirr_str = f"{fund_xirr*100:.1f}%" if fund_xirr else "N/A"
            portfolio_lines.append(
                f"  {fname} ({finfo['cat']}): SIP ₹{sip_amt:,} × {months}m "
                f"= ₹{invested:,} invested → ₹{int(current_val):,} ({xirr_str} XIRR)"
            )
            all_cfs.extend(cfs)
            all_dates.extend(ds)

        overall = _xirr(all_cfs, all_dates)
        total_inv = int(sum(-c for c in all_cfs if c < 0))
        total_val = int(sum(c for c in all_cfs if c > 0))
        abs_ret = (total_val - total_inv) / total_inv * 100 if total_inv else 0.0

        port_str = "\n".join(portfolio_lines)
        q = f"Here is my mutual fund SIP portfolio:\n{port_str}\n\nCalculate my portfolio XIRR and analyse performance."

        if overall and overall > 0.12:
            verdict = "✅ XIRR above 12% — outperforming the typical Nifty 50 long-run CAGR."
        elif overall:
            verdict = "⚠️ XIRR below 12%. Review fund selection vs. benchmark."
        else:
            verdict = "Unable to compute overall XIRR."

        a = (
            f"Portfolio XIRR Analysis:\n\n{port_str}\n\n"
            f"Total Invested:   ₹{total_inv:,}\n"
            f"Current Value:    ₹{total_val:,}\n"
            f"Absolute Return:  {abs_ret:.1f}%\n"
            f"Portfolio XIRR:   {overall*100:.1f}% p.a.\n\n"
            f"{verdict}\n\n"
            f"Note: XIRR accounts for the timing of each SIP instalment — it is more accurate "
            f"than simple CAGR for systematic investments.\n\n"
            f"Disclaimer: Past performance does not guarantee future returns. "
            f"Consult a SEBI-registered advisor before making changes."
        ) if overall else (
            f"Portfolio Summary:\n\n{port_str}\n\n"
            f"Total Invested: ₹{total_inv:,} | Current Value: ₹{total_val:,} | "
            f"Absolute Return: {abs_ret:.1f}%\n\nXIRR computation did not converge — "
            f"verify input dates and amounts."
        )

        return self._make_sample(idx, q, a, difficulty="advanced")


class CASStatementGenerator(BaseGapGenerator):
    """Generates Consolidated Account Statement (CAS) parsing Q&A pairs (Gap 6).

    Builds a realistic CAMS/KFintech-style CAS excerpt with 2–5 folios, computes
    portfolio value and allocation, and answers a portfolio health question.

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance.
    """

    gap_name = "cas_parse"
    task_type = "document_parsing"
    layer = "L3_personal_finance"
    source_dataset = "synthetic_cas"

    def _generate_one(self, idx: int) -> dict[str, Any]:
        n_folios = self.rng.randint(2, 5)
        cas_lines: list[str] = [
            "CONSOLIDATED ACCOUNT STATEMENT (CAS)",
            "Period: 01-Apr-2024 to 31-Mar-2025",
            "PAN: ABCPX0000X  (masked for privacy)",
            "",
        ]
        total_invested = 0
        total_value = 0
        fund_summaries: list[dict[str, Any]] = []

        for _ in range(n_folios):
            fname = self.rng.choice(list(_FUND_DB.keys()))
            finfo = _FUND_DB[fname]
            folio = f"{self.rng.randint(10_000_000, 99_999_999)}/{self.rng.randint(10,99)}"
            n_txns = self.rng.randint(6, 24)
            sip_amt = self.rng.choice(_SIP_AMOUNTS)
            nav_base = round(self.rng.uniform(15, 300), 4)
            units = 0.0

            cas_lines.append(f"Folio: {folio}  |  {fname} - Direct Growth")
            cas_lines.append(
                f"{'Date':<12} {'Description':<25} {'Amount':>10} "
                f"{'NAV':>10} {'Units':>10} {'Balance':>10}"
            )

            for t in range(n_txns):
                txn_date = date(2024, 4, 1) + timedelta(days=t * 30 + self.rng.randint(0, 5))
                nav_now = nav_base * (1 + self.rng.uniform(-0.02, 0.04))
                new_units = round(sip_amt / nav_now, 3)
                units += new_units
                cas_lines.append(
                    f"{txn_date.strftime('%d-%b-%Y'):<12} {'SIP Purchase':<25} "
                    f"{sip_amt:>10,} {nav_now:>10.4f} {new_units:>10.3f} {units:>10.3f}"
                )

            invested = sip_amt * n_txns
            cur_nav = nav_base * (1 + finfo["nav_growth"] * self.rng.uniform(0.5, 1.2))
            value = round(units * cur_nav, 2)
            total_invested += invested
            total_value += value
            gain_pct = (value - invested) / invested * 100
            fund_summaries.append(
                {"fund": fname, "cat": finfo["cat"], "invested": invested,
                 "value": int(value), "gain_pct": round(gain_pct, 1)}
            )
            cas_lines.append(
                f"  Valuation: {units:.3f} units × ₹{cur_nav:.4f} = ₹{value:,.2f}"
            )
            cas_lines.append("")

        # Keep CAS text manageable
        cas_text = "\n".join(cas_lines[:35]) + "\n[...truncated for brevity...]"
        total_gain_pct = (total_value - total_invested) / total_invested * 100

        q = f"Parse my CAS statement and summarise how my portfolio is performing:\n\n{cas_text}"

        summary_lines = "\n".join(
            f"  {fs['fund']}: ₹{fs['invested']:,} → ₹{fs['value']:,} ({fs['gain_pct']:+.1f}%)"
            for fs in fund_summaries
        )
        alloc_lines = "\n".join(
            f"  {fs['cat']}: ₹{fs['value']:,} ({fs['value']/total_value*100:.0f}%)"
            for fs in fund_summaries
        )
        small_mid_count = sum(1 for fs in fund_summaries if fs["cat"] in {"Small Cap", "Mid Cap"})
        risk_note = (
            "\n⚠️ High concentration in small/mid caps — ensure a 7+ year investment horizon."
            if small_mid_count > 2
            else ""
        )

        a = (
            f"CAS Portfolio Summary:\n\n{summary_lines}\n\n"
            f"Total Invested:  ₹{total_invested:,}\n"
            f"Current Value:   ₹{int(total_value):,}\n"
            f"Overall Gain:    ₹{int(total_value-total_invested):,} ({total_gain_pct:.1f}%)\n\n"
            f"Asset Allocation:\n{alloc_lines}{risk_note}\n\n"
            f"Recommendations:\n"
            f"• Review allocation annually — avoid more than 2 funds per category (overlap risk)\n"
            f"• SIP in Direct plans only — save 0.5–1% in expense ratio vs Regular plans\n"
            f"• Check for dividend payouts: switch to Growth option for compounding"
        )

        return self._make_sample(idx, q, a, difficulty="advanced")
