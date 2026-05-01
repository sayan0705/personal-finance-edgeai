"""Investment Q&A template generator for SIP, mutual funds, PPF, NPS scenarios."""

from __future__ import annotations

import math
import random
from typing import Any

from ..constants import (
    DEDUCTION_80C_LIMIT,
    ELSS_LOCK_IN_YEARS,
    PPF_INTEREST_RATE,
    PPF_MAX_ANNUAL_CONTRIBUTION,
    PPF_TENURE_YEARS,
    TYPICAL_EQUITY_MF_CAGR,
)


def _fmt_inr(amount: float) -> str:
    n = int(round(amount))
    s = str(n)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts: list[str] = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return f"₹{','.join(parts)},{last3}"


def calc_sip_maturity(monthly_amount: float, annual_rate: float, years: int) -> dict[str, Any]:
    """Calculate SIP maturity value using the standard compound growth formula."""
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        maturity = monthly_amount * n
    else:
        maturity = monthly_amount * ((((1 + r) ** n) - 1) / r) * (1 + r)
    invested = monthly_amount * n
    returns = maturity - invested
    return {
        "monthly_amount": monthly_amount,
        "annual_rate_pct": annual_rate * 100,
        "years": years,
        "n_months": n,
        "total_invested": invested,
        "total_returns": returns,
        "maturity_amount": maturity,
    }


def calc_lumpsum_maturity(amount: float, annual_rate: float, years: int) -> dict[str, Any]:
    """Calculate lump-sum compound growth."""
    maturity = amount * ((1 + annual_rate) ** years)
    return {
        "amount": amount,
        "annual_rate_pct": annual_rate * 100,
        "years": years,
        "maturity_amount": maturity,
        "returns": maturity - amount,
    }


class InvestmentTemplates:
    """Generates investment-related Q&A pairs.

    Args:
        rng: Random instance for reproducibility.
    """

    _SIP_AMOUNTS = [1_000, 2_000, 3_000, 5_000, 7_500, 10_000, 15_000, 20_000, 25_000, 50_000]
    _DURATIONS = [3, 5, 7, 10, 15, 20, 25, 30]

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._generators = [
            self._sip_returns,
            self._lumpsum_returns,
            self._elss_vs_ppf,
            self._sip_vs_lumpsum,
            self._ppf_maturity,
            self._mf_categories,
        ]

    def generate_sample(self) -> dict[str, str]:
        """Return a dict with ``question`` and ``answer`` keys."""
        gen = self._rng.choice(self._generators)
        return gen()

    # ── scenario generators ───────────────────────────────────────────────────

    def _sip_returns(self) -> dict[str, str]:
        amt = self._rng.choice(self._SIP_AMOUNTS)
        yrs = self._rng.choice(self._DURATIONS)
        rate = self._rng.choice(TYPICAL_EQUITY_MF_CAGR)
        res = calc_sip_maturity(amt, rate, yrs)

        q = (
            f"If I invest {_fmt_inr(amt)} per month in a SIP for {yrs} years at "
            f"an assumed CAGR of {rate*100:.0f}%, what will be my maturity amount?"
        )
        a = (
            f"**SIP Calculator — FY 2024-25**\n\n"
            f"- Monthly investment: {_fmt_inr(amt)}\n"
            f"- Investment duration: {yrs} years ({res['n_months']} months)\n"
            f"- Expected CAGR: {rate*100:.0f}% per annum\n\n"
            f"**Results:**\n"
            f"- Total amount invested: {_fmt_inr(res['total_invested'])}\n"
            f"- Estimated returns: {_fmt_inr(res['total_returns'])}\n"
            f"- **Maturity value: {_fmt_inr(res['maturity_amount'])}**\n\n"
            f"This assumes returns compound monthly at {rate*100/12:.3f}% per month. "
            f"Equity mutual funds carry market risk; actual returns will vary. "
            f"Past performance is not a guarantee of future results.\n\n"
            f"**Tax on redemption:** LTCG over ₹1,00,000 per year is taxed at 12.5% "
            f"(held > 12 months). Plan redemptions to utilise the annual ₹1 L exemption."
        )
        return {"question": q, "answer": a}

    def _lumpsum_returns(self) -> dict[str, str]:
        amt = self._rng.choice([50_000, 1_00_000, 2_00_000, 5_00_000, 10_00_000])
        yrs = self._rng.choice([3, 5, 7, 10, 15])
        rate = self._rng.choice(TYPICAL_EQUITY_MF_CAGR)
        res = calc_lumpsum_maturity(amt, rate, yrs)

        q = (
            f"I want to invest {_fmt_inr(amt)} as a lump sum in an equity mutual fund. "
            f"What will it grow to in {yrs} years at {rate*100:.0f}% CAGR?"
        )
        a = (
            f"**Lump-Sum Investment Calculator**\n\n"
            f"- Investment amount: {_fmt_inr(amt)}\n"
            f"- Duration: {yrs} years\n"
            f"- Assumed CAGR: {rate*100:.0f}%\n\n"
            f"**Results:**\n"
            f"- **Maturity value: {_fmt_inr(res['maturity_amount'])}**\n"
            f"- Wealth gained: {_fmt_inr(res['returns'])}\n"
            f"- Wealth multiple: {res['maturity_amount']/amt:.1f}×\n\n"
            f"**Tip:** For lump-sum investments in volatile markets, consider "
            f"Systematic Transfer Plans (STP) — park the amount in a liquid fund "
            f"and set weekly/monthly transfers to an equity fund to average your entry price."
        )
        return {"question": q, "answer": a}

    def _elss_vs_ppf(self) -> dict[str, str]:
        q = "Should I invest in ELSS or PPF for Section 80C tax saving in FY 2024-25? What are the key differences?"
        a = (
            "**ELSS vs PPF — Tax Saving Comparison**\n\n"
            "| Feature | ELSS | PPF |\n"
            "|---|---|---|\n"
            "| Category | Equity mutual fund | Government-backed debt |\n"
            "| Lock-in | 3 years | 15 years (extendable) |\n"
            "| Returns | Market-linked (~10–14% historically) | 7.1% p.a. (guaranteed, tax-free) |\n"
            "| Risk | High (equity market risk) | Nil (sovereign guarantee) |\n"
            "| Tax on returns | LTCG > ₹1L taxed at 12.5% | Completely tax-free (EEE) |\n"
            "| Max investment | No limit (80C allows ₹1.5L) | ₹1,50,000 per year |\n"
            "| Liquidity | After 3 years | Partial withdrawal from year 7 |\n"
            "| SIP possible | Yes | Yes (min ₹500/year) |\n\n"
            "**Recommendation:**\n"
            "- Young investors with 10+ year horizon and high risk tolerance → **ELSS** "
            "(higher potential returns, shorter lock-in)\n"
            "- Conservative investors or those within 10 years of goal → **PPF** "
            "(guaranteed returns, sovereign safety, EEE tax status)\n"
            "- Ideal: Mix both for a balanced tax-saving portfolio under 80C\n\n"
            "Both are unavailable as deductions under the new tax regime."
        )
        return {"question": q, "answer": a}

    def _sip_vs_lumpsum(self) -> dict[str, str]:
        q = "Is it better to invest via SIP or lump sum in equity mutual funds? Which gives higher returns?"
        a = (
            "**SIP vs Lump Sum — Which is Better?**\n\n"
            "**Mathematically:** Lump sum *can* outperform SIP if you invest at a market low and "
            "hold through a bull run. However, timing the market consistently is extremely difficult.\n\n"
            "**SIP advantages:**\n"
            "- Rupee cost averaging: you buy more units when markets fall, fewer when they rise\n"
            "- Eliminates market timing risk\n"
            "- Enforces financial discipline (automated monthly investment)\n"
            "- Lower cognitive burden — no need to watch the market\n\n"
            "**Lump sum advantages:**\n"
            "- If invested during a market correction (e.g., 20–30% drawdown), "
            "returns can significantly outperform SIP over the same period\n"
            "- Better for investing windfall gains (bonus, inheritance, property sale proceeds)\n\n"
            "**Practical guidance:**\n"
            "- Regular income → **SIP** (align with your salary date)\n"
            "- Lump sum available but market near all-time-high → use **STP** "
            "(Systematic Transfer Plan from liquid fund over 6–12 months)\n"
            "- Lump sum available during a 20%+ correction → **Lump sum** directly into equity"
        )
        return {"question": q, "answer": a}

    def _ppf_maturity(self) -> dict[str, str]:
        annual = self._rng.choice([50_000, 75_000, 1_00_000, 1_50_000])
        rate = PPF_INTEREST_RATE
        years = PPF_TENURE_YEARS
        maturity = sum(annual * ((1 + rate) ** (years - y)) for y in range(years))

        q = (
            f"I plan to invest {_fmt_inr(annual)} per year in PPF for {years} years. "
            "What will be my maturity amount? Is the return tax-free?"
        )
        a = (
            f"**PPF (Public Provident Fund) — {years}-Year Projection**\n\n"
            f"- Annual contribution: {_fmt_inr(annual)}\n"
            f"- Current interest rate: {rate*100:.1f}% p.a. (compounded annually)\n"
            f"- Tenure: {years} years\n\n"
            f"- Total invested: {_fmt_inr(annual * years)}\n"
            f"- **Estimated maturity value: {_fmt_inr(maturity)}**\n"
            f"- Tax-free interest earned: {_fmt_inr(maturity - annual * years)}\n\n"
            f"**Tax status (EEE — Exempt, Exempt, Exempt):**\n"
            f"1. Contribution: Deductible under Section 80C (up to ₹1.5L, old regime only)\n"
            f"2. Interest: Completely tax-free each year\n"
            f"3. Maturity amount: Fully tax-free\n\n"
            f"Note: PPF interest rate is revised quarterly by the government. "
            f"The {rate*100:.1f}% rate is as of Q1 FY 2024-25; it may change. "
            f"Partial withdrawals are allowed from Year 7 onwards."
        )
        return {"question": q, "answer": a}

    def _mf_categories(self) -> dict[str, str]:
        q = "What are the different categories of mutual funds in India and how should I choose between them?"
        a = (
            "**Mutual Fund Categories — SEBI Classification**\n\n"
            "**Equity Funds** (high risk, long-term ≥ 5 years):\n"
            "- Large-cap: Top 100 companies by market cap — lower volatility\n"
            "- Mid-cap: 101–250 companies — higher growth potential\n"
            "- Small-cap: 251+ companies — highest risk and potential return\n"
            "- Flexi-cap: Fund manager decides allocation freely\n"
            "- ELSS: 80% in equity + 80C tax benefit + 3-year lock-in\n\n"
            "**Debt Funds** (lower risk, 1–3 year horizon):\n"
            "- Liquid funds: Overnight to 91-day instruments — for emergency funds\n"
            "- Short duration: 1–3 year maturity — better returns than savings account\n"
            "- Corporate bond: Higher-rated company debt\n\n"
            "**Hybrid Funds** (balanced risk):\n"
            "- Aggressive hybrid: 65–80% equity + 20–35% debt\n"
            "- Conservative hybrid: 10–25% equity + 75–90% debt\n"
            "- Arbitrage funds: Low-risk, equity taxation benefit\n\n"
            "**How to choose:**\n"
            "- < 1 year horizon → Liquid / money market fund\n"
            "- 1–3 years → Short duration debt fund\n"
            "- 3–5 years → Aggressive hybrid or large-cap equity\n"
            "- 5+ years → Flexi-cap or diversified equity fund\n\n"
            "Always check: expense ratio, fund manager tenure, rolling returns, and "
            "Sharpe ratio before investing. Consult a SEBI-registered investment advisor."
        )
        return {"question": q, "answer": a}
