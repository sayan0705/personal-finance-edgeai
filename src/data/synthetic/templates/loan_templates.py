"""Loan Q&A template generator for EMI, home loan, and prepayment scenarios."""

from __future__ import annotations

import random

from ..constants import (
    CAR_LOAN_TYPICAL_RATES,
    HOME_LOAN_TYPICAL_RATES,
    PERSONAL_LOAN_TYPICAL_RATES,
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


def calc_emi(principal: float, annual_rate: float, tenure_months: int) -> dict:
    """Calculate EMI using the standard amortisation formula."""
    r = annual_rate / 12
    if r == 0:
        emi = principal / tenure_months
    else:
        emi = principal * r * ((1 + r) ** tenure_months) / (((1 + r) ** tenure_months) - 1)
    total_payment = emi * tenure_months
    total_interest = total_payment - principal
    return {
        "principal": principal,
        "annual_rate_pct": annual_rate * 100,
        "tenure_months": tenure_months,
        "emi": emi,
        "total_payment": total_payment,
        "total_interest": total_interest,
        "interest_to_principal_ratio": total_interest / principal,
    }


class LoanTemplates:
    """Generates loan-related Q&A pairs.

    Args:
        rng: Random instance for reproducibility.
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._generators = [
            self._home_loan_emi,
            self._personal_loan_emi,
            self._car_loan_emi,
            self._prepayment_advice,
            self._loan_comparison,
            self._cibil_score_advice,
        ]

    def generate_sample(self) -> dict[str, str]:
        """Return a dict with ``question`` and ``answer`` keys."""
        gen = self._rng.choice(self._generators)
        return gen()

    # ── scenario generators ───────────────────────────────────────────────────

    def _home_loan_emi(self) -> dict[str, str]:
        principal = self._rng.choice([2_500_000, 3_000_000, 4_000_000, 5_000_000, 7_500_000, 10_000_000])
        rate = self._rng.choice(HOME_LOAN_TYPICAL_RATES)
        tenure_yrs = self._rng.choice([10, 15, 20, 25, 30])
        res = calc_emi(principal, rate, tenure_yrs * 12)

        q = (
            f"What will be my EMI for a home loan of {_fmt_inr(principal)} "
            f"at {rate*100:.1f}% per annum for {tenure_yrs} years?"
        )
        a = (
            f"**Home Loan EMI Calculator**\n\n"
            f"- Loan amount: {_fmt_inr(principal)}\n"
            f"- Interest rate: {rate*100:.1f}% p.a. (floating, subject to RBI repo rate changes)\n"
            f"- Tenure: {tenure_yrs} years ({res['tenure_months']} months)\n\n"
            f"**Your EMI: {_fmt_inr(res['emi'])} per month**\n\n"
            f"- Total amount payable: {_fmt_inr(res['total_payment'])}\n"
            f"- Total interest cost: {_fmt_inr(res['total_interest'])}\n"
            f"- Interest-to-principal ratio: {res['interest_to_principal_ratio']:.1f}× "
            f"(you pay {res['interest_to_principal_ratio']:.1f}× more in interest than principal)\n\n"
            f"**Tax benefits (old regime):**\n"
            f"- Section 80C: Principal repayment up to ₹1,50,000 per year\n"
            f"- Section 24(b): Interest deduction up to ₹2,00,000 per year (self-occupied)\n\n"
            f"**Tip:** Home loan interest rates are typically floating (linked to MCLR or EBLR). "
            f"A 0.5% rate reduction can save {_fmt_inr(res['total_interest'] * 0.05)} over the loan tenure. "
            f"Always compare HDFC, SBI, ICICI, and Axis Bank rates before deciding."
        )
        return {"question": q, "answer": a}

    def _personal_loan_emi(self) -> dict[str, str]:
        principal = self._rng.choice([50_000, 1_00_000, 2_00_000, 3_00_000, 5_00_000])
        rate = self._rng.choice(PERSONAL_LOAN_TYPICAL_RATES)
        tenure_months = self._rng.choice([12, 24, 36, 48, 60])
        res = calc_emi(principal, rate, tenure_months)

        q = (
            f"I need a personal loan of {_fmt_inr(principal)} at {rate*100:.0f}% interest. "
            f"What will be my EMI for {tenure_months} months and total interest cost?"
        )
        a = (
            f"**Personal Loan EMI Calculation**\n\n"
            f"- Loan amount: {_fmt_inr(principal)}\n"
            f"- Interest rate: {rate*100:.0f}% p.a.\n"
            f"- Tenure: {tenure_months} months\n\n"
            f"**Monthly EMI: {_fmt_inr(res['emi'])}**\n\n"
            f"- Total repayment: {_fmt_inr(res['total_payment'])}\n"
            f"- Total interest: {_fmt_inr(res['total_interest'])} "
            f"({res['total_interest']/principal*100:.0f}% of principal)\n\n"
            f"**Should you take a personal loan?**\n"
            f"Personal loans at {rate*100:.0f}% are expensive. Consider:\n"
            f"1. **Loan against FD/insurance policy**: 1–2% over FD rate (~8–9%)\n"
            f"2. **Gold loan**: 9–12% p.a., faster processing\n"
            f"3. **Top-up on existing home loan**: 9–10% p.a. + tax benefit\n"
            f"4. **Credit card EMI**: Check if lower effective rate after fees\n\n"
            f"If you must take a personal loan, ensure EMI doesn't exceed 40% of net monthly income."
        )
        return {"question": q, "answer": a}

    def _car_loan_emi(self) -> dict[str, str]:
        car_value = self._rng.choice([5_00_000, 7_00_000, 10_00_000, 15_00_000])
        down_pct = self._rng.choice([0.10, 0.15, 0.20, 0.25])
        principal = int(car_value * (1 - down_pct))
        rate = self._rng.choice(CAR_LOAN_TYPICAL_RATES)
        tenure_months = self._rng.choice([36, 48, 60, 72, 84])
        res = calc_emi(principal, rate, tenure_months)

        q = (
            f"I'm buying a car worth {_fmt_inr(car_value)} with {int(down_pct*100)}% down payment. "
            f"At {rate*100:.1f}% interest for {tenure_months} months, what is my EMI?"
        )
        a = (
            f"**Car Loan EMI Calculation**\n\n"
            f"- Car price: {_fmt_inr(car_value)}\n"
            f"- Down payment ({int(down_pct*100)}%): {_fmt_inr(int(car_value * down_pct))}\n"
            f"- Loan amount: {_fmt_inr(principal)}\n"
            f"- Interest rate: {rate*100:.1f}% p.a.\n"
            f"- Tenure: {tenure_months} months ({tenure_months//12} years)\n\n"
            f"**Monthly EMI: {_fmt_inr(res['emi'])}**\n\n"
            f"- Total repayment: {_fmt_inr(res['total_payment'])}\n"
            f"- Total interest: {_fmt_inr(res['total_interest'])}\n\n"
            f"**Things to consider:**\n"
            f"- Car loan interest has **no tax benefit** (unlike home loans)\n"
            f"- A car depreciates ~15–25% per year — your asset loses value while you pay interest\n"
            f"- Rule of thumb: EMI should not exceed 15% of net monthly income for a depreciating asset\n"
            f"- Consider a larger down payment to reduce total interest outgo"
        )
        return {"question": q, "answer": a}

    def _prepayment_advice(self) -> dict[str, str]:
        principal = self._rng.choice([3_000_000, 4_000_000, 5_000_000])
        rate = self._rng.choice(HOME_LOAN_TYPICAL_RATES)
        tenure_yrs = self._rng.choice([15, 20, 25])
        prepay = self._rng.choice([2_00_000, 3_00_000, 5_00_000, 10_00_000])
        mf_rate = self._rng.choice([0.12, 0.13, 0.14])

        res = calc_emi(principal, rate, tenure_yrs * 12)

        q = (
            f"I have a home loan of {_fmt_inr(principal)} at {rate*100:.1f}% for {tenure_yrs} years. "
            f"I have {_fmt_inr(prepay)} extra. Should I prepay the loan or invest in mutual funds?"
        )
        after_tax_loan_rate = rate * (1 - 0.30)  # approximate for 30% bracket
        a = (
            f"**Prepayment vs Investment Decision**\n\n"
            f"Your home loan rate: {rate*100:.1f}% p.a.\n"
            f"After-tax effective cost (30% bracket, assuming max Section 24b interest deduction "
            f"of ₹2L already utilised): ~{after_tax_loan_rate*100:.1f}% p.a.\n\n"
            f"**Option 1 — Prepay ₹{prepay//100_000}L on home loan:**\n"
            f"- Guaranteed return equal to your loan interest rate: {rate*100:.1f}%\n"
            f"- Reduces remaining tenure / EMI burden\n"
            f"- Risk-free (debt reduction is always positive)\n\n"
            f"**Option 2 — Invest in equity mutual fund (assumed {mf_rate*100:.0f}% CAGR):**\n"
            f"- Expected pre-tax return: {mf_rate*100:.0f}% CAGR\n"
            f"- After LTCG tax (12.5% on gains > ₹1L/year): ~{mf_rate*100 - 1.5:.1f}% effective\n"
            f"- Since {mf_rate*100:.0f}% > {rate*100:.1f}%, investment COULD generate more wealth\n\n"
            f"**Recommendation:**\n"
            f"- If your loan rate > 9%: **Prepay** (guaranteed savings > likely market returns after tax)\n"
            f"- If your loan rate < 8.5%: **Invest** (equity likely to outperform after tax)\n"
            f"- At {rate*100:.1f}%: Consider a **50-50 split** — prepay some principal AND invest the rest\n\n"
            f"Additional factor: Home loan interest relief under 24(b) reduces effective rate for high earners. "
            f"Consult a CA to model your exact scenario."
        )
        return {"question": q, "answer": a}

    def _loan_comparison(self) -> dict[str, str]:
        q = "I need ₹3 lakhs urgently. Should I take a personal loan, credit card loan, or loan against FD?"
        a = (
            "**Loan Options Comparison for ₹3,00,000**\n\n"
            "| Option | Typical Rate | Processing | Tax Benefit | Risk |\n"
            "|---|---|---|---|---|\n"
            "| Personal loan | 12–18% p.a. | 1–3 days | None | Low |\n"
            "| Credit card EMI | 13–24% p.a. | Instant | None | Low |\n"
            "| Gold loan | 9–12% p.a. | Same day | None | Lose gold if default |\n"
            "| Loan against FD | FD rate + 1–2% (~8–10%) | Same day | None | Lose FD if default |\n"
            "| Loan against PPF | 1% over PPF rate (~8.1%) | 1 week | None | Limited amount |\n"
            "| Top-up home loan | 9–10.5% p.a. | 5–7 days | Yes (24b) | Home at risk |\n\n"
            "**Best choice for ₹3L:**\n"
            "1. **Loan against FD** (if you have FDs): cheapest option, same-day approval, "
            "FD continues to earn interest\n"
            "2. **Gold loan**: if you have gold idle at home, converts unproductive asset to liquidity\n"
            "3. **Personal loan**: only if no FD/gold available; compare processing fees and "
            "prepayment charges across HDFC, SBI, Axis, ICICI\n\n"
            "**Avoid:** Credit card loan unless you can repay within 3 months (highest effective cost)."
        )
        return {"question": q, "answer": a}

    def _cibil_score_advice(self) -> dict[str, str]:
        q = "My CIBIL score is 680. How does this affect my home loan eligibility and interest rate?"
        a = (
            "**CIBIL Score Impact on Home Loan**\n\n"
            "| CIBIL Score | Eligibility | Interest Rate Impact |\n"
            "|---|---|---|\n"
            "| 750–900 | Excellent — easy approval | Lowest rates offered |\n"
            "| 700–749 | Good — approval likely | Standard rates |\n"
            "| 650–699 | Fair — conditional approval | Higher rate (0.25–0.75% premium) |\n"
            "| Below 650 | Difficult — rejection likely | Very high rate or rejection |\n\n"
            f"**Your score: 680 (Fair)**\n"
            "- Home loan approval is possible but you may face:\n"
            "  - Higher interest rate (0.25–0.50% above best available rate)\n"
            "  - Lower loan-to-value ratio (bank may fund 70% instead of 80%)\n"
            "  - Request for additional collateral or co-applicant\n\n"
            "**How to improve your CIBIL score:**\n"
            "1. Pay all EMIs and credit card bills on or before due date\n"
            "2. Keep credit card utilisation below 30% of limit\n"
            "3. Don't apply for multiple loans simultaneously (hard inquiries hurt score)\n"
            "4. Maintain a mix of secured (home/car) and unsecured (credit card) credit\n"
            "5. Check your CIBIL report for errors at cibil.com (free once per year)\n\n"
            "With consistent repayment behaviour, you can improve to 730+ in 12–18 months."
        )
        return {"question": q, "answer": a}
