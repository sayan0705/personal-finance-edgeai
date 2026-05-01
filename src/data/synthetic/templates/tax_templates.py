"""Tax Q&A template generator for FY 2024-25 Indian income tax scenarios."""

from __future__ import annotations

import random
from typing import Any

from ..constants import (
    DEDUCTION_80C_LIMIT,
    DEDUCTION_80D_PARENTS_LIMIT,
    DEDUCTION_80D_PARENTS_SENIOR_LIMIT,
    DEDUCTION_80D_SELF_LIMIT,
    DEDUCTION_80CCD1B_LIMIT,
    HEALTH_AND_EDUCATION_CESS,
    INCOME_BRACKETS,
    NEW_REGIME_87A_REBATE_LIMIT,
    NEW_REGIME_87A_REBATE_MAX,
    NEW_REGIME_SLABS,
    NEW_REGIME_STANDARD_DEDUCTION,
    OLD_REGIME_87A_REBATE_LIMIT,
    OLD_REGIME_87A_REBATE_MAX,
    OLD_REGIME_SLABS,
    OLD_REGIME_STANDARD_DEDUCTION,
)


def _calc_slab_tax(taxable_income: int, slabs: list[tuple]) -> int:
    """Calculate tax on taxable_income using the given slab table."""
    tax = 0
    prev_limit = 0
    for limit, rate in slabs:
        if taxable_income <= prev_limit:
            break
        band = min(taxable_income, limit) - prev_limit
        tax += int(band * rate)
        prev_limit = limit
    return tax


def _apply_87a(tax: int, taxable_income: int, limit: int, max_rebate: int) -> int:
    """Apply Section 87A rebate."""
    if taxable_income <= limit:
        return max(0, tax - min(tax, max_rebate))
    return tax


def compute_new_regime_tax(gross_income: int) -> dict[str, Any]:
    """Compute net tax under the new tax regime for FY 2024-25."""
    taxable = max(0, gross_income - NEW_REGIME_STANDARD_DEDUCTION)
    tax_before_87a = _calc_slab_tax(taxable, NEW_REGIME_SLABS)
    tax_after_87a = _apply_87a(
        tax_before_87a, taxable, NEW_REGIME_87A_REBATE_LIMIT, NEW_REGIME_87A_REBATE_MAX
    )
    cess = int(tax_after_87a * HEALTH_AND_EDUCATION_CESS)
    total = tax_after_87a + cess
    effective_rate = (total / gross_income * 100) if gross_income else 0.0
    return {
        "gross_income": gross_income,
        "standard_deduction": NEW_REGIME_STANDARD_DEDUCTION,
        "taxable_income": taxable,
        "tax_before_cess": tax_after_87a,
        "cess": cess,
        "total_tax": total,
        "effective_rate": round(effective_rate, 2),
    }


def compute_old_regime_tax(
    gross_income: int,
    deduction_80c: int = 0,
    deduction_80d_self: int = 0,
    deduction_80d_parents: int = 0,
    other_deductions: int = 0,
) -> dict[str, Any]:
    """Compute net tax under the old tax regime for FY 2024-25."""
    deduction_80c = min(deduction_80c, DEDUCTION_80C_LIMIT)
    deduction_80d_self = min(deduction_80d_self, DEDUCTION_80D_SELF_LIMIT)
    deduction_80d_parents = min(deduction_80d_parents, DEDUCTION_80D_PARENTS_LIMIT)
    total_deductions = (
        OLD_REGIME_STANDARD_DEDUCTION
        + deduction_80c
        + deduction_80d_self
        + deduction_80d_parents
        + other_deductions
    )
    taxable = max(0, gross_income - total_deductions)
    tax_before_87a = _calc_slab_tax(taxable, OLD_REGIME_SLABS)
    tax_after_87a = _apply_87a(
        tax_before_87a, taxable, OLD_REGIME_87A_REBATE_LIMIT, OLD_REGIME_87A_REBATE_MAX
    )
    cess = int(tax_after_87a * HEALTH_AND_EDUCATION_CESS)
    total = tax_after_87a + cess
    effective_rate = (total / gross_income * 100) if gross_income else 0.0
    return {
        "gross_income": gross_income,
        "total_deductions": total_deductions,
        "taxable_income": taxable,
        "tax_before_cess": tax_after_87a,
        "cess": cess,
        "total_tax": total,
        "effective_rate": round(effective_rate, 2),
    }


def _fmt_inr(amount: int) -> str:
    """Format integer as Indian currency string (e.g. ₹8,50,000)."""
    s = str(amount)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return f"₹{','.join(parts)},{last3}"


class TaxTemplates:
    """Generates tax-related Q&A pairs for FY 2024-25.

    Args:
        rng: Random instance for reproducibility.
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._generators = [
            self._regime_comparison,
            self._single_regime_query,
            self._deduction_query,
            self._rebate_query,
            self._ltcg_query,
        ]

    def generate_sample(self) -> dict[str, str]:
        """Return a dict with ``question`` and ``answer`` keys."""
        gen = self._rng.choice(self._generators)
        return gen()

    # ── scenario generators ───────────────────────────────────────────────────

    def _regime_comparison(self) -> dict[str, str]:
        income = self._rng.choice(INCOME_BRACKETS)
        d80c = self._rng.randint(0, DEDUCTION_80C_LIMIT // 10_000) * 10_000
        d80d = self._rng.choice([0, 15_000, 25_000])
        d80d_parents = self._rng.choice([0, 25_000, 50_000])

        new = compute_new_regime_tax(income)
        old = compute_old_regime_tax(income, d80c, d80d, d80d_parents)
        better = "new" if new["total_tax"] < old["total_tax"] else "old"
        saving = abs(new["total_tax"] - old["total_tax"])

        q = (
            f"My annual salary is {_fmt_inr(income)}. "
            f"I invest {_fmt_inr(d80c)} in tax-saving instruments under Section 80C"
        )
        if d80d:
            q += f", pay {_fmt_inr(d80d)} as health insurance premium"
        if d80d_parents:
            q += f", and {_fmt_inr(d80d_parents)} for parents' health insurance"
        q += ". Which income tax regime is better for me in FY 2024-25?"

        a = self._regime_comparison_answer(income, d80c, d80d, d80d_parents, new, old, better, saving)
        return {"question": q, "answer": a}

    def _regime_comparison_answer(
        self,
        income: int,
        d80c: int,
        d80d: int,
        d80d_parents: int,
        new: dict,
        old: dict,
        better: str,
        saving: int,
    ) -> str:
        lines = [
            f"Let me calculate your tax liability under both regimes for FY 2024-25:\n",
            f"**New Tax Regime:**",
            f"  Gross income:          {_fmt_inr(income)}",
            f"  Standard deduction:   -₹75,000 (Budget 2024 enhancement)",
            f"  Taxable income:        {_fmt_inr(new['taxable_income'])}",
            f"  Tax (before cess):     {_fmt_inr(new['tax_before_cess'])}",
            f"  Health & Ed. cess 4%:  {_fmt_inr(new['cess'])}",
            f"  **Total tax (New):     {_fmt_inr(new['total_tax'])}** "
            f"(effective rate: {new['effective_rate']}%)\n",
            f"**Old Tax Regime:**",
            f"  Gross income:          {_fmt_inr(income)}",
            f"  Standard deduction:   -₹50,000",
        ]
        if d80c:
            lines.append(f"  Section 80C deduction:-{_fmt_inr(min(d80c, DEDUCTION_80C_LIMIT))}")
        if d80d:
            lines.append(f"  Section 80D (self):   -{_fmt_inr(d80d)}")
        if d80d_parents:
            lines.append(f"  Section 80D (parents):-{_fmt_inr(d80d_parents)}")
        lines += [
            f"  Total deductions:     -{_fmt_inr(old['total_deductions'])}",
            f"  Taxable income:        {_fmt_inr(old['taxable_income'])}",
            f"  Tax (before cess):     {_fmt_inr(old['tax_before_cess'])}",
            f"  Health & Ed. cess 4%:  {_fmt_inr(old['cess'])}",
            f"  **Total tax (Old):     {_fmt_inr(old['total_tax'])}** "
            f"(effective rate: {old['effective_rate']}%)\n",
            f"**Recommendation:** The **{better} tax regime** saves you {_fmt_inr(saving)} "
            f"in taxes for FY 2024-25.",
        ]
        if new["taxable_income"] <= NEW_REGIME_87A_REBATE_LIMIT:
            lines.append(
                "\nNote: Under Section 87A, since your taxable income under the new regime "
                "is ≤ ₹7,00,000, you qualify for a rebate of up to ₹25,000 — effectively "
                "making your new-regime tax nil."
            )
        lines.append(
            "\nAlways consult a SEBI-registered tax advisor or CA for personalised advice "
            "before filing your return."
        )
        return "\n".join(lines)

    def _single_regime_query(self) -> dict[str, str]:
        income = self._rng.choice(INCOME_BRACKETS)
        regime = self._rng.choice(["new", "old"])
        if regime == "new":
            result = compute_new_regime_tax(income)
            q = f"How much income tax will I pay on {_fmt_inr(income)} annual income under the new tax regime in FY 2024-25?"
            a = (
                f"Under the **new tax regime** for FY 2024-25:\n\n"
                f"- Gross income: {_fmt_inr(income)}\n"
                f"- Less standard deduction: ₹75,000\n"
                f"- **Taxable income: {_fmt_inr(result['taxable_income'])}**\n\n"
                f"Tax slab breakdown:\n"
                + self._new_slab_breakdown(result["taxable_income"])
                + f"\n\n- Tax before cess: {_fmt_inr(result['tax_before_cess'])}\n"
                f"- Add 4% health & education cess: {_fmt_inr(result['cess'])}\n"
                f"- **Total tax payable: {_fmt_inr(result['total_tax'])}** "
                f"(effective rate: {result['effective_rate']}%)"
            )
        else:
            result = compute_old_regime_tax(income)
            q = f"What is the income tax on {_fmt_inr(income)} salary under the old tax regime with no deductions in FY 2024-25?"
            a = (
                f"Under the **old tax regime** (no deductions except standard) for FY 2024-25:\n\n"
                f"- Gross income: {_fmt_inr(income)}\n"
                f"- Less standard deduction: ₹50,000\n"
                f"- **Taxable income: {_fmt_inr(result['taxable_income'])}**\n\n"
                f"Tax slab breakdown:\n"
                + self._old_slab_breakdown(result["taxable_income"])
                + f"\n\n- Tax before cess: {_fmt_inr(result['tax_before_cess'])}\n"
                f"- Add 4% health & education cess: {_fmt_inr(result['cess'])}\n"
                f"- **Total tax payable: {_fmt_inr(result['total_tax'])}** "
                f"(effective rate: {result['effective_rate']}%)"
            )
        return {"question": q, "answer": a}

    def _deduction_query(self) -> dict[str, str]:
        topic = self._rng.choice(["80c", "80d", "nps", "hra"])
        if topic == "80c":
            q = "What investments qualify under Section 80C and what is the maximum deduction limit for FY 2024-25?"
            a = (
                "**Section 80C** allows a maximum deduction of **₹1,50,000** per year under "
                "the old tax regime. Qualifying investments include:\n\n"
                "- **ELSS** (Equity Linked Savings Scheme) mutual funds — 3-year lock-in\n"
                "- **PPF** (Public Provident Fund) — 15-year tenure, tax-free returns\n"
                "- **EPF** (Employee Provident Fund) — mandatory for salaried employees\n"
                "- **NSC** (National Savings Certificate)\n"
                "- **5-year tax-saving FD** with banks/post office\n"
                "- **Life insurance premiums** (own, spouse, children)\n"
                "- **ULIP** premiums\n"
                "- **Sukanya Samriddhi Yojana** (for girl child)\n"
                "- **Home loan principal repayment**\n"
                "- **Tuition fees** (up to 2 children)\n\n"
                "Note: Section 80C deductions are **NOT available under the new tax regime** "
                "(except employer NPS contribution under 80CCD(2))."
            )
        elif topic == "80d":
            q = "How much can I claim under Section 80D for health insurance premiums in FY 2024-25?"
            a = (
                "**Section 80D** deductions for health insurance premiums (old regime only):\n\n"
                "| Category | Maximum Deduction |\n"
                "|---|---|\n"
                "| Self, spouse & children (age < 60) | ₹25,000 |\n"
                "| Self, spouse & children (senior citizen ≥60) | ₹50,000 |\n"
                "| Parents (age < 60) | ₹25,000 |\n"
                "| Parents (senior citizen ≥60) | ₹50,000 |\n\n"
                "**Maximum possible:** ₹1,00,000 (both self and parents are senior citizens)\n\n"
                "You can also claim ₹5,000 for preventive health check-ups within the above limits.\n\n"
                "Under Section 80D, you can claim for premiums paid via cheque, bank transfer, "
                "or UPI (cash payments are not eligible)."
            )
        elif topic == "nps":
            q = "What is the tax benefit of investing in NPS (National Pension System) in FY 2024-25?"
            a = (
                "NPS offers **triple tax benefits** under the old regime:\n\n"
                "1. **Section 80CCD(1):** Up to ₹1,50,000 (part of the 80C umbrella)\n"
                "2. **Section 80CCD(1B):** Additional ₹50,000 OVER AND ABOVE the 80C limit — "
                "this is exclusive to NPS and a key advantage\n"
                "3. **Section 80CCD(2):** Employer NPS contribution up to 10% of basic+DA is "
                "fully deductible — **this benefit is also available under the new regime**\n\n"
                "**On maturity:**\n"
                "- 60% of the corpus can be withdrawn as lump sum — completely tax-free\n"
                "- 40% must be used to buy an annuity (annuity income is taxed at slab rates)\n\n"
                "NPS Tier-1 investments have a lock-in till age 60 with partial withdrawal rules."
            )
        else:  # hra
            q = "How is HRA (House Rent Allowance) exemption calculated for a salaried employee?"
            a = (
                "**HRA exemption** under Section 10(13A) is the **minimum of**:\n\n"
                "1. Actual HRA received from employer\n"
                "2. Actual rent paid minus 10% of (Basic + DA)\n"
                "3. 50% of (Basic + DA) for metro cities (Mumbai, Delhi, Kolkata, Chennai); "
                "40% for other cities\n\n"
                "**Example:** Basic = ₹50,000/month, HRA = ₹20,000/month, Rent paid = ₹18,000/month (Delhi)\n"
                "- Actual HRA: ₹20,000\n"
                "- Rent minus 10% of basic: ₹18,000 − ₹5,000 = ₹13,000\n"
                "- 50% of basic (metro): ₹25,000\n"
                "- **Exempt HRA = ₹13,000/month** (minimum of the three)\n\n"
                "Note: HRA exemption is **not available** under the new tax regime.\n"
                "If you live in your own house or don't receive HRA, you cannot claim this."
            )
        return {"question": q, "answer": a}

    def _rebate_query(self) -> dict[str, str]:
        q = "What is Section 87A rebate and who is eligible for it in FY 2024-25?"
        a = (
            "**Section 87A** provides a direct rebate on your income tax liability:\n\n"
            "| Regime | Taxable Income Limit | Maximum Rebate | Effective Result |\n"
            "|---|---|---|---|\n"
            "| New regime | ≤ ₹7,00,000 | ₹25,000 | Zero tax |\n"
            "| Old regime | ≤ ₹5,00,000 | ₹12,500 | Zero tax |\n\n"
            "**Key points:**\n"
            "- The rebate is applied on the tax calculated **before adding the 4% cess**\n"
            "- If your taxable income exceeds the limit by even ₹1, you get NO rebate "
            "(marginal relief provisions apply in some cases)\n"
            "- Special rate income (e.g., LTCG under Section 112A on equity) is NOT eligible "
            "for 87A rebate from FY 2023-24 onwards\n\n"
            "**Budget 2024 impact:** The new regime standard deduction increased to ₹75,000, "
            "meaning a gross salary up to ₹7,75,000 can result in zero tax under the new regime "
            "after the 87A rebate."
        )
        return {"question": q, "answer": a}

    def _ltcg_query(self) -> dict[str, str]:
        gain = self._rng.choice([50_000, 80_000, 1_00_000, 1_50_000, 2_00_000, 5_00_000])
        q = (
            f"I made a long-term capital gain of {_fmt_inr(gain)} from selling equity mutual funds. "
            "How much tax do I pay in FY 2024-25?"
        )
        exempt = 1_00_000
        taxable_ltcg = max(0, gain - exempt)
        tax = int(taxable_ltcg * 0.125)
        cess = int(tax * 0.04)
        total = tax + cess

        a = (
            f"Equity mutual fund LTCG (held > 12 months) is taxed under **Section 112A** "
            f"at **12.5%** (Budget 2024, up from 10%) with the first ₹1,00,000 exempt.\n\n"
            f"**Your calculation:**\n"
            f"- Total LTCG: {_fmt_inr(gain)}\n"
            f"- Less exemption (Section 112A): -₹1,00,000\n"
            f"- **Taxable LTCG: {_fmt_inr(taxable_ltcg)}**\n"
            f"- Tax @ 12.5%: {_fmt_inr(tax)}\n"
            f"- Add 4% cess: {_fmt_inr(cess)}\n"
            f"- **Total LTCG tax: {_fmt_inr(total)}**\n\n"
        )
        if gain <= exempt:
            a = (
                f"Your LTCG of {_fmt_inr(gain)} from equity mutual funds is completely **tax-free**!\n\n"
                "Under Section 112A, the first ₹1,00,000 of long-term capital gains from equity "
                "mutual funds (STT-paid, held > 12 months) is exempt from tax each financial year. "
                "Since your gain is within this limit, you pay ₹0 in LTCG tax."
            )
        else:
            a += (
                "Note: Section 87A rebate is NOT available on LTCG taxed at special rates. "
                "You pay this LTCG tax even if your total income is below the basic exemption limit."
            )
        return {"question": q, "answer": a}

    # ── formatting helpers ────────────────────────────────────────────────────

    @staticmethod
    def _new_slab_breakdown(taxable: int) -> str:
        slabs = [
            (0, 300_000, "0%"),
            (300_000, 700_000, "5%"),
            (700_000, 1_000_000, "10%"),
            (1_000_000, 1_200_000, "15%"),
            (1_200_000, 1_500_000, "20%"),
            (1_500_000, float("inf"), "30%"),
        ]
        lines = []
        for low, high, rate in slabs:
            if taxable <= low:
                break
            band = min(taxable, high) - low
            if band <= 0:
                continue
            tax_on_band = int(band * float(rate.strip("%")) / 100)
            lines.append(
                f"  {_fmt_inr(int(low)+1) if low else '₹0'} – "
                f"{_fmt_inr(int(high)) if high != float('inf') else 'above'}: "
                f"{rate} on {_fmt_inr(int(band))} = {_fmt_inr(tax_on_band)}"
            )
        return "\n".join(lines)

    @staticmethod
    def _old_slab_breakdown(taxable: int) -> str:
        slabs = [
            (0, 250_000, "0%"),
            (250_000, 500_000, "5%"),
            (500_000, 1_000_000, "20%"),
            (1_000_000, float("inf"), "30%"),
        ]
        lines = []
        for low, high, rate in slabs:
            if taxable <= low:
                break
            band = min(taxable, high) - low
            if band <= 0:
                continue
            tax_on_band = int(band * float(rate.strip("%")) / 100)
            lines.append(
                f"  {_fmt_inr(int(low)+1) if low else '₹0'} – "
                f"{_fmt_inr(int(high)) if high != float('inf') else 'above'}: "
                f"{rate} on {_fmt_inr(int(band))} = {_fmt_inr(tax_on_band)}"
            )
        return "\n".join(lines)
