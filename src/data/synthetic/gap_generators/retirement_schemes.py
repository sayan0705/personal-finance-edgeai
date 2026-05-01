"""Gap 12 — NPS / PPF / EPF retirement corpus projection generator."""

from __future__ import annotations

from typing import Any

from ..constants import EPF_INTEREST_RATE, PPF_INTEREST_RATE
from .base import BaseGapGenerator

_MONTHLY_SALARY_CHOICES = list(range(30_000, 300_001, 10_000))
_NPS_MONTHLY_CHOICES = [1_000, 2_000, 3_000, 5_000, 10_000]
_PPF_ANNUAL_CHOICES = [50_000, 100_000, 150_000]
_RETIREMENT_AGES = [55, 58, 60]
_NPS_ALLOCATIONS = {
    "aggressive": 11.0,   # ~80% equity
    "moderate": 9.5,      # ~50% equity
    "conservative": 8.0,  # govt bonds heavy
}


def _fv_monthly_sip(monthly: float, annual_rate: float, years: int) -> float:
    """Future value of regular monthly investment at ``annual_rate``."""
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        return monthly * n
    return monthly * ((1 + r) ** n - 1) / r


def _fv_annual_sip(annual: float, annual_rate: float, years: int) -> float:
    """Future value of annual contribution at end of each year."""
    if annual_rate == 0:
        return annual * years
    return annual * ((1 + annual_rate) ** years - 1) / annual_rate


class RetirementSchemesGenerator(BaseGapGenerator):
    """Generates NPS / PPF / EPF corpus projection Q&A pairs (Gap 12).

    Combines all three instruments into a unified retirement picture, applies the
    4% safe withdrawal rule to estimate monthly pension, and comments on adequacy.
    Uses FY 2024-25 rates from constants.

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance.
    """

    gap_name = "retirement"
    task_type = "retirement_planning"
    layer = "L3_personal_finance"
    source_dataset = "synthetic_nps_ppf_epf"

    def _generate_one(self, idx: int) -> dict[str, Any]:
        age = self.rng.randint(25, 55)
        monthly_salary = self.rng.choice(_MONTHLY_SALARY_CHOICES)
        basic = int(monthly_salary * self.rng.uniform(0.40, 0.60))
        retire_age = self.rng.choice(_RETIREMENT_AGES)
        years = max(1, retire_age - age)

        ppf_annual = self.rng.choice(_PPF_ANNUAL_CHOICES)
        nps_monthly = self.rng.choice(_NPS_MONTHLY_CHOICES)
        nps_alloc = self.rng.choice(list(_NPS_ALLOCATIONS.keys()))
        nps_rate = _NPS_ALLOCATIONS[nps_alloc] / 100

        # EPF: employee 12% of basic + employer 3.67% to EPF (rest to EPS)
        epf_employee = int(basic * 0.12)
        epf_employer = int(basic * 0.0367)
        epf_monthly = epf_employee + epf_employer
        epf_corpus = _fv_monthly_sip(epf_monthly, EPF_INTEREST_RATE, years)

        ppf_corpus = _fv_annual_sip(ppf_annual, PPF_INTEREST_RATE, years)
        nps_corpus = _fv_monthly_sip(nps_monthly, nps_rate, years)

        nps_lumpsum = nps_corpus * 0.60
        nps_annuity_corpus = nps_corpus * 0.40
        nps_extra_tax_saving = int(min(nps_monthly * 12, 50_000) * 0.30)

        total_corpus = epf_corpus + ppf_corpus + nps_corpus
        monthly_pension = total_corpus * 0.04 / 12
        today_value_pension = monthly_pension / (1.06 ** years)  # 6% inflation

        adequacy = (
            "✅ On track — projected pension covers 50%+ of current salary."
            if monthly_pension >= monthly_salary * 0.5
            else "⚠️ Gap detected — consider increasing NPS/PPF contributions."
        )

        q = (
            f"I am {age} years old, basic salary ₹{basic:,}/month. "
            f"I contribute to EPF (mandatory), PPF ₹{ppf_annual:,}/year, and "
            f"NPS ₹{nps_monthly:,}/month ({nps_alloc} allocation). "
            f"When I retire at {retire_age}, what will my corpus be?"
        )

        a = (
            f"Retirement Corpus Projection (retire at {retire_age}, {years} years to go):\n\n"
            f"📌 EPF ({EPF_INTEREST_RATE*100:.2f}% p.a., EEE status):\n"
            f"  Employee contribution: ₹{epf_employee:,}/month (12% of basic)\n"
            f"  Employer EPF: ₹{epf_employer:,}/month (3.67% of basic)\n"
            f"  Projected corpus: ₹{int(epf_corpus):,}  (₹{int(epf_corpus/100_000)} lakh)\n"
            f"  Note: Interest on employee contrib >₹2.5L/yr is taxable from FY 2022-23\n\n"
            f"📌 PPF ({PPF_INTEREST_RATE*100:.1f}% p.a., fully EEE):\n"
            f"  Annual contribution: ₹{ppf_annual:,}\n"
            f"  Projected corpus: ₹{int(ppf_corpus):,}  (₹{int(ppf_corpus/100_000)} lakh)\n\n"
            f"📌 NPS (~{nps_rate*100:.1f}% p.a., {nps_alloc} allocation):\n"
            f"  Monthly contribution: ₹{nps_monthly:,}\n"
            f"  Projected corpus: ₹{int(nps_corpus):,}  (₹{int(nps_corpus/100_000)} lakh)\n"
            f"  At retirement: ₹{int(nps_lumpsum):,} lump sum (tax-free, 60%) + "
            f"₹{int(nps_annuity_corpus):,} annuity (taxable as income, 40%)\n"
            f"  Extra tax benefit: ₹50,000 under 80CCD(1B) → saves ~₹{nps_extra_tax_saving:,}/yr at 30% slab\n\n"
            f"📊 TOTAL PROJECTED CORPUS: ₹{int(total_corpus):,}  "
            f"(₹{total_corpus/10_000_000:.1f} crore)\n\n"
            f"Monthly pension (4% withdrawal rule): ~₹{int(monthly_pension):,}/month\n"
            f"In today's money (6% inflation): ~₹{int(today_value_pension):,}/month\n\n"
            f"{adequacy}\n\n"
            f"Note: Projections assume constant contributions and returns. "
            f"Actual NPS returns vary with market. Rebalance NPS allocation as you approach retirement."
        )

        lang = self.rng.choice(["en", "en", "hinglish"])
        diff = self.rng.choice(["intermediate", "advanced"])
        return self._make_sample(idx, q, a, lang=lang, difficulty=diff)
