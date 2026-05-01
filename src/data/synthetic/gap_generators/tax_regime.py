"""Gaps 2, 3, 4 — Tax regime comparison, HRA exemption, and Section 80C optimisation."""

from __future__ import annotations

from typing import Any

from ..constants import (
    DEDUCTION_80C_LIMIT,
    DEDUCTION_80CCD1B_LIMIT,
    EPF_INTEREST_RATE,
    PPF_INTEREST_RATE,
)
from ..templates.tax_templates import compute_new_regime_tax, compute_old_regime_tax
from .base import BaseGapGenerator

_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Pune",
    "Hyderabad", "Jaipur", "Lucknow", "Kochi", "Noida",
    "Gurgaon", "Ahmedabad", "Kolkata", "Chandigarh",
]
_METRO_CITIES = {"Mumbai", "Delhi", "Kolkata", "Chennai"}

_BASIC_CHOICES = list(range(15_000, 150_001, 5_000))
_DEDUCTIONS_80C = [0, 50_000, 100_000, 150_000]
_DEDUCTIONS_80D = [0, 15_000, 25_000, 50_000, 75_000]
_HOME_LOAN_INT = [0, 0, 0, 100_000, 150_000, 200_000]
_NPS_80CCD = [0, 0, 50_000]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _hra_exemption(basic_m: int, hra_m: int, rent_m: int, metro: bool) -> tuple[int, int, int, int]:
    """Return (a, b, c, exempt) for the HRA three-way minimum rule."""
    a = hra_m
    b = int(basic_m * (0.50 if metro else 0.40))
    c = max(0, rent_m - int(basic_m * 0.10))
    return a, b, c, min(a, b, c) if rent_m > 0 else 0


class TaxRegimeComparisonGenerator(BaseGapGenerator):
    """Generates old-vs-new tax regime comparison Q&A pairs (Gap 2).

    Each sample builds a realistic salary profile (basic, HRA, special allowances),
    chooses random deductions, calculates tax under both regimes using the canonical
    FY 2024-25 calculators from ``tax_templates``, and produces a step-by-step answer.

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance.
    """

    gap_name = "tax_regime"
    task_type = "tax_comparison"
    layer = "L2_indian_regulatory"
    source_dataset = "synthetic_tax_regime"

    def _generate_one(self, idx: int) -> dict[str, Any]:
        basic = self.rng.choice(_BASIC_CHOICES)
        hra_pct = self.rng.choice([0.40, 0.50])
        hra = int(basic * hra_pct)
        da = int(basic * self.rng.choice([0.0, 0.05, 0.10]))
        special = self.rng.randint(5_000, 60_000)
        gross_monthly = basic + hra + da + special
        gross_annual = gross_monthly * 12

        city = self.rng.choice(_CITIES)
        metro = city in _METRO_CITIES
        rent_m = self.rng.randint(3_000, 60_000)

        sec80c = self.rng.choice(_DEDUCTIONS_80C)
        sec80d = self.rng.choice(_DEDUCTIONS_80D)
        hloan_int = self.rng.choice(_HOME_LOAN_INT)
        nps_80ccd = self.rng.choice(_NPS_80CCD)

        _, _, _, hra_exempt_m = _hra_exemption(basic, hra, rent_m, metro)
        hra_exempt_a = hra_exempt_m * 12
        home_loan_capped = min(hloan_int, 200_000)

        old_result = compute_old_regime_tax(
            gross_income=gross_annual,
            deduction_80c=sec80c,
            deduction_80d_self=min(sec80d, 25_000),
            deduction_80d_parents=max(0, sec80d - 25_000),
            other_deductions=hra_exempt_a + home_loan_capped + nps_80ccd,
        )
        new_result = compute_new_regime_tax(gross_annual)

        better = "Old" if old_result["total_tax"] < new_result["total_tax"] else "New"
        saving = abs(old_result["total_tax"] - new_result["total_tax"])
        total_old_deductions = (
            50_000 + hra_exempt_a + sec80c + sec80d + home_loan_capped + nps_80ccd
        )

        q = (
            f"Annual salary: ₹{gross_annual:,} (Basic ₹{basic*12:,}, HRA ₹{hra*12:,}). "
            f"City: {city}. Rent: ₹{rent_m:,}/month. "
            f"80C: ₹{sec80c:,}, 80D: ₹{sec80d:,}"
            + (f", Home loan interest: ₹{hloan_int:,}" if hloan_int else "")
            + (f", NPS 80CCD(1B): ₹{nps_80ccd:,}" if nps_80ccd else "")
            + ". Which tax regime should I choose for FY 2025-26?"
        )

        mp_pct = 50 if metro else 40
        a_v, b_v, c_v, _ = _hra_exemption(basic, hra, rent_m, metro)
        a = (
            f"OLD REGIME:\n"
            f"Gross: ₹{gross_annual:,}\n"
            f"(-) Std Deduction: ₹50,000\n"
            f"(-) HRA Exempt: ₹{hra_exempt_a:,}  "
            f"[min(₹{a_v*12:,}, ₹{b_v*12:,}, ₹{c_v*12:,})] "
            f"({mp_pct}% of basic applies for {city})\n"
            f"(-) 80C: ₹{sec80c:,}  |  80D: ₹{sec80d:,}"
            + (f"\n(-) Sec 24(b) home loan: ₹{home_loan_capped:,}" if hloan_int else "")
            + (f"\n(-) 80CCD(1B) NPS: ₹{nps_80ccd:,}" if nps_80ccd else "")
            + f"\nTaxable: ₹{old_result['taxable_income']:,}  →  "
            f"Tax + Cess: ₹{old_result['total_tax']:,}  "
            f"(eff. {old_result['effective_rate']}%)\n\n"
            f"NEW REGIME:\n"
            f"Gross: ₹{gross_annual:,}\n"
            f"(-) Std Deduction: ₹75,000 (Budget 2024)\n"
            f"Taxable: ₹{new_result['taxable_income']:,}  →  "
            f"Tax + Cess: ₹{new_result['total_tax']:,}  "
            f"(eff. {new_result['effective_rate']}%)\n\n"
            f"✅ {better} Regime saves ₹{saving:,}/year.\n\n"
        )
        if better == "Old":
            a += (
                f"Old regime wins because your deductions total ₹{total_old_deductions:,} — "
                f"well above the ₹75,000 new regime std deduction."
            )
        else:
            a += (
                f"New regime wins — your deductions total only ₹{total_old_deductions:,}, "
                f"not enough to offset the wider new regime slab structure."
            )
        a += "\n\nNote: Verify with a CA. Surcharge applies above ₹50 lakh income."

        return self._make_sample(idx, q, a, difficulty="intermediate")


class HRAExemptionGenerator(BaseGapGenerator):
    """Generates HRA exemption calculation Q&A pairs (Gap 3).

    Produces specific numeric examples using the three-way minimum rule for
    metro and non-metro cities, covering a wide range of basic salaries and rent levels.

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance.
    """

    gap_name = "hra_calc"
    task_type = "tax_calculation"
    layer = "L2_indian_regulatory"
    source_dataset = "synthetic_hra"

    def _generate_one(self, idx: int) -> dict[str, Any]:
        basic_m = self.rng.choice(_BASIC_CHOICES)
        hra_m = int(basic_m * self.rng.choice([0.40, 0.50]))
        rent_m = self.rng.randint(3_000, 60_000)
        city = self.rng.choice(_CITIES)
        metro = city in _METRO_CITIES
        mp = 50 if metro else 40

        a_v, b_v, c_v, exempt = _hra_exemption(basic_m, hra_m, rent_m, metro)
        taxable_hra = hra_m - exempt

        q = (
            f"Calculate HRA exemption: Basic ₹{basic_m:,}/month, "
            f"HRA ₹{hra_m:,}/month, Rent paid ₹{rent_m:,}/month, City: {city}."
        )

        binding = "rent - 10% of basic (c)" if (c_v == exempt) else (
            "actual HRA (a)" if (a_v == exempt) else f"{mp}% of basic (b)"
        )
        tip = (
            "Tip: Increase rent to claim more exemption — even small increments help "
            "if the limiting factor is component (c)."
            if c_v == exempt
            else "Tip: Negotiate a higher HRA component in your next CTC revision to maximise exemption."
        )

        a = (
            f"{city} = {'metro' if metro else 'non-metro'} → {mp}% of Basic applies.\n\n"
            f"HRA exemption = min of three amounts:\n"
            f"  (a) Actual HRA received:       ₹{a_v:,}/month\n"
            f"  (b) {mp}% of Basic salary:     ₹{b_v:,}/month\n"
            f"  (c) Rent paid - 10% of Basic:  ₹{rent_m:,} - ₹{int(basic_m*0.10):,} = ₹{c_v:,}/month\n\n"
            f"Exempt = min(a, b, c) = ₹{exempt:,}/month  (limited by {binding})\n"
            f"Annual exempt HRA = ₹{exempt*12:,}\n"
            f"Taxable HRA = ₹{hra_m:,} - ₹{exempt:,} = ₹{taxable_hra:,}/month\n\n"
            f"{tip}\n"
            f"⚠️ HRA exemption is only available under the Old Tax Regime."
        )

        return self._make_sample(idx, q, a, difficulty="intermediate")


class Section80CGenerator(BaseGapGenerator):
    """Generates Section 80C optimisation Q&A pairs (Gap 4).

    For a given age and existing 80C usage, recommends the best instruments
    to fill the remaining ₹1.5 lakh limit and suggests beyond-80C options
    when the limit is already exhausted.

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance.
    """

    gap_name = "80c_opt"
    task_type = "tax_optimization"
    layer = "L2_indian_regulatory"
    source_dataset = "synthetic_80c"

    _COMPONENTS = {
        "EPF": [0, 21_600, 36_000, 50_000, 72_000, 86_400],
        "Life Insurance": [0, 10_000, 15_000, 25_000],
        "PPF": [0, 0, 50_000, 100_000, 150_000],
        "ELSS": [0, 0, 25_000, 50_000, 100_000],
        "Home Loan Principal": [0, 0, 0, 50_000, 100_000],
        "Children Tuition": [0, 0, 25_000, 50_000],
    }

    def _generate_one(self, idx: int) -> dict[str, Any]:
        existing = {k: self.rng.choice(v) for k, v in self._COMPONENTS.items()}
        total_80c = sum(existing.values())
        remaining = max(0, DEDUCTION_80C_LIMIT - total_80c)
        age = self.rng.randint(24, 58)

        items_str = (
            ", ".join(f"{k}: ₹{v:,}" for k, v in existing.items() if v > 0) or "None"
        )
        q = (
            f"I am {age} years old. My current 80C investments: {items_str}. "
            f"Total: ₹{total_80c:,}. How should I optimize to reach the ₹1.5 lakh limit?"
        )

        epf_rate_pct = EPF_INTEREST_RATE * 100
        ppf_rate_pct = PPF_INTEREST_RATE * 100

        if remaining <= 0:
            a = (
                f"Your 80C is fully utilised at ₹{min(total_80c, DEDUCTION_80C_LIMIT):,} "
                f"(limit: ₹{DEDUCTION_80C_LIMIT:,}). Great!\n\n"
                f"Additional deductions you can still claim:\n"
                f"• 80CCD(1B): ₹{DEDUCTION_80CCD1B_LIMIT:,} extra for NPS Tier-1 — "
                f"saves ₹{int(DEDUCTION_80CCD1B_LIMIT*0.30):,} at 30% slab\n"
                f"• 80D: Health insurance (₹25,000 self + family; ₹50,000 for senior parents)\n"
                f"• 80E: Education loan interest (no cap)\n"
                f"• Sec 24(b): Home loan interest up to ₹2,00,000 (self-occupied)"
            )
        else:
            recs: list[str] = []
            r = remaining
            if age < 45 and r > 0:
                x = min(r, DEDUCTION_80C_LIMIT)
                recs.append(
                    f"ELSS Mutual Fund: ₹{x:,}  "
                    f"(3-year lock-in, ~12-15% historical CAGR — best tax+return combo)"
                )
                r -= x
            if r > 0:
                x = min(r, DEDUCTION_80C_LIMIT)
                recs.append(
                    f"PPF: ₹{x:,}  "
                    f"({ppf_rate_pct:.1f}% guaranteed, EEE status, 15-year tenure)"
                )

            priority = "ELSS > PPF > VPF" if age < 40 else "PPF > SCSS > ELSS"
            a = (
                f"Current: ₹{total_80c:,} / ₹{DEDUCTION_80C_LIMIT:,}. "
                f"Gap: ₹{remaining:,}\n\n"
                f"Recommended investments to fill gap:\n"
                + "\n".join(f"  {j+1}. {rec}" for j, rec in enumerate(recs))
                + f"\n\nAge {age} priority order: {priority}\n\n"
                f"Beyond 80C:\n"
                f"• 80CCD(1B) NPS: ₹50,000 extra deduction — high priority\n"
                f"• EPF rate: {epf_rate_pct:.2f}% p.a. (EEE, auto-invested if salaried)\n"
                f"• 80D health insurance: ₹25,000–₹1,00,000 depending on family"
            )

        return self._make_sample(
            idx, q, a, difficulty=self.rng.choice(["beginner", "intermediate"])
        )
