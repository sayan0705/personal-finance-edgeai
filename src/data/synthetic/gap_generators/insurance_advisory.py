"""Gap 11 — Indian insurance advisory generator."""

from __future__ import annotations

from typing import Any

from .base import BaseGapGenerator

_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Pune",
    "Hyderabad", "Jaipur", "Lucknow", "Kochi", "Noida",
    "Gurgaon", "Ahmedabad",
]
_METRO_CITIES = {"Mumbai", "Delhi", "Kolkata", "Chennai"}
_INCOME_CHOICES = list(range(500_000, 5_000_001, 100_000))
_AGE_CHOICES = list(range(25, 56))
_PREM_CHOICES = [25_000, 50_000, 75_000, 100_000]
_QTYPES = ["term_life", "health", "ulip_exit", "comprehensive"]


def _approx_health_premium(age: int, dependents: int, metro: bool) -> int:
    base = 15_000 + age * 200 + (5_000 if dependents > 2 else 0)
    if metro:
        base = int(base * 1.15)
    return base


def _approx_term_premium(age: int, cover_cr: float, smoker: bool) -> int:
    base = int(cover_cr * 10_000_000 / 1000 * (0.8 if age < 35 else 1.5 if age < 45 else 3.0))
    if smoker:
        base = int(base * 1.5)
    return round(base / 100) * 100


class InsuranceAdvisoryGenerator(BaseGapGenerator):
    """Generates Indian insurance advisory Q&A pairs (Gap 11).

    Covers term life sizing, family health insurance selection, ULIP exit analysis,
    and comprehensive insurance checklists — all parametrised by age, income, city,
    and family structure.

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance.
    """

    gap_name = "insurance"
    task_type = "insurance_advisory"
    layer = "L3_personal_finance"
    source_dataset = "synthetic_insurance"

    def _generate_one(self, idx: int) -> dict[str, Any]:
        age = self.rng.choice(_AGE_CHOICES)
        income = self.rng.choice(_INCOME_CHOICES)
        dependents = self.rng.randint(0, 4)
        city = self.rng.choice(_CITIES)
        metro = city in _METRO_CITIES
        smoker = self.rng.random() > 0.85
        has_employer_health = self.rng.random() > 0.40
        employer_cover = self.rng.choice([200_000, 300_000, 500_000]) if has_employer_health else 0

        multiplier = self.rng.choice([10, 12, 15])
        rec_cover = income * multiplier
        term_premium = _approx_term_premium(age, rec_cover / 10_000_000, smoker)

        rec_health = 1_000_000 if metro else 750_000
        health_premium = _approx_health_premium(age, dependents, metro)

        qtype = self.rng.choice(_QTYPES)

        if qtype == "term_life":
            q, a = self._term_life_qa(
                age, income, dependents, smoker, multiplier, rec_cover, term_premium
            )
        elif qtype == "health":
            q, a = self._health_qa(
                city, metro, dependents, age, has_employer_health, employer_cover,
                rec_health, health_premium
            )
        elif qtype == "ulip_exit":
            prem = self.rng.choice(_PREM_CHOICES)
            years = self.rng.randint(3, 15)
            value = int(prem * years * self.rng.uniform(0.7, 1.3))
            q, a = self._ulip_exit_qa(age, prem, years, value, term_premium)
        else:
            q, a = self._comprehensive_qa(
                age, city, income, dependents, has_employer_health, employer_cover,
                rec_cover, term_premium, rec_health, health_premium, smoker
            )

        lang = self.rng.choice(["en", "en", "hinglish"])
        diff = self.rng.choice(["beginner", "intermediate"])
        return self._make_sample(idx, q, a, lang=lang, difficulty=diff)

    # ── scenario builders ─────────────────────────────────────────────────────

    def _term_life_qa(
        self, age: int, income: int, dependents: int, smoker: bool,
        multiplier: int, rec_cover: int, term_premium: int,
    ) -> tuple[str, str]:
        q = (
            f"I am {age} years old, earning ₹{income:,}/year with {dependents} dependent(s)"
            + (" (smoker)" if smoker else "")
            + ". How much term insurance do I need?"
        )
        dep_advice = (
            f"With {dependents} dependent(s), aim for the higher {multiplier}x multiple. "
            "Add outstanding loans and children's education costs to the base cover."
            if dependents > 1
            else f"With {dependents} dependent(s), 10x may suffice if spouse has own income."
        )
        a = (
            f"Term Insurance Sizing:\n\n"
            f"Rule of thumb: {multiplier}x annual income\n"
            f"Recommended cover: {multiplier} × ₹{income:,} = ₹{rec_cover:,} "
            f"(₹{rec_cover//100_000} lakh)\n\n"
            f"Estimated online premium: ~₹{term_premium:,}/year"
            + (" (smoker loading applies)" if smoker else "")
            + f"\n\n{dep_advice}\n\n"
            f"Buying tips:\n"
            f"• Buy ONLINE — 40–60% cheaper than through an agent\n"
            f"• Policy term: till age 60–65 (not 'whole life')\n"
            f"• Add riders: Critical Illness, Accidental Death Benefit\n"
            f"• Compare on PolicyBazaar / InsuranceDekho — get at least 3 quotes\n"
            f"• Never mix insurance + investment (avoid ULIPs and endowment plans)\n\n"
            f"Tax benefit: Premium up to ₹1,50,000 deductible under Section 80C (Old Regime only)."
        )
        return q, a

    def _health_qa(
        self, city: str, metro: bool, dependents: int, age: int,
        has_employer: bool, employer_cover: int, rec_health: int, health_premium: int,
    ) -> tuple[str, str]:
        q = (
            f"I need health insurance for my family ({dependents+1} members) in {city}. "
            + (f"Employer provides ₹{employer_cover//100_000}L cover. " if has_employer else "No employer coverage. ")
            + "What should I buy?"
        )
        employer_note = (
            f"Employer cover (₹{employer_cover//100_000}L) is NOT sufficient for {city}. "
            "Hospital bills for major procedures easily cross ₹5–10L."
            if has_employer
            else "No employer cover — buying personal health insurance is urgent."
        )
        a = (
            f"Health Insurance Recommendation for {city} ({'metro' if metro else 'non-metro'}):\n\n"
            f"{employer_note}\n\n"
            f"Recommended: ₹{rec_health//100_000}L base + ₹25L super top-up\n"
            f"Why super top-up: ₹25L extra cover for just ₹3,000–5,000/year extra "
            f"(kicks in after base cover is exhausted).\n\n"
            f"Estimated premium: ~₹{health_premium:,}/year "
            f"for family floater ({dependents+1} members, primary aged {age})\n\n"
            f"Non-negotiable features:\n"
            f"✅ No room rent capping\n"
            f"✅ Zero co-payment\n"
            f"✅ Pre-existing disease waiting period ≤ 3 years\n"
            f"✅ Restoration benefit (refills cover mid-year)\n"
            f"✅ Cashless network hospitals in your city\n\n"
            f"Tax benefit (Sec 80D): ₹25,000 self+family + ₹25,000 parents "
            f"= ₹50,000 total (₹50,000 for senior parents).\n\n"
            f"Top insurers to compare: Star Health, HDFC Ergo, Care Health, Niva Bupa."
        )
        return q, a

    def _ulip_exit_qa(
        self, age: int, prem: int, years: int, value: int, term_premium: int,
    ) -> tuple[str, str]:
        invested = prem * years
        cagr = ((value / invested) ** (1 / years) - 1) * 100 if invested > 0 and years > 0 else 0.0
        q = (
            f"I have a ULIP with ₹{prem:,}/year premium for {years} years. "
            f"Fund value: ₹{value:,} (invested ₹{invested:,}). Should I continue or exit?"
        )
        if cagr < 8:
            verdict = f"❌ CAGR {cagr:.1f}% — even PPF (7.1%) beats this. High ULIP charges are eating your returns."
        else:
            verdict = f"⚠️ CAGR {cagr:.1f}% is moderate — a direct equity MF would likely outperform after ULIP charges."

        surrender_note = (
            "Since you have completed 5+ years, surrender is tax-free under Sec 10(10D) "
            "(if annual premium ≤ ₹2.5L). Surrender now."
            if years >= 5
            else f"Lock-in is 5 years. You have {5-years} year(s) remaining. Plan to exit at the 5-year mark."
        )
        alt_term = max(8_000, int(term_premium * 0.15))
        alt_sip = prem - alt_term
        a = (
            f"ULIP Analysis:\n\n"
            f"Invested: ₹{invested:,} over {years} years\n"
            f"Current Value: ₹{value:,}\n"
            f"CAGR: {cagr:.1f}%\n\n"
            f"{verdict}\n\n"
            f"{surrender_note}\n\n"
            f"Better alternative (same ₹{prem:,}/year):\n"
            f"• Term insurance (₹1 Cr): ~₹{alt_term:,}/year\n"
            f"• Equity SIP: ₹{alt_sip:,}/year → at 12% CAGR, far outperforms ULIP after charges\n\n"
            f"Disclaimer: Calculate exact surrender value with your insurer before deciding. "
            f"Tax treatment may vary — consult a CA."
        )
        return q, a

    def _comprehensive_qa(
        self, age: int, city: str, income: int, dependents: int,
        has_employer: bool, employer_cover: int, rec_cover: int, term_premium: int,
        rec_health: int, health_premium: int, smoker: bool,
    ) -> tuple[str, str]:
        q = (
            f"I am {age} in {city}, earning ₹{income:,}/year with {dependents} dependent(s). "
            "Give me a complete insurance checklist."
        )
        dep_urgency = "CRITICAL — buy immediately" if dependents > 0 else "Can defer if no financial dependents"
        employer_note = (
            f"Employer cover ₹{employer_cover//100_000}L supplements but does NOT replace personal cover."
            if has_employer
            else "No employer coverage — highest priority."
        )
        total_annual = term_premium + health_premium + 7_000
        a = (
            f"Insurance Checklist for age {age}, ₹{income:,} income, {dependents} dependent(s):\n\n"
            f"1️⃣  TERM LIFE: ₹{rec_cover//100_000}L cover (~₹{term_premium:,}/yr)"
            + (" — smoker loading applies" if smoker else "")
            + f"\n    Status: {dep_urgency}\n\n"
            f"2️⃣  HEALTH: ₹{rec_health//100_000}L base + ₹25L super top-up "
            f"(~₹{health_premium:,}/yr)\n"
            f"    {employer_note}\n\n"
            f"3️⃣  PERSONAL ACCIDENT: ₹50L cover (~₹2,000/yr)\n"
            f"    Covers disability — term life only pays on death\n\n"
            f"4️⃣  CRITICAL ILLNESS: ₹25–50L lump sum (~₹5,000–10,000/yr)\n"
            f"    Cancer, heart attack, stroke — lump sum on diagnosis\n\n"
            f"❌ DO NOT BUY: ULIPs, endowment plans, money-back policies, 'child plans' "
            f"that bundle insurance + investment.\n\n"
            f"Total annual insurance cost: ~₹{total_annual:,} ({total_annual/income*100:.1f}% of income)\n"
            f"Tax deductions available: 80C (term premium) + 80D (health premium)"
        )
        return q, a
