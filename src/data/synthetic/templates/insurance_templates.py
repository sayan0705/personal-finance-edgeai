"""Insurance Q&A template generator for term, health, and ULIP scenarios."""

from __future__ import annotations

import random

from ..constants import (
    DEDUCTION_80D_SELF_LIMIT,
    DEDUCTION_80D_PARENTS_SENIOR_LIMIT,
    TERM_COVER_MULTIPLE,
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


class InsuranceTemplates:
    """Generates insurance-related Q&A pairs.

    Args:
        rng: Random instance for reproducibility.
    """

    _ANNUAL_INCOMES = [5_00_000, 8_00_000, 10_00_000, 12_00_000, 15_00_000, 20_00_000, 25_00_000]
    _AGES = [25, 28, 30, 32, 35, 38, 40]

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._generators = [
            self._term_insurance_coverage,
            self._health_insurance_query,
            self._ulip_vs_term_mf,
            self._parents_health_insurance,
            self._critical_illness_query,
        ]

    def generate_sample(self) -> dict[str, str]:
        """Return a dict with ``question`` and ``answer`` keys."""
        gen = self._rng.choice(self._generators)
        return gen()

    # ── scenario generators ───────────────────────────────────────────────────

    def _term_insurance_coverage(self) -> dict[str, str]:
        age = self._rng.choice(self._AGES)
        income = self._rng.choice(self._ANNUAL_INCOMES)
        cover = income * TERM_COVER_MULTIPLE
        dependants = self._rng.choice(["spouse and 2 children", "spouse, 1 child, and parents", "parents"])

        q = (
            f"I am {age} years old with an annual income of {_fmt_inr(income)}. "
            f"I have {dependants} as financial dependants. "
            "How much term insurance cover should I buy?"
        )
        liabilities = self._rng.choice([0, 20_00_000, 30_00_000, 50_00_000])
        recommended = cover + liabilities

        a = (
            f"**Term Insurance Coverage Calculator**\n\n"
            f"**Step 1 — Human Life Value (HLV) Method:**\n"
            f"Rule of thumb: 15–20× annual income for the primary earning years\n"
            f"- Your income: {_fmt_inr(income)}\n"
            f"- Base cover (15×): {_fmt_inr(cover)}\n\n"
            f"**Step 2 — Add liabilities:**\n"
            f"- Outstanding loans/liabilities: {_fmt_inr(liabilities)}\n\n"
            f"**Recommended cover: {_fmt_inr(recommended)}**\n\n"
            f"**At age {age}, approximate annual premium:**\n"
            f"- Non-smoker, healthy: ₹8,000 – ₹15,000/year for ₹1 Cr cover (30-year term)\n"
            f"- Buy now: premiums increase significantly every 5 years\n\n"
            f"**What to look for in a term plan:**\n"
            f"1. **Claim Settlement Ratio (CSR):** Choose insurers with CSR > 98%\n"
            f"   (LIC: 98.7%, HDFC Life: 99.5%, Max Life: 99.5% — IRDAI data FY 2023-24)\n"
            f"2. **Policy term:** Cover till at least age 65 or till youngest child is financially independent\n"
            f"3. **Riders:** Consider waiver of premium, accidental death benefit\n"
            f"4. **Avoid:** ULIP or money-back plans for pure life cover — far more expensive\n\n"
            f"Term insurance premiums are eligible for Section 80C deduction (old regime)."
        )
        return {"question": q, "answer": a}

    def _health_insurance_query(self) -> dict[str, str]:
        age = self._rng.choice([25, 28, 30, 32, 35])
        family_size = self._rng.choice(["self", "self and spouse", "self, spouse, and 1 child", "family of 4"])
        city_type = self._rng.choice(["metro", "tier-2"])
        cover = 1_000_000 if city_type == "metro" else 500_000

        q = (
            f"I am {age} years old living in a {city_type} city. "
            f"I need health insurance for {family_size}. "
            "How much cover is adequate and what should I look for?"
        )
        a = (
            f"**Health Insurance Guide for {family_size.title()} in {city_type.title()} city**\n\n"
            f"**Recommended cover: {_fmt_inr(cover)} – {_fmt_inr(cover * 2)}**\n\n"
            f"Why: Hospital costs in {city_type} cities for major procedures:\n"
            f"- Angioplasty: ₹2–5 lakhs\n"
            f"- Knee replacement: ₹2–4 lakhs\n"
            f"- Cancer treatment: ₹5–20 lakhs+\n\n"
            f"**Key features to compare:**\n"
            f"1. **Cashless network hospitals:** Check if your preferred hospitals are in-network\n"
            f"2. **Room rent sub-limit:** Prefer policies with no room rent cap or at least 1% of SI\n"
            f"3. **Disease-specific sub-limits:** Avoid policies capping specific diseases\n"
            f"4. **Waiting periods:** Pre-existing disease waiting period (typically 2–4 years)\n"
            f"5. **No-Claim Bonus (NCB):** Cumulative benefit up to 100% of SI for claim-free years\n"
            f"6. **Restoration benefit:** SI gets restored after full utilisation in a year\n\n"
            f"**Recommended insurers (2024):** Star Health, Niva Bupa, Care Health, HDFC Ergo\n\n"
            f"**Tax benefit:** Premium paid is deductible under Section 80D:\n"
            f"- Self + family (age < 60): up to ₹{DEDUCTION_80D_SELF_LIMIT:,}\n"
            f"- Add parents (senior citizen): up to ₹{DEDUCTION_80D_PARENTS_SENIOR_LIMIT:,} more"
        )
        return {"question": q, "answer": a}

    def _ulip_vs_term_mf(self) -> dict[str, str]:
        q = "My agent is recommending a ULIP plan with ₹1 lakh annual premium for 10 years. Should I buy it or should I go for term insurance + mutual fund?"
        a = (
            "**ULIP vs Term + Mutual Fund — A Critical Comparison**\n\n"
            "This is one of the most common mis-selling scenarios in Indian insurance. "
            "Let's analyse both options with ₹1,00,000 annual investment:\n\n"
            "**Option A — ULIP (₹1L/year for 10 years)**\n"
            "- Year 1 charges: 15–20% of premium (₹15,000–₹20,000)\n"
            "- Ongoing fund management charges: 1.35% p.a.\n"
            "- Mortality charges deducted monthly\n"
            "- After all charges, net annual return: ~8–10% (fund performance minus fees)\n"
            "- Lock-in: 5 years\n"
            "- Estimated 10-year corpus: ₹14–16 lakhs (varies by fund)\n\n"
            "**Option B — Term Plan (₹10,000/year) + MF SIP (₹90,000/year)**\n"
            "- Term insurance: ₹1 Cr cover for ₹8,000–₹12,000/year\n"
            "- Equity MF SIP: ₹90,000/year at assumed 12% CAGR for 10 years\n"
            "- Estimated 10-year corpus: ₹17–19 lakhs (lower charges, higher net return)\n"
            "- Full flexibility: switch funds, withdraw, increase/reduce SIP anytime\n\n"
            "**Verdict: Term + MF is almost always better because:**\n"
            "1. Pure protection (term) at fraction of ULIP cost\n"
            "2. Higher net returns due to significantly lower charges\n"
            "3. Complete transparency and flexibility\n"
            "4. You can switch between funds without insurance lock-in\n\n"
            "**When ULIP might make sense:**\n"
            "- HNI investors who have already exhausted 80C via other instruments\n"
            "- Estate planning in specific structures (consult a wealth advisor)\n\n"
            "Source: IRDAI guidelines on ULIP charges; SEBI mutual fund regulations."
        )
        return {"question": q, "answer": a}

    def _parents_health_insurance(self) -> dict[str, str]:
        parent_age = self._rng.choice([58, 60, 62, 65, 68, 70])
        q = (
            f"My parents are {parent_age} years old. Can I buy health insurance for them? "
            "What are the tax benefits and what should I look for?"
        )
        is_senior = parent_age >= 60
        deduction = DEDUCTION_80D_PARENTS_SENIOR_LIMIT if is_senior else 25_000

        a = (
            f"**Health Insurance for Parents (Age {parent_age})**\n\n"
            f"**Yes, you can and should buy health insurance for your parents.** "
            f"{'They are senior citizens (≥60 years).' if is_senior else ''}\n\n"
            f"**Tax Benefit under Section 80D:**\n"
            f"- Premium paid for parents: deduction up to ₹{deduction:,}/year\n"
            f"- This is IN ADDITION to your own health insurance deduction of ₹25,000\n"
            f"- Maximum total deduction if both you and parents are senior citizens: ₹1,00,000/year\n\n"
            f"**Challenges at age {parent_age}:**\n"
            f"- Many insurers exclude pre-existing diseases for first 2–4 years\n"
            f"- Premiums are significantly higher (expect ₹25,000–₹60,000/year for ₹5L cover)\n"
            f"- Some conditions may be permanently excluded\n\n"
            f"**Best options for senior parents:**\n"
            f"1. **Star Senior Citizens Red Carpet**: Accepts up to age 75, covers pre-existing "
            f"after 1 year (with co-payment)\n"
            f"2. **Niva Bupa Senior First**: Good network, no room rent sub-limit\n"
            f"3. **Care Senior**: Comprehensive cover for 60+ age group\n\n"
            f"**Key tips:**\n"
            f"- Buy now — premiums increase steeply with age\n"
            f"- Check if employer insurance covers parents (add-on option)\n"
            f"- Government's Ayushman Bharat PM-JAY: free cover up to ₹5L for eligible families\n"
            f"- Always declare pre-existing conditions honestly to avoid claim rejection"
        )
        return {"question": q, "answer": a}

    def _critical_illness_query(self) -> dict[str, str]:
        q = "What is critical illness insurance and do I need it if I already have a health insurance policy?"
        a = (
            "**Critical Illness Insurance — What It Is and Why You Need It**\n\n"
            "**Regular health insurance** pays your *hospitalisation bills* — doctor fees, "
            "medicines, surgery costs.\n\n"
            "**Critical illness insurance** pays a *lump sum* (e.g., ₹25L) directly to you "
            "upon diagnosis of specified illnesses — regardless of actual medical costs.\n\n"
            "**Why you need both:**\n\n"
            "Scenario: You are diagnosed with cancer.\n"
            "- Treatment cost: ₹15 lakhs — covered by regular health insurance ✓\n"
            "- Income loss during 6-month treatment: ₹3–5 lakhs — NOT covered ✗\n"
            "- Home loan EMI during treatment: ₹50,000/month — NOT covered ✗\n"
            "- Lifestyle modifications, special diet, caregiver cost — NOT covered ✗\n\n"
            "**Critical illness cover pays** the lump sum for exactly these non-medical costs.\n\n"
            "**Typically covered illnesses:**\n"
            "Heart attack, stroke, cancer (specified stages), kidney failure, major organ "
            "transplant, paralysis, aorta surgery, coronary artery bypass graft (CABG)\n\n"
            "**Recommended cover:** 2–3 years of annual income (minimum ₹25L)\n\n"
            "**Cost:** ₹4,000–₹8,000/year for ₹25L cover (healthy, age 30–35)\n\n"
            "**Tax benefit:** Premium deductible under Section 80D (part of the ₹25,000 limit)\n\n"
            "**Key exclusion:** Most policies exclude claims within 90-day waiting period "
            "and pre-existing conditions. Buy early when you're healthy."
        )
        return {"question": q, "answer": a}
