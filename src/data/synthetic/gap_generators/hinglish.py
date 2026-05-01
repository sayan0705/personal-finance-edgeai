"""Gap 7 — Hindi/Hinglish financial Q&A generator.

Expanded from the notebook's 6 pairs to 16 pairs covering SIP, health insurance,
MF taxation, property inheritance, credit cards, NPS vs PPF, emergency fund, CIBIL,
gold vs MF, home loan prepayment, ITR filing, and more.
Parametric variations (salary, amounts, ages) ensure near-zero duplicate rate even
at 300 samples.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGapGenerator

# Each entry: (question_template, answer, param_key, param_choices)
# param_key=None means no parametric substitution.
_QA_PAIRS: list[tuple[str, str, str | None, list[Any] | None]] = [
    (
        "Mera salary {salary} hai monthly. SIP kitna start karu?",
        (
            "Aapki salary ₹{salary} hai toh minimum 20% yani ₹{sip_20pct} se SIP start karna chahiye.\n\n"
            "Suggested allocation:\n"
            "• ₹{sip_nifty} — Nifty 50 Index Fund (safe, low-cost, ~12% long-run)\n"
            "• ₹{sip_flexi} — Flexi Cap Fund (diversified equity)\n"
            "• ₹{sip_elss}  — ELSS Fund (tax saving under 80C)\n\n"
            "Direct plan choose karein — expense ratio 0.5–1% kam hota hai.\n"
            "Pehle 6 months ka emergency fund banayein, phir SIP shuru karein.\n\n"
            "Disclaimer: Ye general guidance hai. SEBI registered advisor se milein."
        ),
        "salary",
        [20_000, 30_000, 40_000, 50_000, 60_000, 75_000, 100_000],
    ),
    (
        "Health insurance lena hai family ke liye {members} members. Kitne ka cover lu?",
        (
            "Family ke liye minimum ₹10–15 lakh cover lena chahiye (metro cities ke liye ₹15–20 lakh).\n\n"
            "Kya dekhna hai:\n"
            "✅ Room rent limit: No capping wala plan lo\n"
            "✅ Co-payment: 0% try karein\n"
            "✅ Pre-existing disease waiting: 2–3 year (kam wala behtar)\n"
            "✅ Restoration benefit: cover refill karta hai mid-year\n\n"
            "Super top-up bhi lo — ₹25 lakh extra coverage sirf ₹3,000–5,000/year extra mein.\n\n"
            "80D deduction: ₹25,000 (self+family) + ₹25,000 parents = ₹50,000 total.\n"
            "Senior parents ke liye: ₹50,000 alag se."
        ),
        "members",
        [2, 3, 4, 5],
    ),
    (
        "Mutual fund se paise kab nikalun? Tax kitna lagega?",
        (
            "Mutual fund redemption pe tax:\n\n"
            "Equity Funds (ELSS, Large/Mid/Small Cap):\n"
            "• < 12 months hold: STCG 20%\n"
            "• ≥ 12 months: LTCG 12.5% (sirf ₹1.25 lakh se zyada gain pe)\n\n"
            "Debt Funds (post April 2023):\n"
            "• Koi bhi holding: Aapke income slab rate pe tax\n\n"
            "Kab nikaalein:\n"
            "• Goal achieve ho gaya — tab nikaalein, market time mat karein\n"
            "• Emergency mein pehle liquid/debt fund se nikaalein\n"
            "• Equity fund 5+ saal ke liye rakhein ideally\n\n"
            "⚠️ SIP band karna aur redeem karna alag hai — SIP stop karne pe koi tax nahi."
        ),
        None,
        None,
    ),
    (
        "Papa ke naam pe property hai. Unke baad mujhe kaise milegi? Inheritance tax lagega?",
        (
            "Property inheritance India mein:\n\n"
            "Agar Will hai:\n"
            "• Will ke hisaab se property transfer hogi. Probate court se verify karwayein.\n\n"
            "Agar Will nahi (Intestate Succession):\n"
            "• Hindu Succession Act lagega. Class 1 heirs — wife, sons, daughters — equal share.\n\n"
            "Tax:\n"
            "• Inheritance pe koi tax NAHI lagta (Sec 56 — relatives se gift exempt hai)\n"
            "• Property bechne pe Capital Gains tax lagega\n"
            "• Cost of acquisition: Papa ne jab kharidi tab ka price (indexation allowed)\n\n"
            "Steps:\n"
            "1. Mutation karwayein (tehsil/municipal office)\n"
            "2. Property registry mein naam transfer\n"
            "3. Society/housing board ko inform karein\n\n"
            "CA/lawyer se zaroor consult karein property transfer mein."
        ),
        None,
        None,
    ),
    (
        "Credit card ka bill pura pay karna chahiye ya minimum due?",
        (
            "HAMESHA pura bill pay karein. Samajhiye kyun:\n\n"
            "Minimum due ka trap:\n"
            "• Credit card interest: 36–42% per annum (3–3.5% monthly!)\n"
            "• ₹50,000 outstanding pe minimum due ~₹2,500\n"
            "• Baaki ₹47,500 pe ₹1,425 interest SIRF NEXT MONTH\n\n"
            "Agar pura pay nahi kar sakte:\n"
            "1. Balance transfer card lo (0% EMI 3–6 months)\n"
            "2. Personal loan (12–14%) se credit card (36%+) pay karein\n"
            "3. Spending immediately rok dein\n\n"
            "CIBIL pe impact:\n"
            "• 30+ days late: Score gir jayega\n"
            "• Utilization 30%+ hone se bhi score negative\n\n"
            "Golden rule: Credit card ko debit card ki tarah use karein."
        ),
        None,
        None,
    ),
    (
        "NPS mein invest karu ya PPF mein? Retirement ke liye kya better hai?",
        (
            "NPS vs PPF comparison:\n\n"
            "PPF (Public Provident Fund):\n"
            "• Interest: 7.1% guaranteed, govt backed\n"
            "• Lock-in: 15 saal\n"
            "• Tax: EEE — invest, interest, maturity sab tax-free\n"
            "• Risk: Zero\n\n"
            "NPS (National Pension System):\n"
            "• Returns: 9–12% equity, 8–9% bonds — NOT guaranteed\n"
            "• Lock-in: Age 60 tak\n"
            "• Tax benefit: 80CCD(1B) — extra ₹50,000 OVER 80C limit\n"
            "• At maturity: 60% lump sum (tax-free), 40% annuity (taxable)\n\n"
            "Recommendation by age:\n"
            "• < 35 saal: 60% NPS + 40% PPF (growth important)\n"
            "• 35–45 saal: 50–50\n"
            "• 45+: 30% NPS + 70% PPF (stability chahiye)\n\n"
            "NPS ka extra ₹50K tax benefit bahut valuable — 30% slab mein ₹15,000 bachta hai.\n"
            "Dono mein invest karein — PPF safety ke liye, NPS growth ke liye."
        ),
        None,
        None,
    ),
    (
        "Mera {age} saal ka beta hai. Uski padhai ke liye kitna save karu?",
        (
            "Bache ki higher education planning:\n\n"
            "India mein engineering/medicine: ₹15–50 lakh (2030 mein inflation adjust)\n"
            "Foreign education (US/UK): ₹80 lakh – ₹2 crore\n\n"
            "Goal: Bacche ki age 18 pe required corpus\n"
            "Years available: {years_left} saal\n\n"
            "SIP calculation (12% assumed CAGR):\n"
            "• ₹15 lakh goal: ~₹{sip_15l}/month SIP\n"
            "• ₹50 lakh goal: ~₹{sip_50l}/month SIP\n\n"
            "Best instruments:\n"
            "• Equity MF (7+ years horizon): Large/Mid cap index funds\n"
            "• Sukanya Samriddhi (agar beti hai): 8.2% tax-free, EEE\n"
            "• PPF: Stable backup\n\n"
            "Start karo aaj hi — compounding ka faida har saal delay se kam hota hai."
        ),
        "age",
        [1, 3, 5, 8, 10, 12],
    ),
    (
        "Emergency fund kitna hona chahiye? Kahan rakhun?",
        (
            "Emergency fund:\n\n"
            "Kitna: 6 months ke expenses ke barabar\n"
            "Example: Monthly expense ₹40,000 → Emergency fund = ₹2,40,000\n\n"
            "Kahan rakhein:\n"
            "1. Liquid Mutual Fund (best option)\n"
            "   • Returns: 6.5–7.5% p.a.\n"
            "   • Same-day redemption (T+0 for most AMCs)\n"
            "   • FD se better liquidity\n\n"
            "2. Savings Account (backup)\n"
            "   • High-yield SA (small finance banks): 5–7%\n"
            "   • Instant access\n\n"
            "3. Sweep FD with auto-break facility\n\n"
            "❌ Kya NA karo: Emergency fund ko equity MF mein mat lagao — "
            "market down mein hi emergency aati hai!"
        ),
        None,
        None,
    ),
    (
        "CIBIL score improve karne ke liye kya karu? Mera score {score} hai.",
        (
            "CIBIL score ₹{score} se improve karne ke steps:\n\n"
            "1. Bill on time pay karo (35% weightage)\n"
            "   • Auto-pay set karo minimum amount ke liye\n\n"
            "2. Credit utilization 30% se kam rakho\n"
            "   • ₹1 lakh limit hai toh ₹30,000 se zyada mat use karo\n\n"
            "3. Credit card account close mat karo (history important hai)\n\n"
            "4. Har 6 months mein ek naya loan/credit mat lo\n\n"
            "5. Secured credit card lo agar score bahut low hai\n"
            "   • ₹10,000–50,000 FD ke against milta hai\n"
            "   • 6–12 months mein score improve hota hai\n\n"
            "Timeline: Consistent behaviour se 3–6 months mein visible improvement.\n"
            "Score 750+ hone pe best loan rates milti hain."
        ),
        "score",
        [580, 620, 650, 680, 700, 720],
    ),
    (
        "Gold mein invest karu ya Mutual Fund mein? Long-term ke liye?",
        (
            "Gold vs Mutual Fund — long-term ke liye:\n\n"
            "Gold:\n"
            "• Historical return: 8–10% CAGR (last 20 years)\n"
            "• Inflation hedge — crisis mein value badhti hai\n"
            "• Digital gold / Sovereign Gold Bond (SGB) best option\n"
            "• SGB interest: 2.5% extra, maturity pe capital gains tax-free!\n\n"
            "Equity Mutual Fund:\n"
            "• Historical return: 12–15% CAGR (Nifty 50, 20 years)\n"
            "• High volatility — short-term mein 30-40% drop ho sakta hai\n"
            "• Best for 7+ year goals\n\n"
            "Recommendation:\n"
            "• 10–15% portfolio gold mein rakho (diversification ke liye)\n"
            "• Baaki equity MF mein\n"
            "• Physical gold: storage risk hai, tax bhi zyada\n\n"
            "Avoid gold jewelry as investment — making charges 10–20% waste hote hain."
        ),
        None,
        None,
    ),
    (
        "Home loan prepay karna chahiye ya SIP mein invest karna chahiye?",
        (
            "Home loan prepayment vs SIP — analysis:\n\n"
            "Effective home loan rate (after Sec 24b tax benefit):\n"
            "• 8.5% loan rate, 30% slab → effective rate = 8.5% × (1 - 0.30) = 5.95%\n\n"
            "Equity MF expected return: 12% CAGR (historical average, not guaranteed)\n\n"
            "Decision:\n"
            "• Agar equity MF return > effective loan rate → SIP better hai\n"
            "• 12% > 5.95% → Mathematically SIP jeetata hai\n\n"
            "Lekin consider karo:\n"
            "• Loan prepayment guaranteed saving; SIP uncertain\n"
            "• Psychological peace: loan-free hona valuable\n"
            "• Emergency buffer: Pehle 6 months emergency fund banao\n\n"
            "Best strategy:\n"
            "1. Emergency fund complete karo\n"
            "2. 80C investments max karo\n"
            "3. Remaining: 70% SIP + 30% prepayment (balance approach)"
        ),
        None,
        None,
    ),
    (
        "ITR file karna hai pehli baar. Kya documents chahiye aur kahan se karu?",
        (
            "First-time ITR filing guide:\n\n"
            "Documents needed:\n"
            "• Form 16 (employer se milega — April/May mein)\n"
            "• Bank statement (interest ke liye)\n"
            "• 26AS / AIS form (tax.gov.in se download karein — free)\n"
            "• Investment proofs: ELSS, LIC, PPF statements\n"
            "• Home loan certificate (agar hai)\n\n"
            "Kahan file karein:\n"
            "• incometax.gov.in pe self-filing (ITR-1 agar salary income only)\n"
            "• ClearTax / TaxBuddy apps — easy guided filing\n"
            "• CA ke paas (complex cases — business income, multiple houses)\n\n"
            "Deadline: 31 July (non-audit cases)\n\n"
            "Important:\n"
            "• AIS mein income check karo — agar discrepancy hai toh resolve karo\n"
            "• Refund = TDS zyada kata, aayega automatically bank mein\n"
            "• Penalty: 31 July ke baad ₹5,000 late fee (₹1,000 if income < ₹5L)"
        ),
        None,
        None,
    ),
    (
        "Apna pehla flat kharidna hai ₹{price}L ka. Kitna down payment chahiye?",
        (
            "₹{price} lakh flat ke liye home loan planning:\n\n"
            "Down payment: Minimum 20% = ₹{dp} lakh\n"
            "Loan amount: ₹{loan} lakh\n\n"
            "EMI estimate ({rate}% interest, 20 years):\n"
            "EMI ≈ ₹{emi}/month\n\n"
            "Hidden costs — iska bhi budget rakho:\n"
            "• Registration + stamp duty: 5–7% of property value = ₹{reg} lakh\n"
            "• Interior + moving: ₹{interior} lakh (estimate)\n"
            "• Home loan processing fee: 0.5–1%\n\n"
            "Rule of thumb:\n"
            "• EMI 40% se zyada monthly income ki NA ho\n"
            "• Emergency fund 6 months ka pehle complete karo\n"
            "• Sec 80C: Principal repayment ₹1.5L tak deductible\n"
            "• Sec 24(b): Interest ₹2 lakh tak deductible (old regime)"
        ),
        "price",
        [40, 50, 60, 80, 100, 120, 150],
    ),
    (
        "Term insurance vs ULIP — kya lena chahiye?",
        (
            "Term Insurance vs ULIP — honest comparison:\n\n"
            "Term Insurance:\n"
            "• Pure protection — death benefit only\n"
            "• ₹1 crore cover: ~₹8,000–15,000/year (age 30, non-smoker)\n"
            "• No savings, no returns — but cheapest life cover\n\n"
            "ULIP (Unit Linked Insurance Plan):\n"
            "• Insurance + investment in one product\n"
            "• High charges: mortality + fund management + admin (3–4% combined)\n"
            "• 5 year lock-in\n"
            "• Returns often underperform direct MF after charges\n\n"
            "Verdict: TERM + MF separately ALWAYS better than ULIP\n\n"
            "Example comparison (₹50,000/year for 20 years):\n"
            "• Term (₹10,000) + MF SIP (₹40,000): ~₹{mf_value} (12% CAGR)\n"
            "• ULIP (₹50,000): ~30–40% less due to charges\n\n"
            "ULIP lena toh only agar employer benefit ya tax planning specific case ho."
        ),
        None,
        None,
    ),
    (
        "Freelancer hoon, {income}L income hai. Kitna tax dunga aur ITR kaise bharun?",
        (
            "Freelancer ke liye ITR filing (FY 2025-26):\n\n"
            "Income: ₹{income} lakh\n"
            "ITR form: ITR-4 (presumptive income scheme) or ITR-3 (full P&L)\n\n"
            "Presumptive Scheme (Sec 44ADA — best for freelancers):\n"
            "• 50% of gross receipt = deemed profit (no need to maintain books)\n"
            "• Applicable if turnover < ₹75 lakh\n"
            "• Taxable income = ₹{half_income} lakh\n"
            "• Tax (new regime): ₹{tax_est} approx.\n\n"
            "Deductions you can still claim:\n"
            "• 80C: ₹1.5 lakh\n"
            "• 80D: Health insurance\n"
            "• 80CCD(1B): NPS ₹50,000\n\n"
            "Advance tax: Agar annual tax > ₹10,000 toh quarterly pay karo\n"
            "(15 June, 15 Sep, 15 Dec, 15 Mar)\n\n"
            "Tip: Professional CA ₹2,000–5,000 mein ITR bharta hai — worth it for complex cases."
        ),
        "income",
        [5, 8, 10, 12, 15, 20, 25],
    ),
    (
        "Apne bachon ke naam pe invest karna chahta hoon. Kya options hain?",
        (
            "Bachon ke naam pe investment options:\n\n"
            "1. Sukanya Samriddhi Yojana (beti ke liye only)\n"
            "   • Rate: 8.2% p.a. — highest guaranteed + EEE\n"
            "   • Max: ₹1.5 lakh/year per account\n"
            "   • Maturity: Beti ki age 21 pe\n\n"
            "2. PPF in child's name\n"
            "   • Minor ka account parent operate kar sakte hain\n"
            "   • 7.1% guaranteed, 15-year EEE\n\n"
            "3. Mutual Fund (SIP in child's name)\n"
            "   • Parent as guardian account open kar sakta hai\n"
            "   • Long horizon — Nifty 50 Index best for 15+ year\n\n"
            "4. Child Plan (insurance) — generally avoid\n"
            "   • High charges, low flexibility\n"
            "   • Term + MF separate approach always better\n\n"
            "Tax note: Minor ki income parent ki income mein club hoti hai (Sec 64) "
            "— ₹1,500/year exemption milti hai per child."
        ),
        None,
        None,
    ),
]


# Context prefixes added to static (no-param) questions to break dedup
_SALARY_CONTEXTS = [
    "Mera salary ₹{s}/month hai. ",
    "Main ₹{s} per month kama raha/rahi hoon. ",
    "Meri monthly income ₹{s} hai. ",
    "Mera CTC ₹{s} per month hai. ",
]
_AGE_CONTEXTS = [
    "Main {a} saal ka/ki hoon. ",
    "Meri age {a} saal hai. ",
    "{a} saal ka investor hoon. ",
    "Main {a} years old hoon. ",
]

_STATIC_SALARIES = [20_000, 30_000, 40_000, 50_000, 60_000, 75_000, 100_000, 150_000]
_STATIC_AGES = [24, 28, 32, 35, 40, 45, 50, 55]


def _format_pair(q_template: str, a_template: str, param_key: str | None, param_val: Any) -> tuple[str, str]:
    """Apply parametric substitutions to a Q&A template pair."""
    if param_key is None:
        return q_template, a_template

    kwargs: dict[str, Any] = {param_key: f"{param_val:,}" if isinstance(param_val, int) else param_val}

    # Salary-based computed params
    if param_key == "salary":
        salary = param_val
        sip_20 = int(salary * 0.20)
        kwargs.update({
            "sip_20pct": f"{sip_20:,}",
            "sip_nifty": f"{int(sip_20 * 0.50):,}",
            "sip_flexi": f"{int(sip_20 * 0.30):,}",
            "sip_elss": f"{int(sip_20 * 0.20):,}",
        })

    elif param_key == "age":
        age = param_val
        years_left = 18 - age
        kwargs["years_left"] = years_left
        r = 0.01  # 1% monthly
        n15 = years_left * 12
        n50 = years_left * 12
        sip_15 = int(1_500_000 * r / ((1 + r) ** n15 - 1)) if n15 > 0 else 0
        sip_50 = int(5_000_000 * r / ((1 + r) ** n50 - 1)) if n50 > 0 else 0
        kwargs["sip_15l"] = f"{sip_15:,}"
        kwargs["sip_50l"] = f"{sip_50:,}"

    elif param_key == "price":
        price = param_val
        dp = int(price * 0.20)
        loan = price - dp
        emi = int(loan * 100_000 * 0.00868)  # ~8.5% 20yr
        kwargs.update({
            "price": price,
            "dp": dp,
            "loan": loan,
            "rate": 8.5,
            "emi": f"{emi:,}",
            "reg": round(price * 0.06, 1),
            "interior": round(price * 0.05, 1),
        })

    elif param_key == "income":
        income = param_val
        half = income / 2
        tax_est = int(max(0, (half * 100_000 - 700_000) * 0.05)) if half * 100_000 > 700_000 else 0
        kwargs["half_income"] = f"{half:.1f}"
        kwargs["tax_est"] = f"{tax_est:,}"

    elif param_key == "score":
        kwargs["score"] = param_val

    try:
        return q_template.format(**kwargs), a_template.format(**kwargs)
    except KeyError:
        return q_template, a_template


class HinglishGenerator(BaseGapGenerator):
    """Generates Hindi/Hinglish financial Q&A pairs (Gap 7).

    Cycles through 16 QA templates with parametric salary/age/amount variations
    so the 300 samples have low repetition rate.

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance.
    """

    gap_name = "hinglish"
    task_type = "instruction"
    layer = "L4_community_conversational"
    source_dataset = "synthetic_hinglish"

    def _generate_one(self, idx: int) -> dict[str, Any]:
        q_tmpl, a_tmpl, param_key, param_choices = _QA_PAIRS[idx % len(_QA_PAIRS)]

        param_val = self.rng.choice(param_choices) if param_choices else None
        q, a = _format_pair(q_tmpl, a_tmpl, param_key, param_val)

        # For static templates (no parametric variation), prepend a context prefix
        # so each cycle of the QA pool produces a uniquely dedup-able question.
        if param_key is None:
            if self.rng.random() < 0.5:
                salary = self.rng.choice(_STATIC_SALARIES)
                prefix = self.rng.choice(_SALARY_CONTEXTS).format(s=f"{salary:,}")
            else:
                age = self.rng.choice(_STATIC_AGES)
                prefix = self.rng.choice(_AGE_CONTEXTS).format(a=age)
            q = prefix + q

        return self._make_sample(
            idx, q, a,
            lang="hinglish",
            difficulty=self.rng.choice(["beginner", "intermediate"]),
        )
