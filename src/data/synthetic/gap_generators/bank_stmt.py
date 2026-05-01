"""Gap 1 — Indian bank statement analysis generator."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .base import BaseGapGenerator

_UPI_MERCHANTS: dict[str, dict[str, Any]] = {
    "food_delivery": {"merchants": ["SWIGGY", "ZOMATO", "EATSURE"], "range": (99, 899)},
    "groceries": {"merchants": ["BIGBASKET", "BLINKIT", "ZEPTO", "DMART"], "range": (150, 4500)},
    "transport": {"merchants": ["UBER", "OLA", "RAPIDO"], "range": (49, 650)},
    "entertainment": {"merchants": ["NETFLIX", "HOTSTAR", "BOOKMYSHOW"], "range": (149, 1200)},
    "shopping": {"merchants": ["AMAZON", "FLIPKART", "MYNTRA", "NYKAA"], "range": (199, 8000)},
    "utilities": {"merchants": ["JIO", "AIRTEL", "BSNL", "TATAPOWER"], "range": (100, 3500)},
    "fuel": {"merchants": ["HPCL", "BPCL", "IOCL"], "range": (400, 3500)},
    "pharmacy": {"merchants": ["PHARMEASY", "1MG", "APOLLO"], "range": (80, 2500)},
    "education": {"merchants": ["UNACADEMY", "COURSERA", "UDEMY"], "range": (500, 5000)},
}

_SALARY_CHOICES = list(range(25_000, 350_001, 5_000))
_SIP_AMOUNTS = [500, 1000, 2000, 3000, 5000, 7500, 10000]
_EMI_AMOUNTS = [5000, 8000, 12000, 15000, 20000, 25000, 30000]
_EMI_TYPES = ["HOME LOAN", "CAR LOAN", "PERSONAL LOAN", "EDUCATION LOAN"]
_QTYPES = ["breakdown", "savings", "food_spend", "reduce_advice"]


class BankStatementGenerator(BaseGapGenerator):
    """Generates bank statement analysis Q&A pairs (Gap 1).

    Produces synthetic monthly statements with UPI transactions, SIPs, EMIs, and rent,
    then generates one of four question types per sample: spending breakdown, savings rate
    check, food-delivery deep-dive, or cost-reduction advice.

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance.
    """

    gap_name = "bank_stmt"
    task_type = "bank_analysis"
    layer = "L3_personal_finance"
    source_dataset = "synthetic_bank_stmt"

    def _generate_one(self, idx: int) -> dict[str, Any]:
        salary = self.rng.choice(_SALARY_CHOICES)
        rent = (
            round(salary * self.rng.uniform(0.12, 0.28) / 500) * 500
            if self.rng.random() > 0.25
            else 0
        )
        num_sips = self.rng.randint(0, 4)
        sip_amt = self.rng.choice(_SIP_AMOUNTS)
        has_emi = self.rng.random() > 0.45
        emi_amt = self.rng.choice(_EMI_AMOUNTS) if has_emi else 0
        emi_type = self.rng.choice(_EMI_TYPES) if has_emi else ""

        txns: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "count": 0})
        for _ in range(self.rng.randint(18, 55)):
            cat = self.rng.choice(list(_UPI_MERCHANTS.keys()))
            info = _UPI_MERCHANTS[cat]
            merchant = self.rng.choice(info["merchants"])
            amt = self.rng.randint(*info["range"])
            txns[cat]["total"] += amt
            txns[cat]["count"] += 1

        total_upi = sum(v["total"] for v in txns.values())
        total_sip = num_sips * sip_amt
        total_outflow = total_upi + rent + total_sip + emi_amt
        savings = salary - total_outflow
        savings_rate = max(0, savings) / salary * 100
        n_upi = sum(v["count"] for v in txns.values())

        top3 = sorted(txns.items(), key=lambda x: -x[1]["total"])[:3]
        stmt = (
            f"Monthly Salary: ₹{salary:,} | Rent: ₹{rent:,} | "
            f"SIPs: {num_sips}×₹{sip_amt:,} | EMI: ₹{emi_amt:,}"
            + (f" ({emi_type})" if emi_type else "")
            + f"\nUPI Transactions ({n_upi} total): "
            + "; ".join(f"{c}: ₹{d['total']:,} ({d['count']} txns)" for c, d in top3)
        )

        qtype = self.rng.choice(_QTYPES)
        q, a = self._build_qa(
            qtype, stmt, salary, rent, txns, total_upi,
            total_sip, emi_amt, emi_type, total_outflow, savings, savings_rate,
        )

        lang = self.rng.choice(["en", "en", "en", "hinglish"])
        diff = self.rng.choice(["beginner", "intermediate"])
        return self._make_sample(idx, q, a, lang=lang, difficulty=diff)

    def _build_qa(
        self,
        qtype: str,
        stmt: str,
        salary: int,
        rent: int,
        txns: dict,
        total_upi: int,
        total_sip: int,
        emi_amt: int,
        emi_type: str,
        total_outflow: int,
        savings: int,
        savings_rate: float,
    ) -> tuple[str, str]:
        if qtype == "breakdown":
            q = f"[Bank Statement]\n{stmt}\n\nGive me a complete spending breakdown and savings analysis."
            breakdown = "\n".join(
                f"  {c}: ₹{d['total']:,} ({d['count']} txns, avg ₹{d['total']//max(1,d['count']):,})"
                for c, d in sorted(txns.items(), key=lambda x: -x[1]["total"])
            )
            warn = (
                "⚠️ Savings rate below 20%. Review discretionary spending."
                if savings_rate < 20
                else "✅ Savings rate is healthy (above 20%)."
            )
            a = (
                f"Monthly Spending Breakdown:\n\n{breakdown}\n\n"
                f"Fixed Costs:\n  Rent: ₹{rent:,}\n  SIPs: ₹{total_sip:,}\n"
                f"  EMI: ₹{emi_amt:,}" + (f" ({emi_type})" if emi_type else "")
                + f"\n\nTotal UPI Spend: ₹{total_upi:,}\n"
                f"Total Outflow: ₹{total_outflow:,}\n"
                f"Net Savings: ₹{savings:,}\n"
                f"Savings Rate: {savings_rate:.1f}%\n\n{warn}"
            )

        elif qtype == "food_spend":
            food = txns.get("food_delivery", {}).get("total", 0)
            food_n = txns.get("food_delivery", {}).get("count", 1)
            q = f"[Bank Statement]\n{stmt}\n\nHow much am I spending on food delivery? Is it too much?"
            pct = food / salary * 100
            comment = (
                "This exceeds 10% of income — consider meal-prepping 3 days/week to halve this."
                if pct > 10.0
                else "Under 10% of income — reasonable."
            )
            a = (
                f"Food delivery spend: ₹{food:,} across {food_n} orders "
                f"(avg ₹{food//max(1,food_n):,}/order).\n"
                f"As % of income: {pct:.1f}%\n\n{comment}\n"
                f"Annual projection: ₹{food*12:,}/year. "
                f"Cutting by 50% saves ₹{food*6:,}/year."
            )

        elif qtype == "savings":
            utilities = txns.get("utilities", {}).get("total", 0)
            food_d = txns.get("food_delivery", {}).get("total", 0)
            shopping = txns.get("shopping", {}).get("total", 0)
            entertainment = txns.get("entertainment", {}).get("total", 0)
            needs = rent + emi_amt + utilities
            wants = food_d + shopping + entertainment
            invest = total_sip + max(0, savings)
            sip_advice = (
                f"⚠️ Increase SIP by at least ₹{int(salary*0.20 - total_sip):,} more."
                if total_sip / salary < 0.20
                else "✅ Investment rate is strong."
            )
            q = f"[Bank Statement]\n{stmt}\n\nWhat is my savings rate? Am I on track?"
            a = (
                f"Income: ₹{salary:,} | Outflows: ₹{total_outflow:,} | Savings: ₹{savings:,}\n"
                f"Savings Rate: {savings_rate:.1f}%\n\n"
                f"50/30/20 Framework:\n"
                f"  Needs (rent+EMI+utilities): ₹{needs:,} ({needs/salary*100:.0f}%)\n"
                f"  Wants (food+shopping+ent): ₹{wants:,} ({wants/salary*100:.0f}%)\n"
                f"  Save/Invest: ₹{invest:,} ({invest/salary*100:.0f}%)\n\n{sip_advice}"
            )

        else:  # reduce_advice
            food_d = txns.get("food_delivery", {}).get("total", 0)
            shopping = txns.get("shopping", {}).get("total", 0)
            entertainment = txns.get("entertainment", {}).get("total", 0)
            groceries = txns.get("groceries", {}).get("total", 0)
            suggestions: list[str] = []
            if food_d > salary * 0.08:
                suggestions.append(
                    f"1. Cut food delivery by 50%: save ₹{food_d//2:,}/month (meal-prep weekdays)"
                )
            if shopping > salary * 0.05:
                suggestions.append(
                    f"{len(suggestions)+1}. 48-hour rule before online purchases: "
                    f"save ~₹{shopping//3:,}/month"
                )
            if entertainment > 1500:
                suggestions.append(
                    f"{len(suggestions)+1}. Keep max 2 streaming subscriptions: "
                    f"save ~₹{entertainment//2:,}/month"
                )
            if len(suggestions) < 3:
                suggestions.append(
                    f"{len(suggestions)+1}. Switch to store brands for groceries: "
                    f"save ~₹{groceries//5:,}/month"
                )
            pot = food_d // 3 + shopping // 3 + entertainment // 3
            q = f"[Bank Statement]\n{stmt}\n\nSuggest 3 actionable ways to reduce my expenses."
            a = "Expense Reduction Plan:\n\n" + "\n".join(suggestions[:3]) + f"\n\nTotal potential monthly savings: ₹{pot:,}"

        return q, a
