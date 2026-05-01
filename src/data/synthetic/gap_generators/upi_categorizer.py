"""Gap 8 — UPI transaction categorisation generator."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .base import BaseGapGenerator

# (description_template, category, display_name, amount)
_UPI_PATTERNS: list[tuple[str, str, str, int]] = [
    ("UPI/DR/40712XXXXX/SWIGGY/swiggy@ybl/HDFC", "food_delivery", "Swiggy", 456),
    ("UPI/DR/40898XXXXX/AMAZON-PAY/merchant@apl/SBI", "shopping", "Amazon Pay", 2499),
    ("UPI/DR/40911XXXXX/UBER-INDIA/uber@axisbank/ICICI", "transport", "Uber", 186),
    ("UPI/CR/40755XXXXX/SALARY-EMPLOYER/hr@corp/HDFC", "salary", "Employer Salary", 85000),
    ("UPI/DR/41033XXXXX/BLINKIT/blinkit@razorpay/Kotak", "groceries", "Blinkit", 847),
    ("UPI/DR/41144XXXXX/NETFLIX/netflix@icici/ICICI", "entertainment", "Netflix", 649),
    ("UPI/DR/41277XXXXX/HPCL-PETROL/fuel@sbi/SBI", "fuel", "HPCL Petrol", 2100),
    ("UPI/DR/41311XXXXX/LIC-PREMIUM/lic@pnb/PNB", "insurance", "LIC Premium", 12500),
    ("UPI/DR/41311XXXXX/PPFAS-MF-SIP/mf@bse/BSE", "investment", "PPFAS SIP", 5000),
    ("NEFT/RENT-TRANSFER/MR-SHARMA/IMPS-REF123", "rent", "Rent Payment", 18000),
    ("EMI/HDFC-HOME-LOAN/LN00123456/AUTO-DEBIT", "emi", "Home Loan EMI", 25000),
    ("UPI/DR/41422XXXXX/IRCTC/irctc@sbi/SBI", "travel", "IRCTC Train Ticket", 1245),
    ("UPI/DR/41533XXXXX/APOLLO-PHARMACY/apl@paytm/Paytm", "pharmacy", "Apollo Pharmacy", 890),
    ("UPI/DR/41644XXXXX/ZERODHA/zerodha@hdfcbank/HDFC", "investment", "Zerodha Stocks", 15000),
    ("UPI/DR/41755XXXXX/ZOMATO/zomato@icici/ICICI", "food_delivery", "Zomato", 378),
    ("UPI/DR/41866XXXXX/OLA-CABS/ola@paytm/Paytm", "transport", "Ola Cabs", 220),
    ("UPI/DR/41977XXXXX/HOTSTAR-DISNEY/hotstar@axisbank/Axis", "entertainment", "Disney+ Hotstar", 299),
    ("UPI/DR/42088XXXXX/DMART/dmart@upi/SBI", "groceries", "D-Mart", 1240),
    ("UPI/DR/42199XXXXX/JIO-RECHARGE/jio@icici/ICICI", "utilities", "Jio Recharge", 349),
    ("UPI/DR/42200XXXXX/AIRTEL-POSTPAID/airtel@airtel/HDFC", "utilities", "Airtel Bill", 899),
    ("UPI/DR/42311XXXXX/BESCOM-ELECTRICITY/bescom@sbi/SBI", "utilities", "BESCOM Electricity", 1450),
    ("UPI/DR/42422XXXXX/HDFC-CREDITCARD/hdfc@hdfc/HDFC", "credit_card_payment", "HDFC Credit Card", 35000),
    ("UPI/CR/42533XXXXX/FREELANCE-CLIENT/client@ybl/HDFC", "freelance_income", "Freelance Income", 25000),
    ("UPI/DR/42644XXXXX/GYM-MEMBERSHIP/gym@razorpay/Kotak", "health_fitness", "Gym Membership", 2000),
    ("UPI/DR/42755XXXXX/MYNTRA/myntra@icici/ICICI", "shopping", "Myntra Fashion", 1899),
]

_DECODE_NOTES = (
    "\nUPI reference decoding guide:\n"
    "• UPI/CR = Credit (incoming money) | UPI/DR = Debit (outgoing)\n"
    "• The long number after CR/DR is the UPI transaction reference ID\n"
    "• @ybl = PhonePe | @paytm = Paytm | @apl = Amazon Pay | "
    "@razorpay = Razorpay | @icici/@hdfc/@sbi = bank's own UPI"
)


class UPICategorizerGenerator(BaseGapGenerator):
    """Generates UPI transaction categorisation Q&A pairs (Gap 8).

    Selects a random subset of transaction patterns and produces a question asking
    the model to categorise them, with a detailed answer that also explains the UPI
    reference format — a key skill for personal finance assistants.

    Args:
        samples_per_gap: Number of samples to produce.
        rng: Seeded Random instance.
    """

    gap_name = "upi_categorize"
    task_type = "categorization"
    layer = "L3_personal_finance"
    source_dataset = "synthetic_upi"

    def _generate_one(self, idx: int) -> dict[str, Any]:
        n_txns = self.rng.randint(5, 12)
        # Sample without replacement, then possibly repeat patterns with different amounts
        base = self.rng.sample(_UPI_PATTERNS * 2, n_txns)

        txn_lines = "\n".join(
            f"  {j+1}. {t[0]}  ₹{t[3]:,}" for j, t in enumerate(base)
        )
        q = f"Categorise these transactions from my bank statement:\n\n{txn_lines}"

        categorised: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "items": []})
        for t in base:
            categorised[t[1]]["total"] += t[3]
            categorised[t[1]]["items"].append(f"{t[2]}: ₹{t[3]:,}")

        cat_lines = "\n".join(
            f"  {cat.replace('_', ' ').title()}: ₹{info['total']:,}  "
            f"[{', '.join(info['items'])}]"
            for cat, info in sorted(categorised.items(), key=lambda x: -x[1]["total"])
        )
        total = sum(info["total"] for info in categorised.values())
        top4 = sorted(categorised.items(), key=lambda x: -x[1]["total"])[:4]
        pct_line = ", ".join(
            f"{cat.replace('_',' ').title()}: {info['total']/total*100:.0f}%"
            for cat, info in top4
        )

        a = (
            f"Transaction Categorisation:\n\n{cat_lines}\n\n"
            f"Total: ₹{total:,}\n"
            f"Top categories: {pct_line}"
            f"{_DECODE_NOTES}"
        )

        return self._make_sample(idx, q, a, difficulty="beginner")
