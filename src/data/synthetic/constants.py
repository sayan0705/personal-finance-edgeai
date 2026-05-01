"""FY 2024-25 Indian personal finance constants.

These are domain facts for generating factually accurate training data and
for the TaxCalculator tool. They encode the Finance Act 2024 rules.
"""

from __future__ import annotations

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are FinEdge, an expert personal finance advisor specializing in Indian personal "
    "finance including income tax, mutual funds, insurance, banking, and investments. "
    "Always provide accurate, actionable advice based on current Indian regulations."
)

# ── Income Tax (FY 2024-25) ───────────────────────────────────────────────────

# New regime slabs (default regime from FY 2023-24 onwards)
# Finance Act 2024: enhanced std deduction ₹75,000 + 87A rebate up to ₹7L
NEW_REGIME_SLABS = [
    (300_000, 0.00),     # 0 – 3 L → 0 %
    (700_000, 0.05),     # 3 – 7 L → 5 %
    (1_000_000, 0.10),   # 7 – 10 L → 10 %
    (1_200_000, 0.15),   # 10 – 12 L → 15 %
    (1_500_000, 0.20),   # 12 – 15 L → 20 %
    (float("inf"), 0.30),  # > 15 L → 30 %
]

NEW_REGIME_STANDARD_DEDUCTION = 75_000   # Budget 2024 enhancement
NEW_REGIME_87A_REBATE_LIMIT = 700_000    # taxable income threshold
NEW_REGIME_87A_REBATE_MAX = 25_000

# Old regime slabs
OLD_REGIME_SLABS = [
    (250_000, 0.00),
    (500_000, 0.05),
    (1_000_000, 0.20),
    (float("inf"), 0.30),
]

OLD_REGIME_STANDARD_DEDUCTION = 50_000
OLD_REGIME_87A_REBATE_LIMIT = 500_000
OLD_REGIME_87A_REBATE_MAX = 12_500

HEALTH_AND_EDUCATION_CESS = 0.04  # 4 % on income tax

# ── Old-regime deduction limits ───────────────────────────────────────────────

DEDUCTION_80C_LIMIT = 150_000          # PPF, ELSS, life insurance premium, etc.
DEDUCTION_80D_SELF_LIMIT = 25_000      # Health insurance premium (self + family)
DEDUCTION_80D_PARENTS_LIMIT = 25_000   # Parents (50,000 if senior citizen)
DEDUCTION_80D_PARENTS_SENIOR_LIMIT = 50_000
DEDUCTION_80CCD1B_LIMIT = 50_000       # NPS Tier-1 additional (over 80C)
DEDUCTION_80G_MAX_PERCENTAGE = 1.0     # Up to 100 % of donation (select funds)
DEDUCTION_HRA_MAX_PERCENTAGE = 0.50   # Max 50 % of basic (metro cities)
DEDUCTION_24B_HOME_LOAN_LIMIT = 200_000  # Interest on self-occupied property

# ── Equity LTCG / STCG ───────────────────────────────────────────────────────

EQUITY_LTCG_EXEMPTION = 100_000        # First ₹1 L LTCG is exempt
EQUITY_LTCG_RATE = 0.125               # 12.5 % (Budget 2024, was 10 %)
EQUITY_STCG_RATE = 0.20                # 20 % (Budget 2024, was 15 %)
EQUITY_HOLDING_THRESHOLD_DAYS = 365

# ── Debt / other instruments LTCG ────────────────────────────────────────────

DEBT_MF_LTCG_RATE = 0.20               # Without indexation (post Mar 2023 change)
DEBT_MF_STCG_RATE = None               # Added to income, taxed at slab rate

# ── PPF / EPF / NPS rates (illustrative — subject to quarterly revision) ─────

PPF_INTEREST_RATE = 0.071              # 7.1 % p.a. (tax-free, EEE)
EPF_INTEREST_RATE = 0.0825             # 8.25 % p.a. (FY 2024-25, announced Mar 2024)
NPS_ANNUITY_REQUIREMENT = 0.40         # Minimum 40 % must be used to buy annuity
NPS_60_PERCENT_LUMP_SUM_EXEMPT = True  # 60 % lump sum is tax-free on maturity

# ── Surcharge thresholds ──────────────────────────────────────────────────────

SURCHARGE_THRESHOLDS = [
    (5_000_000, 0.10),   # income > 50 L → 10 %
    (10_000_000, 0.15),  # income > 1 Cr → 15 %
    (20_000_000, 0.25),  # income > 2 Cr → 25 %
    (50_000_000, 0.37),  # income > 5 Cr → 37 % (not in new regime)
]

NEW_REGIME_MAX_SURCHARGE = 0.25        # Capped at 25 % in new regime

# ── Useful income brackets for synthetic data ─────────────────────────────────

INCOME_BRACKETS = [
    300_000, 500_000, 600_000, 700_000, 800_000,
    900_000, 1_000_000, 1_200_000, 1_500_000,
    2_000_000, 2_500_000, 3_000_000, 5_000_000,
]

# ── SIP / investment constants ────────────────────────────────────────────────

TYPICAL_EQUITY_MF_CAGR = [0.10, 0.12, 0.14, 0.15]   # 10–15 % range
TYPICAL_DEBT_MF_CAGR = [0.06, 0.07, 0.08]
ELSS_LOCK_IN_YEARS = 3
PPF_TENURE_YEARS = 15
PPF_MIN_ANNUAL_CONTRIBUTION = 500
PPF_MAX_ANNUAL_CONTRIBUTION = 150_000

# ── Loan constants ────────────────────────────────────────────────────────────

HOME_LOAN_TYPICAL_RATES = [0.085, 0.09, 0.095, 0.10]   # 8.5 – 10 %
PERSONAL_LOAN_TYPICAL_RATES = [0.12, 0.14, 0.16, 0.18]
CAR_LOAN_TYPICAL_RATES = [0.085, 0.09, 0.095]
EDUCATION_LOAN_TYPICAL_RATES = [0.085, 0.10, 0.11]

# ── Insurance constants ───────────────────────────────────────────────────────

TERM_COVER_MULTIPLE = 15               # Rule of thumb: 15× annual income
TERM_POLICY_TENURE_OPTIONS = [20, 25, 30, 35, 40]
HEALTH_COVER_RECOMMENDED_METRO = 1_000_000   # ₹10 L for metro cities
HEALTH_COVER_RECOMMENDED_NON_METRO = 500_000
