"""Deterministic math checks for Indian personal finance — FY 2024-25.

All calculations use correct FY 2024-25 tax slabs, standard deductions, 87A rebate,
and 4% Health & Education Cess. No LLM involved — pure deterministic evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class TaxResult:
    """FY 2024-25 tax computation output."""

    annual_income: float
    regime: str  # "new" or "old"
    taxable_income: float
    tax_before_cess: float
    rebate_87a: float
    cess: float
    tax_payable: float
    effective_rate: float
    slab_breakdown: list[dict] = field(default_factory=list)


@dataclass
class MetricResult:
    """Single metric evaluation result."""

    passed: bool
    score: float  # 0.0–1.0
    expected: float
    extracted: Optional[float]
    tolerance_pct: float
    notes: str = ""


# ─── Tax Math Checker ──────────────────────────────────────────────────────────


class TaxMathChecker:
    """FY 2024-25 Indian income tax checker — new and old regimes.

    New regime slabs (FY 2024-25, per Budget 2024):
        0–3L: 0%, 3–7L: 5%, 7–10L: 10%, 10–12L: 15%, 12–15L: 20%, >15L: 30%
        Standard deduction (salaried): ₹75,000
        Section 87A rebate: up to ₹25,000 if taxable income ≤ ₹7,00,000

    Old regime slabs:
        0–2.5L: 0%, 2.5–5L: 5%, 5–10L: 20%, >10L: 30%
        Standard deduction (salaried): ₹50,000
        Section 87A rebate: up to ₹12,500 if taxable income ≤ ₹5,00,000

    Both regimes: 4% Health & Education Cess on tax payable after rebate.
    """

    _NEW_SLABS = [
        (300_000, 0.00),
        (700_000, 0.05),
        (1_000_000, 0.10),
        (1_200_000, 0.15),
        (1_500_000, 0.20),
        (float("inf"), 0.30),
    ]

    _OLD_SLABS = [
        (250_000, 0.00),
        (500_000, 0.05),
        (1_000_000, 0.20),
        (float("inf"), 0.30),
    ]

    _CESS_RATE = 0.04

    def compute_new_regime(
        self, annual_income: float, is_salaried: bool = True
    ) -> TaxResult:
        """Compute tax under new regime for FY 2024-25.

        Args:
            annual_income: Gross annual income in INR.
            is_salaried: If True, applies ₹75,000 standard deduction.

        Returns:
            TaxResult with full breakdown.
        """
        std_deduction = 75_000 if is_salaried else 0
        taxable = max(0.0, annual_income - std_deduction)

        tax_raw, breakdown = self._apply_slabs(taxable, self._NEW_SLABS)

        rebate = min(tax_raw, 25_000) if taxable <= 700_000 else 0.0
        tax_after_rebate = max(0.0, tax_raw - rebate)
        cess = round(tax_after_rebate * self._CESS_RATE)
        tax_payable = round(tax_after_rebate + cess)

        effective_rate = (tax_payable / annual_income * 100) if annual_income > 0 else 0.0

        return TaxResult(
            annual_income=annual_income,
            regime="new",
            taxable_income=taxable,
            tax_before_cess=tax_raw,
            rebate_87a=rebate,
            cess=cess,
            tax_payable=tax_payable,
            effective_rate=round(effective_rate, 2),
            slab_breakdown=breakdown,
        )

    def compute_old_regime(
        self,
        annual_income: float,
        is_salaried: bool = True,
        deductions_80c: float = 0,
        deductions_80d: float = 0,
        hra_exemption: float = 0,
        other_deductions: float = 0,
    ) -> TaxResult:
        """Compute tax under old regime for FY 2024-25.

        Args:
            annual_income: Gross annual income in INR.
            is_salaried: If True, applies ₹50,000 standard deduction.
            deductions_80c: 80C investments (capped at ₹1,50,000).
            deductions_80d: 80D health insurance premium.
            hra_exemption: HRA exemption amount under Section 10(13A).
            other_deductions: Any other eligible deductions.

        Returns:
            TaxResult with full breakdown.
        """
        std_deduction = 50_000 if is_salaried else 0
        cap_80c = min(deductions_80c, 150_000)
        total_deductions = std_deduction + cap_80c + deductions_80d + hra_exemption + other_deductions
        taxable = max(0.0, annual_income - total_deductions)

        tax_raw, breakdown = self._apply_slabs(taxable, self._OLD_SLABS)

        rebate = min(tax_raw, 12_500) if taxable <= 500_000 else 0.0
        tax_after_rebate = max(0.0, tax_raw - rebate)
        cess = round(tax_after_rebate * self._CESS_RATE)
        tax_payable = round(tax_after_rebate + cess)

        effective_rate = (tax_payable / annual_income * 100) if annual_income > 0 else 0.0

        return TaxResult(
            annual_income=annual_income,
            regime="old",
            taxable_income=taxable,
            tax_before_cess=tax_raw,
            rebate_87a=rebate,
            cess=cess,
            tax_payable=tax_payable,
            effective_rate=round(effective_rate, 2),
            slab_breakdown=breakdown,
        )

    def score_response(
        self,
        response: str,
        expected: TaxResult,
        tolerance_pct: float = 2.0,
    ) -> MetricResult:
        """Extract the tax amount from a model response and compare to expected.

        Args:
            response: Raw model response text.
            expected: Expected TaxResult from deterministic computation.
            tolerance_pct: Acceptable percentage deviation (default 2%).

        Returns:
            MetricResult with pass/fail and score.
        """
        extracted = self._extract_rupee_amount(response, expected.tax_payable)
        if extracted is None:
            logger.debug("tax_score: could not extract amount from response")
            return MetricResult(
                passed=False,
                score=0.0,
                expected=expected.tax_payable,
                extracted=None,
                tolerance_pct=tolerance_pct,
                notes="Could not extract tax amount from response",
            )

        if expected.tax_payable == 0:
            passed = extracted < 1_000  # allow small rounding
            score = 1.0 if passed else 0.0
        else:
            pct_error = abs(extracted - expected.tax_payable) / expected.tax_payable * 100
            passed = pct_error <= tolerance_pct
            score = max(0.0, 1.0 - pct_error / 100)

        return MetricResult(
            passed=passed,
            score=round(score, 3),
            expected=expected.tax_payable,
            extracted=extracted,
            tolerance_pct=tolerance_pct,
            notes=f"Extracted ₹{extracted:,.0f}, expected ₹{expected.tax_payable:,.0f}",
        )

    @staticmethod
    def _apply_slabs(taxable: float, slabs: list[tuple]) -> tuple[float, list[dict]]:
        """Apply progressive tax slabs and return (total_tax, breakdown)."""
        tax = 0.0
        breakdown = []
        prev = 0.0
        for limit, rate in slabs:
            if taxable <= prev:
                break
            slab_income = min(taxable, limit) - prev
            slab_tax = slab_income * rate
            tax += slab_tax
            breakdown.append({"from": prev, "to": min(taxable, limit), "rate": rate, "tax": slab_tax})
            prev = limit
        return tax, breakdown

    @staticmethod
    def _extract_rupee_amount(text: str, hint: float) -> Optional[float]:
        """Extract the most relevant rupee amount from response text.

        Tries common Indian number formats: ₹44,200 / Rs. 44200 / 44,200 / 44.2 lakh.
        Uses `hint` (the expected value) to pick the closest match when multiple found.
        """
        patterns = [
            r"₹\s*([\d,]+(?:\.\d+)?)\s*(?:lakh|lakhs|L)?",
            r"Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(?:lakh|lakhs|L)?",
            r"INR\s*([\d,]+(?:\.\d+)?)\s*(?:lakh|lakhs|L)?",
            r"([\d,]+(?:\.\d+)?)\s*(?:lakh|lakhs|L)\b",
            r"(?:tax|payable|liability)[^\d]*([\d,]+(?:\.\d+)?)",
            r"\b([\d,]{5,})\b",  # bare numbers >= 10000
        ]
        candidates: list[float] = []
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                raw = match.group(1).replace(",", "")
                try:
                    val = float(raw)
                    if "lakh" in match.group(0).lower() or match.group(0).lower().endswith("l"):
                        val *= 100_000
                    candidates.append(val)
                except ValueError:
                    pass

        if not candidates:
            return None

        # Return candidate closest to the hint value
        return min(candidates, key=lambda x: abs(x - hint))


# ─── SIP Math Checker ─────────────────────────────────────────────────────────


class SIPMathChecker:
    """SIP maturity, lumpsum FV, step-up SIP, and reverse SIP computation.

    Standard SIP future value (end-of-period payments):
        FV = P × [(1+r)^n - 1] / r × (1+r)
    where r = monthly rate = annual_rate / 12, n = years × 12.

    Lumpsum FV:
        FV = P × (1 + annual_rate)^years
    """

    def compute_sip_maturity(
        self,
        monthly_amount: float,
        annual_rate: float,
        years: int,
    ) -> float:
        """Compute SIP maturity amount.

        Args:
            monthly_amount: Monthly SIP instalment in INR.
            annual_rate: Expected annual return rate as decimal (e.g. 0.12 for 12%).
            years: Investment duration in years.

        Returns:
            Maturity amount in INR (rounded to nearest rupee).
        """
        r = annual_rate / 12
        n = years * 12
        if r == 0:
            return monthly_amount * n
        fv = monthly_amount * ((1 + r) ** n - 1) / r * (1 + r)
        return round(fv)

    def compute_lumpsum(self, amount: float, annual_rate: float, years: int) -> float:
        """Compute lumpsum investment future value.

        Args:
            amount: Initial investment in INR.
            annual_rate: Annual return rate as decimal.
            years: Investment duration in years.

        Returns:
            Future value in INR.
        """
        return round(amount * (1 + annual_rate) ** years)

    def compute_step_up_sip(
        self,
        initial_monthly: float,
        annual_rate: float,
        years: int,
        step_up_rate: float,
    ) -> float:
        """Compute step-up (increasing) SIP maturity amount.

        Each year the monthly SIP amount increases by `step_up_rate`.

        Args:
            initial_monthly: First year's monthly SIP amount in INR.
            annual_rate: Expected annual return rate as decimal.
            years: Total investment duration in years.
            step_up_rate: Annual increase rate as decimal (e.g. 0.10 for 10%).

        Returns:
            Maturity amount in INR.
        """
        r = annual_rate / 12
        total_fv = 0.0
        for year in range(years):
            monthly = initial_monthly * ((1 + step_up_rate) ** year)
            months_remaining = (years - year) * 12
            if r == 0:
                year_fv = monthly * months_remaining
            else:
                year_fv = monthly * ((1 + r) ** months_remaining - 1) / r * (1 + r)
            total_fv += year_fv
        return round(total_fv)

    def compute_reverse_sip(
        self, target_amount: float, annual_rate: float, years: int
    ) -> float:
        """Compute monthly SIP needed to reach a target corpus.

        Args:
            target_amount: Desired future corpus in INR.
            annual_rate: Expected annual return rate as decimal.
            years: Investment duration in years.

        Returns:
            Required monthly SIP instalment in INR.
        """
        r = annual_rate / 12
        n = years * 12
        if r == 0:
            return round(target_amount / n)
        monthly = target_amount * r / ((1 + r) ** n - 1) / (1 + r)
        return round(monthly)

    def score_response(
        self,
        response: str,
        expected: float,
        tolerance_pct: float = 2.0,
    ) -> MetricResult:
        """Extract maturity/corpus amount from response and compare to expected.

        Args:
            response: Raw model response text.
            expected: Expected maturity amount in INR.
            tolerance_pct: Acceptable percentage deviation.

        Returns:
            MetricResult.
        """
        extracted = self._extract_amount(response, expected)
        if extracted is None:
            return MetricResult(
                passed=False,
                score=0.0,
                expected=expected,
                extracted=None,
                tolerance_pct=tolerance_pct,
                notes="Could not extract maturity amount from response",
            )

        pct_error = abs(extracted - expected) / max(expected, 1) * 100
        passed = pct_error <= tolerance_pct
        score = max(0.0, 1.0 - pct_error / 100)

        return MetricResult(
            passed=passed,
            score=round(score, 3),
            expected=expected,
            extracted=extracted,
            tolerance_pct=tolerance_pct,
            notes=f"Extracted ₹{extracted:,.0f}, expected ₹{expected:,.0f}",
        )

    @staticmethod
    def _extract_amount(text: str, hint: float) -> Optional[float]:
        """Extract the closest monetary amount to hint from text."""
        # Match patterns like ₹11.62 lakh, ₹11,62,000, 11.62 lakhs, Rs 11.62 lakh
        patterns = [
            r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)\s*(crore|cr|lakh|lakhs|L)?",
            r"([\d,]+(?:\.\d+)?)\s*(crore|cr|lakh|lakhs|L)\b",
            r"(?:corpus|maturity|amount|returns?|value)[^\d]*([\d,]+(?:\.\d+)?)\s*(crore|cr|lakh|lakhs|L)?",
            r"\b([\d,]{5,}(?:\.\d+)?)\b",
        ]
        candidates: list[float] = []
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                raw = m.group(1).replace(",", "")
                try:
                    val = float(raw)
                    unit = (m.group(2) or "").lower()
                    if "crore" in unit or unit == "cr":
                        val *= 1e7
                    elif "lakh" in unit or unit == "l":
                        val *= 1e5
                    candidates.append(val)
                except (ValueError, IndexError):
                    pass

        if not candidates:
            return None
        return min(candidates, key=lambda x: abs(x - hint))


# ─── EMI Math Checker ─────────────────────────────────────────────────────────


class EMIMathChecker:
    """EMI calculation and amortization checks.

    Standard EMI formula:
        EMI = P × r × (1+r)^n / ((1+r)^n - 1)
    where P = principal, r = monthly rate = annual_rate/12, n = tenure_months.
    """

    def compute_emi(
        self, principal: float, annual_rate: float, tenure_months: int
    ) -> float:
        """Compute monthly EMI for a loan.

        Args:
            principal: Loan amount in INR.
            annual_rate: Annual interest rate as decimal (e.g. 0.085 for 8.5%).
            tenure_months: Loan tenure in months.

        Returns:
            Monthly EMI in INR (rounded to nearest rupee).
        """
        r = annual_rate / 12
        n = tenure_months
        if r == 0:
            return round(principal / n)
        emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
        return round(emi)

    def compute_total_interest(
        self, principal: float, annual_rate: float, tenure_months: int
    ) -> float:
        """Compute total interest payable over the loan tenure.

        Args:
            principal: Loan amount in INR.
            annual_rate: Annual interest rate as decimal.
            tenure_months: Loan tenure in months.

        Returns:
            Total interest in INR.
        """
        emi = self.compute_emi(principal, annual_rate, tenure_months)
        return round(emi * tenure_months - principal)

    def compute_outstanding_balance(
        self,
        principal: float,
        annual_rate: float,
        tenure_months: int,
        emis_paid: int,
    ) -> float:
        """Compute outstanding loan balance after N EMIs paid.

        Args:
            principal: Original loan amount in INR.
            annual_rate: Annual interest rate as decimal.
            tenure_months: Total loan tenure in months.
            emis_paid: Number of EMIs already paid.

        Returns:
            Outstanding balance in INR.
        """
        r = annual_rate / 12
        n = tenure_months
        emi = self.compute_emi(principal, annual_rate, tenure_months)
        # Balance after k payments: P(1+r)^k - EMI × ((1+r)^k - 1)/r
        k = emis_paid
        if r == 0:
            return round(principal - emi * k)
        balance = principal * (1 + r) ** k - emi * ((1 + r) ** k - 1) / r
        return round(max(0.0, balance))

    def score_response(
        self,
        response: str,
        expected_emi: float,
        tolerance_pct: float = 1.0,
    ) -> MetricResult:
        """Extract EMI amount from response and compare to expected.

        Args:
            response: Raw model response text.
            expected_emi: Expected monthly EMI in INR.
            tolerance_pct: Acceptable percentage deviation.

        Returns:
            MetricResult.
        """
        extracted = SIPMathChecker._extract_amount(response, expected_emi)
        if extracted is None:
            return MetricResult(
                passed=False,
                score=0.0,
                expected=expected_emi,
                extracted=None,
                tolerance_pct=tolerance_pct,
                notes="Could not extract EMI from response",
            )

        pct_error = abs(extracted - expected_emi) / max(expected_emi, 1) * 100
        passed = pct_error <= tolerance_pct
        score = max(0.0, 1.0 - pct_error / 100)

        return MetricResult(
            passed=passed,
            score=round(score, 3),
            expected=expected_emi,
            extracted=extracted,
            tolerance_pct=tolerance_pct,
            notes=f"Extracted ₹{extracted:,.0f}, expected ₹{expected_emi:,.0f}",
        )
