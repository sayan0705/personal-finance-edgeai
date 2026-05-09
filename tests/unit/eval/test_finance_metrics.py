"""Unit tests for eval/metrics/finance_metrics.py.

Tests deterministic math checkers — no LLM or model required.
"""

from __future__ import annotations

import pytest
from loguru import logger

from eval.metrics.finance_metrics import EMIMathChecker, SIPMathChecker, TaxMathChecker


# ─── TaxMathChecker ──────────────────────────────────────────────────────────


class TestTaxMathCheckerNewRegime:
    """FY 2024-25 new regime: 0–3L:0%, 3–7L:5%, 7–10L:10%, 10–12L:15%, 12–15L:20%, >15L:30%.
    ₹75K std deduction. 87A rebate ≤ ₹25K if net taxable ≤ ₹7L. 4% cess."""

    def setup_method(self):
        self.checker = TaxMathChecker()

    def test_below_300k_zero_tax(self):
        result = self.checker.compute_new_regime(300_000)
        assert result.tax_payable == 0, f"Expected 0 for 3L income, got {result.tax_payable}"

    def test_500k_fully_rebated(self):
        # After ₹75K std deduction: 4.25L net → tax = ₹6,250 → rebate ≤ ₹25K → 0
        result = self.checker.compute_new_regime(500_000)
        assert result.tax_payable == 0, f"Expected 0 (rebate) for 5L, got {result.tax_payable}"

    def test_700k_boundary(self):
        # 7L gross → 6.25L net → tax = ₹16,250 → rebate → 0
        result = self.checker.compute_new_regime(700_000)
        assert result.tax_payable == 0, f"Expected 0 (rebate) for 7L, got {result.tax_payable}"

    def test_775k_marginal_relief_zero(self):
        # 7.75L → 7L net → just above rebate threshold; marginal relief keeps net tax 0 or near 0
        result = self.checker.compute_new_regime(775_000)
        assert result.tax_payable == 0, f"Expected 0 (marginal relief) for 7.75L, got {result.tax_payable}"

    def test_800k_pays_tax(self):
        result = self.checker.compute_new_regime(800_000)
        assert result.tax_payable > 0, "Should pay tax at 8L gross"
        expected = 23_400
        assert abs(result.tax_payable - expected) <= max(500, expected * 0.02), (
            f"Expected ~{expected} for 8L, got {result.tax_payable}"
        )

    def test_1000k_new_regime(self):
        result = self.checker.compute_new_regime(1_000_000)
        expected = 44_200
        assert abs(result.tax_payable - expected) <= max(500, expected * 0.02), (
            f"Expected ~{expected} for 10L, got {result.tax_payable}"
        )

    def test_1200k_new_regime(self):
        result = self.checker.compute_new_regime(1_200_000)
        expected = 71_500
        assert abs(result.tax_payable - expected) <= max(500, expected * 0.02), (
            f"Expected ~{expected} for 12L, got {result.tax_payable}"
        )

    def test_1500k_new_regime(self):
        result = self.checker.compute_new_regime(1_500_000)
        expected = 130_000
        assert abs(result.tax_payable - expected) <= max(500, expected * 0.02), (
            f"Expected ~{expected} for 15L, got {result.tax_payable}"
        )

    def test_tax_result_has_required_fields(self):
        result = self.checker.compute_new_regime(1_000_000)
        assert hasattr(result, "tax_payable")
        assert hasattr(result, "effective_rate")
        assert result.effective_rate >= 0.0

    def test_cess_applied(self):
        # Cess is 4% on top — tax at 10L (before cess) ≈ 42,500 → after cess ≈ 44,200
        result = self.checker.compute_new_regime(1_000_000)
        assert result.tax_payable > 42_000, "Cess should push tax above base amount"


class TestTaxMathCheckerOldRegime:
    def setup_method(self):
        self.checker = TaxMathChecker()

    def test_1000k_old_no_deductions(self):
        result = self.checker.compute_old_regime(1_000_000)
        expected = 106_600
        assert abs(result.tax_payable - expected) <= max(500, expected * 0.02), (
            f"Expected ~{expected} for 10L old no deductions, got {result.tax_payable}"
        )

    def test_1000k_old_with_80c_80d(self):
        result = self.checker.compute_old_regime(
            1_000_000, deductions_80c=150_000, deductions_80d=25_000
        )
        expected = 70_200
        assert abs(result.tax_payable - expected) <= max(500, expected * 0.02), (
            f"Expected ~{expected} for 10L old with 80C+80D, got {result.tax_payable}"
        )

    def test_deductions_reduce_tax(self):
        no_deductions = self.checker.compute_old_regime(1_000_000)
        with_deductions = self.checker.compute_old_regime(1_000_000, deductions_80c=150_000)
        assert with_deductions.tax_payable < no_deductions.tax_payable


class TestTaxScoreResponse:
    def setup_method(self):
        self.checker = TaxMathChecker()

    def test_exact_match(self):
        result = self.checker.compute_new_regime(1_000_000)
        response = f"Your tax liability is ₹{result.tax_payable:,.0f}"
        score = self.checker.score_response(response, result, tolerance_pct=2.0)
        assert score.passed

    def test_wrong_amount_fails(self):
        result = self.checker.compute_new_regime(1_000_000)
        response = "Your tax is ₹1,00,000"
        score = self.checker.score_response(response, result, tolerance_pct=2.0)
        assert not score.passed

    def test_within_tolerance_passes(self):
        result = self.checker.compute_new_regime(1_000_000)
        close_val = result.tax_payable * 1.015  # 1.5% off
        response = f"Approximately ₹{close_val:,.0f}"
        score = self.checker.score_response(response, result, tolerance_pct=2.0)
        assert score.passed


# ─── SIPMathChecker ───────────────────────────────────────────────────────────


class TestSIPMathChecker:
    def setup_method(self):
        self.checker = SIPMathChecker()

    def test_sip_maturity_basic(self):
        # ₹5,000/month at 12% for 10 years → ~₹11.62L
        result = self.checker.compute_sip_maturity(5_000, 0.12, 10)
        expected = 1_161_695
        pct_err = abs(result - expected) / expected * 100
        assert pct_err <= 2.0, f"Expected ~{expected:,.0f}, got {result:,.0f} ({pct_err:.1f}% error)"

    def test_sip_higher_amount_longer_term(self):
        # ₹10,000/month at 15% for 20 years → ~₹1.51 Cr
        result = self.checker.compute_sip_maturity(10_000, 0.15, 20)
        expected = 15_079_000
        pct_err = abs(result - expected) / expected * 100
        assert pct_err <= 2.0, f"Expected ~{expected:,.0f}, got {result:,.0f} ({pct_err:.1f}% error)"

    def test_lumpsum_fv(self):
        # ₹5,00,000 lumpsum at 12% for 15 years → ~₹27.35L
        result = self.checker.compute_lumpsum(500_000, 0.12, 15)
        expected = 2_735_083
        pct_err = abs(result - expected) / expected * 100
        assert pct_err <= 2.0, f"Expected ~{expected:,.0f}, got {result:,.0f} ({pct_err:.1f}% error)"

    def test_reverse_sip_calculation(self):
        # Want ₹5 Cr in 12 years at 12% → monthly SIP ≈ ₹16,863
        result = self.checker.compute_reverse_sip(50_000_000, 0.12, 12)
        expected = 16_863
        pct_err = abs(result - expected) / expected * 100
        assert pct_err <= 2.0, f"Expected ~{expected:,.0f}/month, got {result:,.0f} ({pct_err:.1f}% error)"

    def test_ppf_approximation(self):
        # PPF: ₹12,500/month at 7.1% for 15 years → ~₹40.68L
        result = self.checker.compute_sip_maturity(12_500, 0.071, 15)
        expected = 4_068_000
        pct_err = abs(result - expected) / expected * 100
        assert pct_err <= 2.0, f"PPF: Expected ~{expected:,.0f}, got {result:,.0f} ({pct_err:.1f}% error)"

    def test_score_response_finds_amount(self):
        result_val = self.checker.compute_sip_maturity(5_000, 0.12, 10)
        response = f"Your SIP will grow to ₹{result_val:,.0f} after 10 years"
        score = self.checker.score_response(response, result_val, tolerance_pct=2.0)
        assert score.passed

    def test_higher_rate_higher_corpus(self):
        low_rate = self.checker.compute_sip_maturity(5_000, 0.10, 10)
        high_rate = self.checker.compute_sip_maturity(5_000, 0.15, 10)
        assert high_rate > low_rate


# ─── EMIMathChecker ───────────────────────────────────────────────────────────


class TestEMIMathChecker:
    def setup_method(self):
        self.checker = EMIMathChecker()

    def test_home_loan_emi(self):
        # ₹20L at 8.5% for 20 years → EMI ≈ ₹17,356
        result = self.checker.compute_emi(2_000_000, 0.085, 240)
        expected = 17_356
        pct_err = abs(result - expected) / expected * 100
        assert pct_err <= 1.0, f"Expected ~{expected}, got {result:.0f} ({pct_err:.2f}% error)"

    def test_personal_loan_emi(self):
        # ₹5L at 14% for 3 years → EMI ≈ ₹17,089
        result = self.checker.compute_emi(500_000, 0.14, 36)
        expected = 17_089
        pct_err = abs(result - expected) / expected * 100
        assert pct_err <= 1.0, f"Expected ~{expected}, got {result:.0f} ({pct_err:.2f}% error)"

    def test_car_loan_emi(self):
        # ₹8L at 10% for 5 years → EMI ≈ ₹17,002
        result = self.checker.compute_emi(800_000, 0.10, 60)
        expected = 17_002
        pct_err = abs(result - expected) / expected * 100
        assert pct_err <= 1.0, f"Expected ~{expected}, got {result:.0f} ({pct_err:.2f}% error)"

    def test_education_loan_emi(self):
        # ₹10L at 11% for 7 years → EMI ≈ ₹17,058
        result = self.checker.compute_emi(1_000_000, 0.11, 84)
        expected = 17_058
        pct_err = abs(result - expected) / expected * 100
        assert pct_err <= 1.0, f"Expected ~{expected}, got {result:.0f} ({pct_err:.2f}% error)"

    def test_gold_loan_short_tenure(self):
        # ₹2L at 12% for 1 year → EMI ≈ ₹17,747
        result = self.checker.compute_emi(200_000, 0.12, 12)
        expected = 17_747
        pct_err = abs(result - expected) / expected * 100
        assert pct_err <= 1.0, f"Expected ~{expected}, got {result:.0f} ({pct_err:.2f}% error)"

    def test_total_interest_exceeds_principal(self):
        # Long tenure loans always pay more than principal in total interest
        emi = self.checker.compute_emi(2_000_000, 0.085, 240)
        total_interest = self.checker.compute_total_interest(2_000_000, 0.085, 240)
        assert total_interest > 0
        assert total_interest < 2_000_000 * 10  # sanity: not absurdly high

    def test_higher_rate_higher_emi(self):
        emi_low = self.checker.compute_emi(1_000_000, 0.08, 120)
        emi_high = self.checker.compute_emi(1_000_000, 0.12, 120)
        assert emi_high > emi_low

    def test_score_response_finds_emi(self):
        emi = self.checker.compute_emi(2_000_000, 0.085, 240)
        response = f"Your monthly EMI will be ₹{emi:,.0f}"
        score = self.checker.score_response(response, emi, tolerance_pct=1.0)
        assert score.passed
