"""Unit tests for the synthetic data generation pipeline."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.data.synthetic.constants import (
    INCOME_BRACKETS,
    NEW_REGIME_87A_REBATE_LIMIT,
    NEW_REGIME_SLABS,
    OLD_REGIME_SLABS,
)
from src.data.synthetic.templates.tax_templates import (
    TaxTemplates,
    compute_new_regime_tax,
    compute_old_regime_tax,
)
from src.data.synthetic.templates.investment_templates import (
    InvestmentTemplates,
    calc_sip_maturity,
    calc_lumpsum_maturity,
)
from src.data.synthetic.templates.loan_templates import LoanTemplates, calc_emi
from src.data.synthetic.templates.insurance_templates import InsuranceTemplates
from src.data.synthetic.generator import SyntheticDataGenerator
from src.data.synthetic.validators import DataQualityValidator


# ── Tax calculation correctness ───────────────────────────────────────────────


class TestTaxCalculations:
    def test_new_regime_zero_below_rebate(self) -> None:
        """Income ≤ rebate threshold should result in zero tax (Section 87A)."""
        result = compute_new_regime_tax(700_000)
        assert result["total_tax"] == 0, "Taxable income ≤ ₹7L should attract zero tax in new regime"

    def test_new_regime_above_rebate_has_tax(self) -> None:
        result = compute_new_regime_tax(800_000)
        assert result["total_tax"] > 0

    def test_new_regime_standard_deduction_applied(self) -> None:
        result = compute_new_regime_tax(1_000_000)
        assert result["taxable_income"] == 1_000_000 - 75_000

    def test_old_regime_basic_10l(self) -> None:
        """10L income with no deductions: taxable = 9.5L, tax in old regime."""
        result = compute_old_regime_tax(1_000_000)
        # taxable = 1000000 - 50000 = 950000
        # tax: 0 + 12500 + (950000-500000)*0.20 = 12500 + 90000 = 102500
        # cess: 102500 * 0.04 = 4100
        # total: 106600
        assert result["taxable_income"] == 950_000
        assert result["tax_before_cess"] == 102_500
        assert result["total_tax"] == 106_600

    def test_old_regime_with_max_80c_deduction(self) -> None:
        no_deduct = compute_old_regime_tax(1_000_000)
        with_deduct = compute_old_regime_tax(1_000_000, deduction_80c=150_000)
        assert with_deduct["total_tax"] < no_deduct["total_tax"]

    def test_old_regime_80c_capped_at_limit(self) -> None:
        result1 = compute_old_regime_tax(1_000_000, deduction_80c=150_000)
        result2 = compute_old_regime_tax(1_000_000, deduction_80c=200_000)
        assert result1["total_tax"] == result2["total_tax"]

    def test_effective_rate_never_exceeds_30(self) -> None:
        for income in INCOME_BRACKETS:
            result = compute_new_regime_tax(income)
            assert result["effective_rate"] <= 30.0, f"Effective rate {result['effective_rate']} > 30% for income {income}"

    def test_effective_rate_positive_for_high_income(self) -> None:
        result = compute_new_regime_tax(5_000_000)
        assert result["effective_rate"] > 0


# ── SIP / Investment calculations ─────────────────────────────────────────────


class TestInvestmentCalculations:
    def test_sip_maturity_exceeds_invested(self) -> None:
        res = calc_sip_maturity(5_000, 0.12, 10)
        assert res["maturity_amount"] > res["total_invested"]

    def test_sip_total_invested_correct(self) -> None:
        res = calc_sip_maturity(5_000, 0.12, 10)
        assert res["total_invested"] == 5_000 * 10 * 12

    def test_lumpsum_grows_with_positive_rate(self) -> None:
        res = calc_lumpsum_maturity(100_000, 0.12, 10)
        assert res["maturity_amount"] > 100_000

    def test_sip_zero_rate_returns_principal(self) -> None:
        res = calc_sip_maturity(5_000, 0.0, 5)
        assert abs(res["maturity_amount"] - res["total_invested"]) < 1

    def test_longer_sip_gives_more(self) -> None:
        res10 = calc_sip_maturity(5_000, 0.12, 10)
        res20 = calc_sip_maturity(5_000, 0.12, 20)
        assert res20["maturity_amount"] > res10["maturity_amount"]


# ── Loan (EMI) calculations ───────────────────────────────────────────────────


class TestLoanCalculations:
    def test_total_payment_exceeds_principal(self) -> None:
        res = calc_emi(1_000_000, 0.085, 240)
        assert res["total_payment"] > 1_000_000

    def test_total_interest_positive(self) -> None:
        res = calc_emi(500_000, 0.12, 36)
        assert res["total_interest"] > 0

    def test_zero_rate_emi_equals_principal_divided_by_months(self) -> None:
        res = calc_emi(120_000, 0.0, 12)
        assert abs(res["emi"] - 10_000) < 1

    def test_higher_rate_means_more_interest(self) -> None:
        low = calc_emi(1_000_000, 0.08, 240)
        high = calc_emi(1_000_000, 0.12, 240)
        assert high["total_interest"] > low["total_interest"]


# ── Template sample generation ────────────────────────────────────────────────


class TestTemplates:
    _RNG = random.Random(42)

    def test_tax_template_returns_qa(self) -> None:
        tmpl = TaxTemplates(self._RNG)
        sample = tmpl.generate_sample()
        assert "question" in sample
        assert "answer" in sample
        assert len(sample["question"]) > 10
        assert len(sample["answer"]) > 50

    def test_investment_template_returns_qa(self) -> None:
        tmpl = InvestmentTemplates(self._RNG)
        sample = tmpl.generate_sample()
        assert "question" in sample and "answer" in sample

    def test_loan_template_returns_qa(self) -> None:
        tmpl = LoanTemplates(self._RNG)
        sample = tmpl.generate_sample()
        assert "question" in sample and "answer" in sample

    def test_insurance_template_returns_qa(self) -> None:
        tmpl = InsuranceTemplates(self._RNG)
        sample = tmpl.generate_sample()
        assert "question" in sample and "answer" in sample


# ── DataQualityValidator ──────────────────────────────────────────────────────


class TestDataQualityValidator:
    _VALID_SAMPLE = {
        "messages": [
            {"role": "system", "content": "You are FinEdge."},
            {"role": "user", "content": "What is the best tax regime?"},
            {"role": "assistant", "content": "A" * 100},
        ]
    }

    def test_valid_sample_passes(self) -> None:
        validator = DataQualityValidator(min_answer_length=50)
        errors = validator.validate(self._VALID_SAMPLE)
        assert errors == []

    def test_missing_messages_key(self) -> None:
        validator = DataQualityValidator()
        errors = validator.validate({"data": []})
        assert errors

    def test_wrong_number_of_messages(self) -> None:
        validator = DataQualityValidator()
        errors = validator.validate({"messages": [{"role": "user", "content": "hi"}]})
        assert errors

    def test_answer_too_short(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Short"},
            ]
        }
        validator = DataQualityValidator(min_answer_length=50)
        errors = validator.validate(sample)
        assert any("short" in e.lower() for e in errors)

    def test_empty_question_flagged(self) -> None:
        sample = {
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "  "},
                {"role": "assistant", "content": "A" * 100},
            ]
        }
        validator = DataQualityValidator(min_answer_length=50)
        errors = validator.validate(sample)
        assert errors


# ── SyntheticDataGenerator integration ───────────────────────────────────────


class TestSyntheticDataGenerator:
    def test_generate_produces_correct_count(self) -> None:
        cfg = {
            "random_seed": 42,
            "min_answer_length": 50,
            "max_answer_length": 5000,
            "topic_counts": {"tax": 5, "investment": 5},
        }
        gen = SyntheticDataGenerator(cfg)
        samples = gen.generate()
        assert len(samples) > 0
        assert len(samples) <= 10

    def test_samples_are_valid_chatml(self) -> None:
        cfg = {
            "random_seed": 42,
            "min_answer_length": 50,
            "max_answer_length": 5000,
            "topic_counts": {"tax": 3},
        }
        gen = SyntheticDataGenerator(cfg)
        samples = gen.generate()
        for s in samples:
            assert "messages" in s
            roles = {m["role"] for m in s["messages"]}
            assert roles == {"system", "user", "assistant"}

    def test_save_creates_jsonl(self, tmp_path: Path) -> None:
        cfg = {
            "random_seed": 42,
            "min_answer_length": 50,
            "max_answer_length": 5000,
            "topic_counts": {"loan": 2},
        }
        gen = SyntheticDataGenerator(cfg)
        samples = gen.generate()
        out = tmp_path / "synthetic.jsonl"
        gen.save(samples, out)
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(samples)

    def test_deterministic_with_same_seed(self) -> None:
        cfg = {
            "random_seed": 99,
            "min_answer_length": 50,
            "max_answer_length": 5000,
            "topic_counts": {"tax": 3},
        }
        gen1 = SyntheticDataGenerator(cfg)
        gen2 = SyntheticDataGenerator(cfg)
        s1 = gen1.generate()
        s2 = gen2.generate()
        assert s1 == s2
