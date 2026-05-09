"""Unit tests for eval/harnesses/bfcl_harness.py.

Uses stub data to test the harness logic without a real model.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from loguru import logger

from eval.harnesses.bfcl_harness import BFCLHarness, BFCLResult


# ─── Stub cases (no file I/O required) ───────────────────────────────────────

STUB_SINGLE_CALL_CASES = [
    {
        "id": "tc_stub_001",
        "category": "single_call",
        "tool_category": "sip_calculator",
        "query": "Calculate SIP returns for ₹5000/month at 12% for 10 years",
        "expected_tool": "sip_calculator",
        "expected_args": {"monthly_amount": 5000, "annual_rate": 0.12, "years": 10},
        "expected_value": 1_161_695,
        "tolerance_pct": 2.0,
    },
    {
        "id": "tc_stub_002",
        "category": "single_call",
        "tool_category": "tax_calculator",
        "query": "What is my tax for ₹10 lakh income under new regime?",
        "expected_tool": "tax_calculator",
        "expected_args": {"annual_income": 1_000_000, "regime": "new"},
        "expected_value": 44_200,
        "tolerance_pct": 2.0,
    },
]

STUB_NO_CALL_CASES = [
    {
        "id": "tc_stub_003",
        "category": "no_call",
        "tool_category": "none",
        "query": "What is Section 80C of the Income Tax Act?",
        "expected_tool": None,
        "expected_args": {},
        "notes": "Informational — no calculator needed",
    },
    {
        "id": "tc_stub_004",
        "category": "no_call",
        "tool_category": "none",
        "query": "Explain the difference between old and new tax regime",
        "expected_tool": None,
        "expected_args": {},
    },
]

STUB_WRONG_TOOL_CASES = [
    {
        "id": "tc_stub_005",
        "category": "single_call",
        "tool_category": "loan_advisor",
        "query": "Calculate EMI for ₹20 lakh home loan at 8.5% for 20 years",
        "expected_tool": "loan_advisor",
        "expected_args": {"principal": 2_000_000, "annual_rate": 0.085, "tenure_months": 240},
        "expected_value": 17_356,
        "tolerance_pct": 1.0,
    },
]


def _make_mock_adapter(tool_name: str, args: dict, value_response: str | None = None):
    """Build a mock adapter that always returns a specific tool call."""
    adapter = MagicMock()
    gen_result = MagicMock()
    if tool_name:
        gen_result.response = (
            f"Thought: I need to use {tool_name}.\n"
            f"Action: {tool_name}\n"
            f"Action Input: {json.dumps(args)}\n"
        )
        if value_response:
            gen_result.response += f"\nFinal Answer: {value_response}"
    else:
        gen_result.response = "This is a general explanation without any tool calls."
    adapter.generate.return_value = gen_result
    return adapter


# ─── BFCLHarness unit tests ───────────────────────────────────────────────────


class TestBFCLHarnessInit:
    def test_harness_instantiates(self):
        harness = BFCLHarness()
        assert harness is not None

    def test_harness_load_with_stub(self, tmp_path):
        # Write stub cases to a temp file and load from there
        cases_file = tmp_path / "tool_call_cases.json"
        all_cases = STUB_SINGLE_CALL_CASES + STUB_NO_CALL_CASES + STUB_WRONG_TOOL_CASES
        cases_file.write_text(json.dumps(all_cases), encoding="utf-8")
        harness = BFCLHarness()
        harness._cases_path = cases_file
        harness.load()
        assert len(harness._cases) == len(all_cases)


class TestBFCLSingleCall:
    def test_correct_tool_and_args_passes(self):
        harness = BFCLHarness()
        harness._cases = STUB_SINGLE_CALL_CASES + STUB_WRONG_TOOL_CASES

        adapter = _make_mock_adapter(
            "sip_calculator",
            {"monthly_amount": 5000, "annual_rate": 0.12, "years": 10},
        )

        result = harness.run_single_call(adapter)
        assert isinstance(result, BFCLResult)
        assert result.total_cases > 0
        assert 0.0 <= result.overall_accuracy <= 1.0

    def test_wrong_tool_reduces_accuracy(self):
        harness = BFCLHarness()
        harness._cases = STUB_SINGLE_CALL_CASES[:1]

        # Adapter returns wrong tool
        adapter = _make_mock_adapter("tax_calculator", {"annual_income": 1_000_000})

        result = harness.run_single_call(adapter)
        assert result.overall_accuracy < 1.0


class TestBFCLRelevanceDetection:
    def test_no_tool_call_for_informational_query(self):
        harness = BFCLHarness()
        harness._cases = STUB_NO_CALL_CASES

        # Adapter correctly returns no tool call
        adapter = _make_mock_adapter("", {})

        result = harness.run_relevance_detection(adapter)
        assert isinstance(result, BFCLResult)
        assert result.total_cases == len(STUB_NO_CALL_CASES)
        assert result.overall_accuracy == 1.0

    def test_spurious_tool_call_reduces_accuracy(self):
        harness = BFCLHarness()
        harness._cases = STUB_NO_CALL_CASES[:1]

        # Adapter wrongly calls a tool for an informational query
        adapter = _make_mock_adapter("sip_calculator", {"monthly_amount": 5000})

        result = harness.run_relevance_detection(adapter)
        assert result.overall_accuracy < 1.0


class TestBFCLAggregate:
    def test_aggregate_equal_weight(self):
        harness = BFCLHarness()

        r1 = BFCLResult(category="single_call", total_cases=10, passed_cases=8)
        r2 = BFCLResult(category="no_call", total_cases=10, passed_cases=6)

        agg = harness.aggregate([r1, r2])
        expected = (r1.overall_accuracy + r2.overall_accuracy) / 2
        assert abs(agg - expected) < 1e-6

    def test_aggregate_empty_returns_zero(self):
        harness = BFCLHarness()
        assert harness.aggregate([]) == 0.0

    def test_aggregate_single_result(self):
        harness = BFCLHarness()
        r = BFCLResult(category="single_call", total_cases=5, passed_cases=5)
        assert harness.aggregate([r]) == 1.0


class TestBFCLResultDataclass:
    def test_overall_accuracy_computed(self):
        r = BFCLResult(category="single_call", total_cases=10, passed_cases=7)
        assert abs(r.overall_accuracy - 0.7) < 1e-6

    def test_zero_cases_accuracy(self):
        r = BFCLResult(category="single_call", total_cases=0, passed_cases=0)
        assert r.overall_accuracy == 0.0

    def test_per_case_list_initialized(self):
        r = BFCLResult(category="no_call", total_cases=3, passed_cases=2)
        assert isinstance(r.per_case, list)
