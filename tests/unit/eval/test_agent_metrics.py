"""Unit tests for eval/metrics/agent_metrics.py.

Tests tool-call accuracy, pass@k, and redundant-call detection — no model needed.
"""

from __future__ import annotations

import math
import pytest
from loguru import logger

from eval.metrics.agent_metrics import (
    RedundantCallDetector,
    ToolCallAccuracyChecker,
    TrajectoryScorer,
)


# ─── ToolCallAccuracyChecker ─────────────────────────────────────────────────


class TestToolSelection:
    def setup_method(self):
        self.checker = ToolCallAccuracyChecker()

    def test_correct_tool_passes(self):
        assert self.checker.check_tool_selection("sip_calculator", "sip_calculator") is True

    def test_wrong_tool_fails(self):
        assert self.checker.check_tool_selection("tax_calculator", "sip_calculator") is False

    def test_case_insensitive_match(self):
        assert self.checker.check_tool_selection("SIP_Calculator", "sip_calculator") is True


class TestArgCheck:
    def setup_method(self):
        self.checker = ToolCallAccuracyChecker()

    def test_exact_numeric_args_pass(self):
        predicted = {"monthly_amount": 5000, "annual_rate": 0.12, "years": 10}
        expected = {"monthly_amount": 5000, "annual_rate": 0.12, "years": 10}
        result = self.checker.check_args(predicted, expected)
        assert result.passed

    def test_within_tolerance_passes(self):
        # 0.1199 vs 0.12 → 0.08% error, well within default 1%
        predicted = {"monthly_amount": 5000, "annual_rate": 0.1199, "years": 10}
        expected = {"monthly_amount": 5000, "annual_rate": 0.12, "years": 10}
        result = self.checker.check_args(predicted, expected, tolerance=0.01)
        assert result.passed

    def test_outside_tolerance_fails(self):
        predicted = {"monthly_amount": 5000, "annual_rate": 0.15, "years": 10}
        expected = {"monthly_amount": 5000, "annual_rate": 0.12, "years": 10}
        result = self.checker.check_args(predicted, expected, tolerance=0.01)
        assert not result.passed

    def test_missing_required_arg_fails(self):
        predicted = {"monthly_amount": 5000}
        expected = {"monthly_amount": 5000, "annual_rate": 0.12, "years": 10}
        result = self.checker.check_args(predicted, expected)
        assert not result.passed

    def test_string_arg_exact_match(self):
        predicted = {"regime": "new", "annual_income": 1_000_000}
        expected = {"regime": "new", "annual_income": 1_000_000}
        result = self.checker.check_args(predicted, expected)
        assert result.passed

    def test_string_arg_mismatch_fails(self):
        predicted = {"regime": "old", "annual_income": 1_000_000}
        expected = {"regime": "new", "annual_income": 1_000_000}
        result = self.checker.check_args(predicted, expected)
        assert not result.passed

    def test_extra_args_ok(self):
        # Extra predicted args should not penalize the check
        predicted = {"monthly_amount": 5000, "annual_rate": 0.12, "years": 10, "extra_field": True}
        expected = {"monthly_amount": 5000, "annual_rate": 0.12, "years": 10}
        result = self.checker.check_args(predicted, expected)
        assert result.passed


class TestExtractToolCall:
    def setup_method(self):
        self.checker = ToolCallAccuracyChecker()

    def test_react_format_extraction(self):
        response = """Thought: I need to calculate the SIP returns.
Action: sip_calculator
Action Input: {"monthly_amount": 5000, "annual_rate": 0.12, "years": 10}
"""
        tc = self.checker.extract_tool_call_from_response(response)
        assert tc is not None
        assert tc["tool"] == "sip_calculator"
        assert tc["args"]["monthly_amount"] == 5000

    def test_json_format_extraction(self):
        response = """{"tool": "tax_calculator", "args": {"annual_income": 1000000, "regime": "new"}}"""
        tc = self.checker.extract_tool_call_from_response(response)
        assert tc is not None
        assert tc["tool"] == "tax_calculator"

    def test_no_tool_call_returns_none(self):
        response = "I don't need any tools for this question. The answer is simply 42."
        tc = self.checker.extract_tool_call_from_response(response)
        assert tc is None

    def test_function_call_format(self):
        response = 'tax_calculator({"annual_income": 700000, "regime": "new"})'
        tc = self.checker.extract_tool_call_from_response(response)
        assert tc is not None
        assert tc["tool"] == "tax_calculator"


class TestPassAtK:
    def setup_method(self):
        self.checker = ToolCallAccuracyChecker()

    def test_all_pass_returns_1(self):
        results = [True, True, True, True, True]
        assert self.checker.compute_pass_at_k(results, k=1) == 1.0

    def test_all_fail_returns_0(self):
        results = [False, False, False, False, False]
        assert self.checker.compute_pass_at_k(results, k=1) == 0.0

    def test_pass_at_1_simple(self):
        # 3 passes out of 5 → pass@1 = 3/5 = 0.6
        results = [True, True, False, True, False]
        assert abs(self.checker.compute_pass_at_k(results, k=1) - 0.6) < 1e-6

    def test_pass_at_k_is_nondecreasing(self):
        # More attempts can only help — pass@k ≥ pass@(k-1)
        results = [True, False, False, False, False]
        p1 = self.checker.compute_pass_at_k(results, k=1)
        p3 = self.checker.compute_pass_at_k(results, k=3)
        p5 = self.checker.compute_pass_at_k(results, k=5)
        assert p1 <= p3 <= p5

    def test_pass_at_k_single_success(self):
        # 1 success in 5 runs: pass@5 should be 1.0 (at least one pass in any 5 draws)
        results = [True, False, False, False, False]
        assert self.checker.compute_pass_at_k(results, k=5) == 1.0

    def test_k_larger_than_n_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            self.checker.compute_pass_at_k([True, False], k=5)


# ─── RedundantCallDetector ───────────────────────────────────────────────────


class TestRedundantCallDetector:
    def setup_method(self):
        self.detector = RedundantCallDetector()

    def _make_calls(self, items: list[tuple[str, dict]]):
        """Build minimal ToolCall-like dicts for the detector."""
        class _TC:
            def __init__(self, tool, args):
                self.tool_name = tool
                self.args = args
        return [_TC(t, a) for t, a in items]

    def test_no_redundancy(self):
        calls = self._make_calls([
            ("sip_calculator", {"monthly": 5000, "rate": 0.12, "years": 10}),
            ("tax_calculator", {"income": 1_000_000, "regime": "new"}),
        ])
        result = self.detector.analyze(calls)
        assert result.redundant_count == 0
        assert result.redundant_rate == 0.0

    def test_duplicate_call_detected(self):
        calls = self._make_calls([
            ("sip_calculator", {"monthly": 5000, "rate": 0.12, "years": 10}),
            ("sip_calculator", {"monthly": 5000, "rate": 0.12, "years": 10}),
        ])
        result = self.detector.analyze(calls)
        assert result.redundant_count >= 1

    def test_different_args_not_redundant(self):
        calls = self._make_calls([
            ("sip_calculator", {"monthly": 5000, "rate": 0.12, "years": 10}),
            ("sip_calculator", {"monthly": 10_000, "rate": 0.12, "years": 10}),
        ])
        result = self.detector.analyze(calls)
        assert result.redundant_count == 0

    def test_empty_calls(self):
        result = self.detector.analyze([])
        assert result.redundant_count == 0
        assert result.redundant_rate == 0.0


# ─── TrajectoryScorer ────────────────────────────────────────────────────────


class TestTrajectoryScorer:
    def setup_method(self):
        self.scorer = TrajectoryScorer()

    def _make_step(self, tool_correct: bool, args_correct: bool, response_ok: bool = True):
        class _Step:
            pass
        s = _Step()
        s.tool_correct = tool_correct
        s.args_correct = args_correct
        s.response_ok = response_ok
        s.error = None
        return s

    def test_perfect_trajectory_passes(self):
        steps = [self._make_step(True, True, True) for _ in range(3)]
        score = self.scorer.score(steps)
        assert score.overall_score >= 0.9

    def test_all_fail_scores_zero(self):
        steps = [self._make_step(False, False, False) for _ in range(3)]
        score = self.scorer.score(steps)
        assert score.overall_score <= 0.1

    def test_partial_trajectory(self):
        steps = [
            self._make_step(True, True, True),
            self._make_step(False, False, True),
            self._make_step(True, True, True),
        ]
        score = self.scorer.score(steps)
        assert 0.3 < score.overall_score < 0.9

    def test_empty_trajectory(self):
        score = self.scorer.score([])
        assert score.overall_score == 0.0
