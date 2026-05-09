"""BFCL-style function-calling evaluation harness.

Evaluates four BFCL categories against eval/datasets/tool_call_cases.json:
1. Single-call: correct tool + args
2. No-call (relevance detection): correctly NOT calling a tool
3. Multi-call: multiple tool calls needed
4. Wrong-tool detection: agent calls wrong tool

See: Berkeley Function Calling Leaderboard v4
     gorilla.cs.berkeley.edu/leaderboard.html
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

from eval.metrics.agent_metrics import ToolCallAccuracyChecker

if TYPE_CHECKING:
    from eval.model_adapter import ModelAdapter


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class BFCLCaseResult:
    """Result for a single BFCL test case."""

    case_id: str
    category: str
    tool_category: str
    tool_selection_correct: bool
    args_correct: bool
    value_correct: Optional[bool]  # None if no expected_value
    overall_passed: bool
    predicted_tool: Optional[str]
    expected_tool: Optional[str]
    notes: str = ""


@dataclass
class BFCLResult:
    """Aggregate result for a BFCL evaluation run."""

    category: str
    total_cases: int
    passed_cases: int
    tool_selection_accuracy: float
    args_accuracy: float
    overall_accuracy: float
    per_case: list[BFCLCaseResult] = field(default_factory=list)


# ─── BFCL Harness ─────────────────────────────────────────────────────────────


class BFCLHarness:
    """Evaluates tool/function-calling accuracy across four BFCL categories.

    Loads test cases from eval/datasets/tool_call_cases.json.
    Uses ToolCallAccuracyChecker for AST-level argument comparison.
    """

    _DATASET_PATH = Path(__file__).parents[2] / "eval" / "datasets" / "tool_call_cases.json"

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        checker: Optional[ToolCallAccuracyChecker] = None,
    ) -> None:
        self._dataset_path = dataset_path or self._DATASET_PATH
        self._checker = checker or ToolCallAccuracyChecker()
        self._cases: list[dict] = []
        self._tools: dict = {}

    def load(self) -> None:
        """Load test cases from JSON file."""
        with open(self._dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        self._cases = data.get("cases", [])
        self._tools = data.get("tools", {})
        logger.info("Loaded {} BFCL test cases from {}", len(self._cases), self._dataset_path)

    def run_all(self, adapter: "ModelAdapter") -> dict[str, BFCLResult]:
        """Run all four BFCL evaluation categories.

        Args:
            adapter: Loaded ModelAdapter.

        Returns:
            Dict mapping category name to BFCLResult.
        """
        if not self._cases:
            self.load()

        results = {
            "single_call": self.run_single_call(adapter),
            "no_call": self.run_relevance_detection(adapter),
            "multi_call": self.run_multi_call(adapter),
            "wrong_tool": self.run_wrong_tool_detection(adapter),
        }

        overall_acc = sum(r.overall_accuracy for r in results.values()) / len(results)
        logger.info("BFCL overall accuracy: {:.1f}%", overall_acc * 100)
        return results

    def run_single_call(self, adapter: "ModelAdapter") -> BFCLResult:
        """Evaluate single tool-call cases."""
        cases = [c for c in self._cases if c["category"] == "single_call"]
        return self._evaluate_category("single_call", cases, adapter)

    def run_relevance_detection(self, adapter: "ModelAdapter") -> BFCLResult:
        """Evaluate no-call (relevance detection) cases.

        Tests whether the agent correctly identifies when NOT to call a tool.
        """
        cases = [c for c in self._cases if c["category"] == "no_call"]
        return self._evaluate_category("no_call", cases, adapter)

    def run_multi_call(self, adapter: "ModelAdapter") -> BFCLResult:
        """Evaluate multi-tool-call cases."""
        cases = [c for c in self._cases if c["category"] == "multi_call"]
        return self._evaluate_category("multi_call", cases, adapter)

    def run_wrong_tool_detection(self, adapter: "ModelAdapter") -> BFCLResult:
        """Evaluate wrong-tool-selection cases.

        Tests whether the agent selects the correct tool even when a plausible
        but wrong tool might be called.
        """
        cases = [c for c in self._cases if c["category"] == "wrong_tool"]
        return self._evaluate_category("wrong_tool", cases, adapter)

    def _evaluate_category(
        self,
        category: str,
        cases: list[dict],
        adapter: "ModelAdapter",
    ) -> BFCLResult:
        """Run evaluation for a set of cases within a category.

        Args:
            category: Category name for logging.
            cases: Test cases to evaluate.
            adapter: Loaded ModelAdapter.

        Returns:
            BFCLResult for this category.
        """
        if not cases:
            logger.info("No cases for BFCL category '{}' — skipping", category)
            return BFCLResult(
                category=category,
                total_cases=0,
                passed_cases=0,
                tool_selection_accuracy=0.0,
                args_accuracy=0.0,
                overall_accuracy=0.0,
            )

        per_case: list[BFCLCaseResult] = []
        tool_sel_correct = 0
        args_correct_count = 0

        for case in cases:
            result = self._evaluate_case(case, adapter)
            per_case.append(result)
            if result.tool_selection_correct:
                tool_sel_correct += 1
            if result.args_correct:
                args_correct_count += 1

        n = len(cases)
        passed = sum(r.overall_passed for r in per_case)

        bfcl = BFCLResult(
            category=category,
            total_cases=n,
            passed_cases=passed,
            tool_selection_accuracy=round(tool_sel_correct / n, 3),
            args_accuracy=round(args_correct_count / n, 3),
            overall_accuracy=round(passed / n, 3),
            per_case=per_case,
        )
        logger.info("BFCL {}: {}/{} passed ({:.1f}%)", category, passed, n, bfcl.overall_accuracy * 100)
        return bfcl

    def _evaluate_case(self, case: dict, adapter: "ModelAdapter") -> BFCLCaseResult:
        """Evaluate a single BFCL case.

        Args:
            case: Test case dict from JSON.
            adapter: Loaded ModelAdapter.

        Returns:
            BFCLCaseResult.
        """
        result = adapter.generate(case["query"])
        predicted_tool, predicted_args = self._checker.extract_tool_call_from_response(result.response)

        expected_tool = case.get("expected_tool")
        expected_args = case.get("expected_args", {})

        # Tool selection check
        tool_sel = self._checker.check_tool_selection(predicted_tool, expected_tool)

        # Arg check (only if tool selection correct and args are provided)
        args_ok = False
        if tool_sel and expected_tool is not None and predicted_args:
            arg_result = self._checker.check_args(
                predicted_args or {}, expected_args or {}
            )
            args_ok = arg_result.args_correct
        elif expected_tool is None and predicted_tool is None:
            # Correct no-call
            args_ok = True

        # Value check (for deterministic cases)
        value_ok = None
        expected_value = case.get("expected_value")
        if expected_value is not None and result.response:
            tolerance = case.get("tolerance_pct", 2.0) / 100
            value_ok = self._check_value_in_response(result.response, expected_value, tolerance)

        overall = tool_sel and args_ok and (value_ok is None or value_ok)

        return BFCLCaseResult(
            case_id=case["id"],
            category=case["category"],
            tool_category=case.get("tool_category", ""),
            tool_selection_correct=tool_sel,
            args_correct=args_ok,
            value_correct=value_ok,
            overall_passed=overall,
            predicted_tool=predicted_tool,
            expected_tool=expected_tool,
            notes=case.get("notes", ""),
        )

    @staticmethod
    def _check_value_in_response(response: str, expected: float, tolerance: float) -> bool:
        """Quick check if the expected numeric value appears (approximately) in the response."""
        import re
        # Find all numbers in the response
        numbers = re.findall(r"([\d,]+(?:\.\d+)?)\s*(?:lakh|lakhs|L|crore|Cr)?", response, re.IGNORECASE)
        for raw in numbers:
            try:
                val = float(raw.replace(",", ""))
                # Check for lakh multiplier in surrounding context
                if abs(val - expected) / max(abs(expected), 1) <= tolerance:
                    return True
                if abs(val * 100_000 - expected) / max(abs(expected), 1) <= tolerance:
                    return True
            except ValueError:
                pass
        return False

    def aggregate(self, results: dict[str, BFCLResult]) -> float:
        """Compute overall BFCL accuracy across all categories.

        Args:
            results: Dict from run_all().

        Returns:
            Weighted mean accuracy across categories (equal weights).
        """
        accuracies = [r.overall_accuracy for r in results.values() if r.total_cases > 0]
        return round(sum(accuracies) / len(accuracies), 3) if accuracies else 0.0
