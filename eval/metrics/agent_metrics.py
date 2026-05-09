"""Agent-level evaluation metrics — tool-call accuracy, pass@k, trajectory scoring.

Implements BFCL-style AST checks: verifies tool selection and argument correctness
without executing any tool calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ArgCheckResult:
    """Result of comparing predicted vs expected tool arguments."""

    tool_selection_correct: bool
    args_correct: bool
    missing_args: list[str]
    wrong_args: dict[str, dict]  # arg_name -> {expected, predicted}
    extra_args: list[str]
    score: float  # 0.0–1.0


@dataclass
class ToolCall:
    """A single tool call in an agent trajectory."""

    tool_name: str
    args: dict[str, Any]
    result: Optional[Any] = None
    step_index: int = 0


@dataclass
class RedundancyResult:
    """Result of redundant-call analysis on a trajectory."""

    total_calls: int
    redundant_calls: int
    redundant_rate: float  # calls per task (lower is better)
    redundant_pairs: list[dict]  # list of (step_i, step_j) pairs


@dataclass
class TrajectoryStep:
    """One reasoning step in a ReAct trajectory."""

    thought: str
    action: Optional[str]
    action_input: Optional[dict]
    observation: Optional[str]
    step_index: int


@dataclass
class TrajectoryScore:
    """Aggregate score over a full agent trajectory."""

    task_completed: bool
    tool_call_accuracy: float
    redundant_call_rate: float
    recovery_from_error: bool
    total_steps: int
    correct_steps: int
    step_score: float  # correct_steps / total_steps


# ─── Tool Call Accuracy ────────────────────────────────────────────────────────


class ToolCallAccuracyChecker:
    """BFCL-style evaluation of tool selection and argument correctness.

    Checks:
    1. Tool selection: Is the correct tool called?
    2. Argument correctness: Are required args present with correct values?
    3. Relevance detection: Does agent correctly NOT call a tool when not needed?
    """

    _NUMERIC_TOLERANCE = 0.01  # 1% default tolerance for numeric arg comparison

    def check_tool_selection(self, predicted: Optional[str], expected: Optional[str]) -> bool:
        """Check if the correct tool was selected.

        Args:
            predicted: Tool name the agent called (None if no call).
            expected: Expected tool name (None if no call expected).

        Returns:
            True if selection is correct (including correct no-call).
        """
        if expected is None and predicted is None:
            return True  # correct no-call (relevance detection)
        if expected is None and predicted is not None:
            logger.debug("tool_selection: spurious call to {}", predicted)
            return False  # unnecessary tool call
        if expected is not None and predicted is None:
            logger.debug("tool_selection: missed required call to {}", expected)
            return False  # missed required tool call
        match = predicted == expected
        if not match:
            logger.debug("tool_selection: expected={} got={}", expected, predicted)
        return match

    def check_args(
        self,
        predicted: dict[str, Any],
        expected: dict[str, Any],
        tolerance: float = _NUMERIC_TOLERANCE,
    ) -> ArgCheckResult:
        """Check argument correctness via AST-style comparison.

        Args:
            predicted: Args the agent supplied.
            expected: Ground-truth args.
            tolerance: Fractional tolerance for numeric comparisons.

        Returns:
            ArgCheckResult with per-arg breakdown.
        """
        missing = [k for k in expected if k not in predicted]
        extra = [k for k in predicted if k not in expected]
        wrong: dict[str, dict] = {}

        for key, exp_val in expected.items():
            if key not in predicted:
                continue
            pred_val = predicted[key]
            if not self._values_match(pred_val, exp_val, tolerance):
                wrong[key] = {"expected": exp_val, "predicted": pred_val}

        total_required = len(expected)
        correct = total_required - len(missing) - len(wrong)
        score = correct / total_required if total_required > 0 else 1.0

        args_correct = len(missing) == 0 and len(wrong) == 0

        return ArgCheckResult(
            tool_selection_correct=True,  # caller sets this
            args_correct=args_correct,
            missing_args=missing,
            wrong_args=wrong,
            extra_args=extra,
            score=round(score, 3),
        )

    def extract_tool_call_from_response(
        self, response: str
    ) -> tuple[Optional[str], Optional[dict]]:
        """Extract tool name and args from a ReAct-formatted response.

        Parses patterns like:
            Action: tax_calculator
            Action Input: {"annual_income": 1000000, "regime": "new"}

        Also handles JSON function-call format:
            {"name": "tax_calculator", "arguments": {...}}

        Args:
            response: Raw model response text.

        Returns:
            (tool_name, args_dict) or (None, None) if no call found.
        """
        # ReAct format
        action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", response, re.IGNORECASE | re.DOTALL)
        if action_match and input_match:
            tool_name = action_match.group(1).strip()
            try:
                args = json.loads(input_match.group(1))
                return tool_name, args
            except json.JSONDecodeError:
                logger.warning("Could not parse Action Input JSON")

        # JSON function-call format
        json_match = re.search(r'\{"name":\s*"(\w+)".*?"arguments":\s*(\{.*?\})\}', response, re.DOTALL)
        if json_match:
            tool_name = json_match.group(1)
            try:
                args = json.loads(json_match.group(2))
                return tool_name, args
            except json.JSONDecodeError:
                pass

        # No tool call detected
        return None, None

    def compute_pass_at_k(self, run_results: list[bool], k: int) -> float:
        """Compute pass@k — probability that at least one of k runs passes.

        Args:
            run_results: List of bool pass/fail results for repeated runs.
            k: Number of attempts per problem.

        Returns:
            pass@k as float 0.0–1.0.
        """
        if not run_results:
            return 0.0
        n = len(run_results)
        passes = sum(run_results)
        if k >= n:
            return 1.0 if passes > 0 else 0.0
        # Exact formula: 1 - C(n-passes, k) / C(n, k)
        from math import comb
        if n - passes < k:
            return 1.0
        return 1.0 - comb(n - passes, k) / comb(n, k)

    @staticmethod
    def _values_match(pred: Any, expected: Any, tolerance: float) -> bool:
        """Compare two argument values with tolerance for numeric types."""
        if isinstance(expected, (int, float)) and isinstance(pred, (int, float)):
            if expected == 0:
                return abs(pred) <= tolerance
            return abs(pred - expected) / abs(expected) <= tolerance
        if isinstance(expected, str) and isinstance(pred, str):
            return pred.strip().lower() == expected.strip().lower()
        if isinstance(expected, bool):
            return pred == expected
        return pred == expected


# ─── Redundant Call Detector ──────────────────────────────────────────────────


class RedundantCallDetector:
    """Detects unnecessary repeated tool calls in an agent trajectory.

    A call is redundant if:
    - Same tool called with identical (or near-identical) args within the same task.
    - A tool is called after its result was already available.

    Target: < 0.5 redundant calls per task (from eval/research/summary.md).
    """

    def analyze(self, tool_calls: list[ToolCall]) -> RedundancyResult:
        """Analyze a sequence of tool calls for redundancy.

        Args:
            tool_calls: Ordered list of tool calls in the trajectory.

        Returns:
            RedundancyResult with redundant call count and pairs.
        """
        seen: dict[str, dict] = {}  # tool_name -> args_hash -> step_index
        redundant_pairs: list[dict] = []

        for call in tool_calls:
            key = call.tool_name
            args_hash = self._hash_args(call.args)
            full_key = f"{key}::{args_hash}"

            if full_key in seen:
                redundant_pairs.append({
                    "tool": call.tool_name,
                    "args": call.args,
                    "first_call_step": seen[full_key],
                    "redundant_call_step": call.step_index,
                })
            else:
                seen[full_key] = call.step_index

        return RedundancyResult(
            total_calls=len(tool_calls),
            redundant_calls=len(redundant_pairs),
            redundant_rate=len(redundant_pairs),  # per task
            redundant_pairs=redundant_pairs,
        )

    @staticmethod
    def _hash_args(args: dict) -> str:
        """Create a stable string hash of tool arguments."""
        try:
            return json.dumps(args, sort_keys=True)
        except (TypeError, ValueError):
            return str(sorted(args.items()))


# ─── Trajectory Scorer ────────────────────────────────────────────────────────


class TrajectoryScorer:
    """Aggregates per-step evaluation results into a trajectory-level score.

    Evaluates the full ReAct trace rather than just the final answer —
    critical per eval/research/summary.md §Core Evaluation Philosophy:
    'Trajectory-level eval is non-negotiable.'
    """

    def __init__(
        self,
        tool_checker: Optional[ToolCallAccuracyChecker] = None,
        redundancy_detector: Optional[RedundantCallDetector] = None,
    ) -> None:
        self._tool_checker = tool_checker or ToolCallAccuracyChecker()
        self._redundancy = redundancy_detector or RedundantCallDetector()

    def score(
        self,
        trajectory: list[TrajectoryStep],
        expected_tool_calls: Optional[list[dict]] = None,
        task_completed: bool = False,
    ) -> TrajectoryScore:
        """Score a complete agent trajectory.

        Args:
            trajectory: Ordered list of ReAct steps.
            expected_tool_calls: Optional list of expected tool calls for accuracy scoring.
            task_completed: Whether the task was ultimately completed successfully.

        Returns:
            TrajectoryScore with all trajectory-level metrics.
        """
        tool_calls = [
            ToolCall(
                tool_name=step.action or "",
                args=step.action_input or {},
                step_index=step.step_index,
            )
            for step in trajectory
            if step.action is not None
        ]

        redundancy = self._redundancy.analyze(tool_calls)

        # Compute tool-call accuracy if ground truth provided
        tool_accuracy = 1.0
        if expected_tool_calls:
            correct = 0
            for i, exp in enumerate(expected_tool_calls):
                if i >= len(tool_calls):
                    break
                pred = tool_calls[i]
                if self._tool_checker.check_tool_selection(pred.tool_name, exp.get("tool")):
                    arg_result = self._tool_checker.check_args(pred.args, exp.get("args", {}))
                    if arg_result.args_correct:
                        correct += 1
            tool_accuracy = correct / len(expected_tool_calls) if expected_tool_calls else 1.0

        # Check recovery from error: did agent continue after an observation indicating error?
        recovery = self._check_recovery(trajectory)

        correct_steps = sum(
            1 for step in trajectory
            if step.action is not None and step.observation is not None
            and "error" not in (step.observation or "").lower()
        )

        return TrajectoryScore(
            task_completed=task_completed,
            tool_call_accuracy=round(tool_accuracy, 3),
            redundant_call_rate=redundancy.redundant_rate,
            recovery_from_error=recovery,
            total_steps=len(trajectory),
            correct_steps=correct_steps,
            step_score=round(correct_steps / max(len(trajectory), 1), 3),
        )

    @staticmethod
    def _check_recovery(trajectory: list[TrajectoryStep]) -> bool:
        """Return True if the agent recovered from at least one error observation."""
        had_error = False
        recovered = False
        for step in trajectory:
            obs = (step.observation or "").lower()
            if any(word in obs for word in ("error", "failed", "exception", "not found")):
                had_error = True
            elif had_error and step.action is not None:
                recovered = True
                had_error = False
        return recovered
