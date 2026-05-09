"""Abstract base class for LLM-as-judge evaluation.

Loads YAML rubrics and provides a common interface for scoring model responses.
Subclasses provide the actual judge model invocation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class RubricDimension:
    """One scoring dimension within a rubric."""

    name: str
    weight: float
    criteria: str
    max_score: int = 5  # 1–5 scale


@dataclass
class JudgeScore:
    """Result of LLM-as-judge scoring."""

    rubric_name: str
    query: str
    response_snippet: str
    dimension_scores: dict[str, int] = field(default_factory=dict)  # dimension -> raw score (1-5)
    dimension_weights: dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0  # 0.0–1.0
    passed: bool = False
    pass_threshold: float = 0.6
    reasoning: str = ""
    raw_judge_output: str = ""


# ─── Base Judge ────────────────────────────────────────────────────────────────


class BaseJudge(ABC):
    """Abstract LLM-as-judge with YAML rubric loading.

    Rubrics are stored in eval/judges/rubrics/*.yaml.
    Each rubric defines dimensions with weights summing to 1.0.

    Validation: before trusting aggregate scores, callers should run
    validate_agreement() on a held-out human-graded sample.
    Target: >= 80% agreement (per eval/research/summary.md §Key Watch-outs).
    """

    _RUBRICS_DIR = Path(__file__).parent / "rubrics"

    def __init__(self, pass_threshold: float = 0.6) -> None:
        self._pass_threshold = pass_threshold
        self._rubric_cache: dict[str, list[RubricDimension]] = {}

    def load_rubric(self, rubric_name: str) -> list[RubricDimension]:
        """Load a rubric from YAML file.

        Args:
            rubric_name: Base filename without .yaml extension.

        Returns:
            List of RubricDimension objects.
        """
        if rubric_name in self._rubric_cache:
            return self._rubric_cache[rubric_name]

        path = self._RUBRICS_DIR / f"{rubric_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Rubric not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        dimensions = [
            RubricDimension(
                name=d["name"],
                weight=d["weight"],
                criteria=d["criteria"],
                max_score=d.get("max_score", 5),
            )
            for d in data.get("dimensions", [])
        ]

        total_weight = sum(d.weight for d in dimensions)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning("Rubric '{}' weights sum to {:.2f}, not 1.0", rubric_name, total_weight)

        self._rubric_cache[rubric_name] = dimensions
        logger.debug("Loaded rubric '{}' with {} dimensions", rubric_name, len(dimensions))
        return dimensions

    def score(
        self,
        query: str,
        response: str,
        rubric_name: str,
        expected_answer: Optional[str] = None,
        context: Optional[str] = None,
    ) -> JudgeScore:
        """Score a response using the named rubric.

        Args:
            query: User query.
            response: Model response to evaluate.
            rubric_name: Name of rubric YAML to use.
            expected_answer: Optional ground-truth answer for reference.
            context: Optional context (e.g. retrieved documents).

        Returns:
            JudgeScore with weighted score and pass/fail.
        """
        dimensions = self.load_rubric(rubric_name)
        raw_output, dimension_scores = self._invoke_judge(
            query, response, dimensions, expected_answer, context
        )

        weighted = sum(
            dimension_scores.get(dim.name, 3) / dim.max_score * dim.weight
            for dim in dimensions
        )

        passed = weighted >= self._pass_threshold

        return JudgeScore(
            rubric_name=rubric_name,
            query=query,
            response_snippet=response[:300],
            dimension_scores=dimension_scores,
            dimension_weights={d.name: d.weight for d in dimensions},
            weighted_score=round(weighted, 4),
            passed=passed,
            pass_threshold=self._pass_threshold,
            raw_judge_output=raw_output,
        )

    def validate_agreement(
        self,
        sample_cases: list[dict],
        human_scores: list[bool],
        rubric_name: str,
    ) -> float:
        """Validate judge accuracy against human grading on a sample.

        Args:
            sample_cases: List of dicts with 'query' and 'response' keys.
            human_scores: Human pass/fail verdicts for each case.
            rubric_name: Rubric to use for judge scoring.

        Returns:
            Agreement rate (0.0–1.0). Target >= 0.80 before trusting scores.
        """
        if len(sample_cases) != len(human_scores):
            raise ValueError("sample_cases and human_scores must have same length")

        agreements = 0
        for case, human_verdict in zip(sample_cases, human_scores):
            judge_result = self.score(case["query"], case["response"], rubric_name)
            if judge_result.passed == human_verdict:
                agreements += 1

        rate = agreements / len(sample_cases)
        logger.info(
            "Judge validation: {}/{} agreements = {:.1f}% (target >= 80%)",
            agreements, len(sample_cases), rate * 100,
        )
        if rate < 0.80:
            logger.warning(
                "Judge agreement {:.1f}% is below 80% threshold — "
                "do not trust aggregate scores without reviewing failing cases",
                rate * 100,
            )
        return round(rate, 4)

    @abstractmethod
    def _invoke_judge(
        self,
        query: str,
        response: str,
        dimensions: list[RubricDimension],
        expected_answer: Optional[str],
        context: Optional[str],
    ) -> tuple[str, dict[str, int]]:
        """Invoke the judge model and return (raw_output, dimension_scores).

        Args:
            query: User query.
            response: Model response.
            dimensions: Rubric dimensions to score.
            expected_answer: Optional expected answer.
            context: Optional context.

        Returns:
            (raw_judge_output_text, {dimension_name: score_1_to_5})
        """
        ...

    @staticmethod
    def _build_scoring_prompt(
        query: str,
        response: str,
        dimensions: list[RubricDimension],
        expected_answer: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """Build the scoring prompt for the judge model."""
        dim_descriptions = "\n".join(
            f"- **{d.name}** (weight {d.weight:.0%}): {d.criteria}"
            for d in dimensions
        )

        expected_section = (
            f"\n**Expected Answer (reference):**\n{expected_answer}\n"
            if expected_answer else ""
        )
        context_section = (
            f"\n**Retrieved Context:**\n{context[:500]}...\n"
            if context else ""
        )

        dim_score_list = "\n".join(
            f"- {d.name}: [1-5]" for d in dimensions
        )

        return f"""You are evaluating an AI financial advisor response for an Indian personal finance context.

**User Query:**
{query}
{expected_section}{context_section}
**AI Response to Evaluate:**
{response}

**Scoring Rubric:**
Score each dimension on a scale of 1 (very poor) to 5 (excellent):

{dim_descriptions}

**Provide your scores in this exact JSON format:**
```json
{{
{dim_score_list.replace('[1-5]', '<score>')}
}}
```

**Reasoning:** [2-3 sentences explaining your scores]

Important: Be strict. A score of 5 requires genuinely excellent, accurate, India-specific advice.
A score of 1 indicates dangerous, incorrect, or irresponsible financial advice.
"""
