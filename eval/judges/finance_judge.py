"""Finance domain LLM-as-judge using the Anthropic Claude API.

Uses claude-sonnet-4-6 by default to score model responses against India
personal finance rubrics. Falls back to a heuristic scorer if the API is
not available (no ANTHROPIC_API_KEY).

Validation requirement: run validate_agreement() on a 10-case human-graded
sample before trusting aggregate scores. Target >= 80% agreement.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from loguru import logger

from eval.judges.base_judge import BaseJudge, RubricDimension


class FinanceJudge(BaseJudge):
    """LLM-as-judge for Indian personal finance evaluation.

    Supported rubrics:
    - tax_advice: Correctness, regime clarity, section citation, safety posture
    - investment_advice: Math accuracy, India-relevance, risk disclosure, actionability
    - refusal_scope: Correct identification of out-of-scope queries, refusal quality

    Use validate_agreement() with 10 human-graded cases before using in production.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        pass_threshold: float = 0.60,
        max_tokens: int = 512,
    ) -> None:
        super().__init__(pass_threshold=pass_threshold)
        self._model = model
        self._max_tokens = max_tokens
        self._client = self._init_client()

    def _init_client(self):
        """Initialize Anthropic client if API key is available."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set — FinanceJudge will use heuristic fallback scorer"
            )
            return None
        try:
            import anthropic
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            logger.warning("anthropic package not installed — using heuristic fallback scorer")
            return None

    def _invoke_judge(
        self,
        query: str,
        response: str,
        dimensions: list[RubricDimension],
        expected_answer: Optional[str],
        context: Optional[str],
    ) -> tuple[str, dict[str, int]]:
        """Invoke Claude as judge.

        Falls back to heuristic scorer if Claude API is unavailable.

        Args:
            query: User query.
            response: Model response.
            dimensions: Rubric dimensions to score.
            expected_answer: Optional expected answer.
            context: Optional context.

        Returns:
            (raw_output, {dimension_name: score_1_to_5})
        """
        prompt = self._build_scoring_prompt(query, response, dimensions, expected_answer, context)

        if self._client is None:
            return self._heuristic_fallback(response, dimensions)

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_output = message.content[0].text
            scores = self._parse_scores(raw_output, dimensions)
            return raw_output, scores

        except Exception as exc:
            logger.error("Claude judge API error: {} — using heuristic fallback", exc)
            return self._heuristic_fallback(response, dimensions)

    @staticmethod
    def _parse_scores(
        raw_output: str, dimensions: list[RubricDimension]
    ) -> dict[str, int]:
        """Parse dimension scores from judge output.

        Handles JSON block extraction and falls back to regex extraction.

        Args:
            raw_output: Raw judge model output.
            dimensions: Expected dimension names.

        Returns:
            Dict mapping dimension name to integer score 1–5.
        """
        scores: dict[str, int] = {}

        # Try JSON block first
        json_match = re.search(r"```json\s*(.*?)\s*```", raw_output, re.DOTALL)
        if not json_match:
            json_match = re.search(r"\{[^{}]+\}", raw_output, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(1) if "```" in raw_output else json_match.group(0))
                for dim in dimensions:
                    key = dim.name.lower().replace(" ", "_")
                    if dim.name in data:
                        scores[dim.name] = max(1, min(5, int(data[dim.name])))
                    elif key in data:
                        scores[dim.name] = max(1, min(5, int(data[key])))
                if len(scores) == len(dimensions):
                    return scores
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.debug("JSON parse failed: {} — trying regex", e)

        # Regex fallback: look for "dimension_name: 4" patterns
        for dim in dimensions:
            pattern = re.compile(
                rf"{re.escape(dim.name)}[:\s]*([1-5])", re.IGNORECASE
            )
            match = pattern.search(raw_output)
            if match:
                scores[dim.name] = int(match.group(1))
            else:
                scores[dim.name] = 3  # default to middle score if not found
                logger.debug("Could not parse score for dimension '{}', defaulting to 3", dim.name)

        return scores

    @staticmethod
    def _heuristic_fallback(
        response: str, dimensions: list[RubricDimension]
    ) -> tuple[str, dict[str, int]]:
        """Simple heuristic scorer when Claude API is unavailable.

        Assigns scores based on response length and presence of key signals.
        This is not a substitute for real LLM judging — use only for development.

        Args:
            response: Model response text.
            dimensions: Rubric dimensions.

        Returns:
            (note, {dimension_name: heuristic_score})
        """
        response_lower = response.lower()
        scores: dict[str, int] = {}

        quality_signals = {
            "calculation_accuracy": ["₹", "lakh", "crore", "%", "total", "tax"],
            "regime_clarity": ["new regime", "old regime", "section", "80c", "FY 2024"],
            "section_citation": ["section 80", "section 10", "section 24", "under section"],
            "safety_posture": ["consult", "sebi", "advisor", "professional", "ca"],
            "india_relevance": ["india", "Indian", "₹", "SEBI", "RBI", "lakh"],
            "math_accuracy": ["₹", "calculated", "formula", "per month", "annually"],
            "risk_disclosure": ["risk", "market", "not guaranteed", "may vary", "past performance"],
            "actionability": ["you can", "you should", "consider", "recommended", "steps"],
            "refusal_quality": ["cannot", "not able", "regret", "recommend consulting", "sebi-registered"],
            "scope_identification": ["out of scope", "cannot recommend", "not financial advice", "consult"],
        }

        for dim in dimensions:
            signals = quality_signals.get(dim.name.lower().replace(" ", "_"), [])
            hits = sum(1 for s in signals if s.lower() in response_lower)
            # Scale hits to 1-5: 0→2, 1→3, 2→3, 3→4, 4+→4
            base = 2 + min(hits, 2)
            # Penalize very short responses
            if len(response) < 100:
                base = max(1, base - 1)
            # Reward thorough responses
            if len(response) > 500:
                base = min(5, base + 1)
            scores[dim.name] = base

        note = "[HEURISTIC FALLBACK — Claude API unavailable. These scores are unreliable.]"
        return note, scores
