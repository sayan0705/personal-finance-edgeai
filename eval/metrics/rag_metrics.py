"""RAG evaluation metrics — RAGAS wrapper + source citation rate.

Wraps the RAGAS library (faithfulness, answer relevancy, context precision/recall)
and adds an India-specific source citation presence check.

If RAGAS is not installed, all methods return 0.0 with a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class RAGScore:
    """RAGAS metric scores for a single query."""

    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    citation_present: Optional[bool] = None

    def aggregate(self) -> float:
        """Compute mean of available numeric scores."""
        scores = [
            v for v in [self.faithfulness, self.answer_relevancy,
                        self.context_precision, self.context_recall]
            if v is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0


# ─── RAG Metrics ──────────────────────────────────────────────────────────────


class RAGMetrics:
    """RAGAS-backed RAG evaluation + citation rate check.

    Targets (from eval/configs/eval_config.yaml):
    - faithfulness > 0.85
    - answer_relevancy: directionally informative
    - context_precision > 0.80
    - context_recall > 0.75
    - source_citation_rate > 90%
    """

    def __init__(self) -> None:
        self._ragas_available = self._check_ragas()

    def score_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> Optional[float]:
        """Compute RAGAS faithfulness score.

        Measures: are all claims in the answer entailed by the retrieved contexts?

        Args:
            question: User query.
            answer: Model-generated answer.
            contexts: List of retrieved document chunks.

        Returns:
            Faithfulness score 0.0–1.0, or None if RAGAS unavailable.
        """
        if not self._ragas_available:
            logger.warning("RAGAS not installed — faithfulness score unavailable")
            return None

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness

            sample = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            })
            result = evaluate(sample, metrics=[faithfulness])
            score = result["faithfulness"]
            logger.debug("faithfulness={:.3f}", score)
            return round(float(score), 4)
        except Exception as exc:
            logger.error("RAGAS faithfulness error: {}", exc)
            return None

    def score_answer_relevancy(
        self,
        question: str,
        answer: str,
        contexts: Optional[list[str]] = None,
    ) -> Optional[float]:
        """Compute RAGAS answer relevancy score.

        Measures: is the answer relevant and responsive to the question?

        Args:
            question: User query.
            answer: Model-generated answer.
            contexts: Retrieved context (optional for this metric).

        Returns:
            Answer relevancy score 0.0–1.0, or None if RAGAS unavailable.
        """
        if not self._ragas_available:
            return None

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy

            sample = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts or [""]],
            })
            result = evaluate(sample, metrics=[answer_relevancy])
            score = result["answer_relevancy"]
            return round(float(score), 4)
        except Exception as exc:
            logger.error("RAGAS answer_relevancy error: {}", exc)
            return None

    def score_context_precision(
        self,
        question: str,
        contexts: list[str],
        ground_truth: str,
    ) -> Optional[float]:
        """Compute RAGAS context precision.

        Measures: are the retrieved chunks actually useful for answering the question?

        Args:
            question: User query.
            contexts: Retrieved document chunks.
            ground_truth: Ground-truth answer for reference.

        Returns:
            Context precision score 0.0–1.0, or None if RAGAS unavailable.
        """
        if not self._ragas_available:
            return None

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import context_precision

            sample = Dataset.from_dict({
                "question": [question],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            })
            result = evaluate(sample, metrics=[context_precision])
            score = result["context_precision"]
            return round(float(score), 4)
        except Exception as exc:
            logger.error("RAGAS context_precision error: {}", exc)
            return None

    def score_context_recall(
        self,
        question: str,
        contexts: list[str],
        ground_truth: str,
    ) -> Optional[float]:
        """Compute RAGAS context recall.

        Measures: does the retrieved context contain all information needed?

        Args:
            question: User query.
            contexts: Retrieved document chunks.
            ground_truth: Ground-truth answer for reference.

        Returns:
            Context recall score 0.0–1.0, or None if RAGAS unavailable.
        """
        if not self._ragas_available:
            return None

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import context_recall

            sample = Dataset.from_dict({
                "question": [question],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            })
            result = evaluate(sample, metrics=[context_recall])
            score = result["context_recall"]
            return round(float(score), 4)
        except Exception as exc:
            logger.error("RAGAS context_recall error: {}", exc)
            return None

    def score_all(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
        expected_sources: Optional[list[str]] = None,
    ) -> RAGScore:
        """Compute all available RAGAS metrics for a single query.

        Args:
            question: User query.
            answer: Model-generated answer.
            contexts: Retrieved document chunks.
            ground_truth: Ground-truth expected answer.
            expected_sources: Source identifiers that should appear in the answer.

        Returns:
            RAGScore with all available metric values.
        """
        return RAGScore(
            faithfulness=self.score_faithfulness(question, answer, contexts),
            answer_relevancy=self.score_answer_relevancy(question, answer, contexts),
            context_precision=self.score_context_precision(question, contexts, ground_truth),
            context_recall=self.score_context_recall(question, contexts, ground_truth),
            citation_present=self.check_citation_present(answer, expected_sources or []),
        )

    @staticmethod
    def check_citation_present(response: str, expected_sources: list[str]) -> bool:
        """Check whether the response cites expected regulatory sources.

        Looks for SEBI/RBI/IT Act section references, circular numbers, or
        explicit source mentions. Required for India finance RAG responses.

        Args:
            response: Model response text.
            expected_sources: List of source types to look for (e.g. ["sebi_circulars"]).

        Returns:
            True if at least one expected source type is cited.
        """
        if not expected_sources:
            return True

        # Citation signal patterns mapped to source types
        citation_patterns = {
            "sebi_circulars": [
                r"SEBI\s+circular",
                r"SEBI/",
                r"sebi\.gov\.in",
                r"Securities\s+and\s+Exchange\s+Board",
                r"PFUTP",
                r"SEBI\s+Regulation",
            ],
            "rbi_guidelines": [
                r"RBI\s+circular",
                r"Reserve\s+Bank\s+of\s+India",
                r"rbi\.org\.in",
                r"Master\s+Direction",
                r"RBI/",
            ],
            "income_tax": [
                r"Section\s+\d+",
                r"Income\s+Tax\s+Act",
                r"IT\s+Act",
                r"Income-tax\s+Act",
                r"80[A-Z]{1,2}",
                r"incometax\.gov\.in",
            ],
        }

        import re
        for source in expected_sources:
            patterns = citation_patterns.get(source, [])
            for pat in patterns:
                if re.search(pat, response, re.IGNORECASE):
                    return True

        return False

    @staticmethod
    def _check_ragas() -> bool:
        """Check whether RAGAS is importable."""
        try:
            import ragas  # noqa: F401
            return True
        except ImportError:
            logger.info("RAGAS not installed — install with: pip install ragas")
            return False
