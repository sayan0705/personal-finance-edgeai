"""Thin container orchestrator that delegates to original BanyanTree RAG.

This file intentionally does not contain the RAG implementation. It preserves
the container structure while keeping the original BanyanTree design in the
dedicated RAG adapter:

``rag/banyantree_rag.py`` -> original ``FINANCIAL_HIERARCHICAL_LIGHT_RAG``.
"""

from __future__ import annotations

from typing import AsyncIterator

from app.api.config import get
from app.api.llm_client import LLMClient
from rag.banyantree_rag import get_banyantree_rag_service


class FinEdgeOrchestrator:
    """API-facing wrapper over the original BanyanTree query_agentic flow."""

    def __init__(self, llm_client: LLMClient) -> None:
        # Kept in the constructor signature so app/api/orchestrator.py remains
        # unchanged. The original RAG uses env-mapped API settings internally.
        self._llm = llm_client
        self._rag_service = get_banyantree_rag_service()

    async def run(
        self,
        user_message: str,
        history: list[dict[str, str]],
        rag_context: str = "",
    ) -> AsyncIterator[str]:
        result = await self._rag_service.query(user_message)
        text = result.answer

        if str(get("app.show_flow_trace", "true")).lower() == "true":
            raw = result.raw
            flow = raw.get("sentiment_analysis") or raw.get("routing") or raw.get("classifier") or {}
            tools = ", ".join(raw.get("tool_calls", []) or []) or "none"
            sources = ", ".join((raw.get("sources", []) or [])[:4]) or "none"
            text += (
                "\n\n---\n"
                "Flow: original FINANCIAL_HIERARCHICAL_LIGHT_RAG.query_agentic\n"
                f"Mode: {raw.get('mode', 'unknown')}\n"
                f"Guardrail: {flow.get('guardrail', 'n/a')} | Workflow: {flow.get('workflow', flow.get('intent', 'n/a'))}\n"
                f"Sentiment: {flow.get('sentiment', 'n/a')} | Risk: {flow.get('risk_profile', 'n/a')} | Urgency: {flow.get('urgency', 'n/a')}\n"
                f"Tools: {tools}\n"
                f"Sources: {sources}\n"
                f"KG: {raw.get('kg_stats', {})}"
            )

        yield text
