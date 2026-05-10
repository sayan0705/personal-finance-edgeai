"""HTTP-level adapter for the original BanyanTree flow.

This layer intentionally avoids adding a second guardrail/RAG implementation.
The original flow starts inside ``FINANCIAL_HIERARCHICAL_LIGHT_RAG.query_agentic``
via ``agents.orchestrator`` -> ``rag.banyantree_rag``.
"""

from __future__ import annotations

from typing import AsyncIterator

from loguru import logger

from agents.memory.short_term import ShortTermMemory
from agents.orchestrator import FinEdgeOrchestrator
from app.api.llm_client import LLMClient
from app.api.schemas import ChatMessage


class GuardrailError(Exception):
    """Kept for API compatibility; original BanyanTree guardrail handles blocks."""


class FinEdgeAPIOrchestrator:
    """Pass API messages to the original BanyanTree query flow."""

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._agent = FinEdgeOrchestrator(self._llm)
        self._sessions: dict[str, ShortTermMemory] = {}
        logger.info("BanyanTree API adapter initialised; original query_agentic owns guardrails/RAG")

    def _get_memory(self, session_id: str) -> ShortTermMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ShortTermMemory()
        return self._sessions[session_id]

    def _extract_user_text(self, messages: list[ChatMessage]) -> str:
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return ""

    async def stream_response(
        self,
        messages: list[ChatMessage],
        session_id: str,
    ) -> AsyncIterator[str]:
        user_text = self._extract_user_text(messages)

        full_response_parts: list[str] = []
        async for chunk in self._agent.run(user_text, history=[], rag_context=""):
            full_response_parts.append(chunk)

        final_text = "".join(full_response_parts)
        chunk_size = 20
        for i in range(0, len(final_text), chunk_size):
            yield final_text[i : i + chunk_size]
