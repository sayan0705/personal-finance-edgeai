"""HTTP-level orchestrator — wires guardrails, memory, RAG, and the ReAct agent."""

from __future__ import annotations

from typing import AsyncIterator

from loguru import logger

from agents.memory.short_term import ShortTermMemory
from agents.orchestrator import FinEdgeOrchestrator
from app.api.guardrails.input_guard import InputGuardrail
from app.api.guardrails.output_guard import OutputGuardrail
from app.api.llm_client import LLMClient
from app.api.schemas import ChatMessage
from rag.pipeline import RAGPipeline


class GuardrailError(Exception):
    """Raised when an input guardrail rejects a message."""


class FinEdgeAPIOrchestrator:
    """Ties together all components for a single request.

    Responsibilities:
    - Input guardrail check
    - RAG context retrieval (stub)
    - Per-session short-term memory management
    - ReAct agent execution
    - Output guardrail processing
    - Streaming token delivery

    One instance is shared across all requests (stateless except for
    lazily initialised singletons). Memory is keyed by session_id.
    """

    def __init__(self) -> None:
        self._input_guard = InputGuardrail()
        self._output_guard = OutputGuardrail()
        self._rag = RAGPipeline()
        self._llm = LLMClient()
        self._agent = FinEdgeOrchestrator(self._llm)
        # session_id → ShortTermMemory
        self._sessions: dict[str, ShortTermMemory] = {}
        logger.info("FinEdgeAPIOrchestrator initialised")

    def _get_memory(self, session_id: str) -> ShortTermMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ShortTermMemory()
        return self._sessions[session_id]

    def _extract_user_text(self, messages: list[ChatMessage]) -> str:
        """Return the last user message content."""
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return ""

    def _extract_history(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Return all messages except the last user turn as history dicts."""
        history = []
        for msg in messages[:-1]:
            if msg.role in ("user", "assistant"):
                history.append({"role": msg.role, "content": msg.content})
        return history

    async def stream_response(
        self,
        messages: list[ChatMessage],
        session_id: str,
    ) -> AsyncIterator[str]:
        """Run the full pipeline and stream response chunks.

        Args:
            messages: Full conversation messages from the client request.
            session_id: Unique session identifier (used for memory).

        Yields:
            Text chunks of the final response.

        Raises:
            GuardrailError: If input guardrail rejects the message.
        """
        user_text = self._extract_user_text(messages)

        # 1. Input guardrail
        guard_result = self._input_guard.validate(user_text)
        if not guard_result.allowed:
            logger.warning(f"Input blocked for session {session_id}: {guard_result.reason}")
            raise GuardrailError(guard_result.reason)

        # 2. RAG context (returns "" when disabled)
        rag_context = self._rag.query(user_text)

        # 3. Conversation history from client messages (not memory, to stay stateless per request)
        history = self._extract_history(messages)

        # 4. Run ReAct agent — buffer full response for output guardrail
        full_response_parts: list[str] = []
        async for chunk in self._agent.run(user_text, history, rag_context):
            full_response_parts.append(chunk)

        full_response = "".join(full_response_parts)

        # 5. Output guardrail
        final_text = self._output_guard.process(full_response)

        # 6. Yield in small chunks for streaming UX
        chunk_size = 20
        for i in range(0, len(final_text), chunk_size):
            yield final_text[i : i + chunk_size]
