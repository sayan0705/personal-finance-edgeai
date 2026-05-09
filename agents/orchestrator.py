"""FinEdge ReAct agent orchestrator — reason, act, observe loop."""

from __future__ import annotations

import json
import re
from typing import AsyncIterator

from loguru import logger

from agents.memory.short_term import ShortTermMemory
from agents.tools.loan_advisor import LoanAdvisor
from agents.tools.registry import ToolRegistry
from agents.tools.sip_calculator import SIPCalculator
from agents.tools.tax_calculator import TaxCalculator
from app.api.config import get
from app.api.llm_client import LLMClient

_SYSTEM_PROMPT = """\
You are FinEdge, an expert personal finance advisor specialising in Indian personal finance. \
You have deep knowledge of:
- Indian income tax (IT Act, 80C, 80D, new vs old regime, FY 2024-25 slabs)
- Investment products (mutual funds, SIP, ELSS, PPF, NPS, FD, bonds)
- SEBI and RBI regulations
- Insurance (term, health, ULIP)
- Banking (loans, EMI, CIBIL score)
- NSE/BSE markets

Always give specific, actionable advice for the Indian context. \
Cite relevant sections (e.g., "Under Section 80C..."). \
Mention current limits and thresholds (FY 2024-25). \
Never hallucinate specific returns or guaranteed profits. \
Recommend consulting a SEBI-registered advisor for large investments.\
"""

_TOOL_PROMPT_TEMPLATE = """\

You have access to the following tools:

{tool_block}

When you need to use a tool, respond EXACTLY in this format (no extra text before Thought):

Thought: <your reasoning>
Action: <tool_name>
Action Input: <valid JSON object>

After receiving the Observation, continue reasoning. When you have the final answer, respond with:

Thought: I have enough information to answer.
Final Answer: <your complete response to the user>

If no tool is needed, go directly to Final Answer.\
"""

# Regex to parse Action/Action Input from LLM output
_ACTION_RE = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)


class FinEdgeOrchestrator:
    """ReAct agent that routes finance queries to tools and the fine-tuned LLM.

    The loop:
    1. Build system + history + user message (with tool descriptions injected).
    2. Call LLM (non-streaming).
    3. Parse response for Action / Action Input.
    4. If found: execute tool → append Observation → goto 2.
    5. If Final Answer or no Action: stream final answer back.
    6. Repeat up to ``max_iterations``.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._max_iter: int = int(get("agent.max_iterations", 5))

        self._registry = ToolRegistry()
        self._registry.register(TaxCalculator())
        self._registry.register(SIPCalculator())
        self._registry.register(LoanAdvisor())

        logger.info(
            f"FinEdgeOrchestrator ready — tools: {[t.name for t in self._registry.list_all()]}"
        )

    def _build_system_message(self) -> dict[str, str]:
        tool_block = self._registry.to_prompt_block()
        content = _SYSTEM_PROMPT + _TOOL_PROMPT_TEMPLATE.format(tool_block=tool_block)
        return {"role": "system", "content": content}

    @staticmethod
    def _parse_action(text: str) -> tuple[str | None, dict | None]:
        """Extract (action_name, action_input_dict) from LLM output.

        Returns:
            Tuple of (action_name, args_dict) or (None, None) if not found.
        """
        action_match = _ACTION_RE.search(text)
        if not action_match:
            return None, None

        action_name = action_match.group(1).strip()
        input_match = _ACTION_INPUT_RE.search(text)
        if not input_match:
            return action_name, {}

        try:
            args = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            logger.warning(f"Could not parse Action Input JSON for action '{action_name}'")
            args = {}

        return action_name, args

    @staticmethod
    def _parse_final_answer(text: str) -> str | None:
        """Extract the Final Answer from LLM output."""
        match = _FINAL_ANSWER_RE.search(text)
        if match:
            return match.group(1).strip()
        return None

    async def run(
        self,
        user_message: str,
        history: list[dict[str, str]],
        rag_context: str = "",
    ) -> AsyncIterator[str]:
        """Execute the ReAct loop and stream the final answer.

        Args:
            user_message: The user's current message.
            history: Previous conversation turns (role/content dicts).
            rag_context: Retrieved context from RAG pipeline (may be empty).

        Yields:
            Text chunks of the final assistant response.
        """
        system_msg = self._build_system_message()

        # Prepend RAG context to the user message if available
        effective_user = user_message
        if rag_context:
            effective_user = (
                f"[Context from knowledge base]\n{rag_context}\n\n[User Question]\n{user_message}"
            )

        messages: list[dict[str, str]] = [system_msg, *history, {"role": "user", "content": effective_user}]

        for iteration in range(1, self._max_iter + 1):
            logger.debug(f"ReAct iteration {iteration}/{self._max_iter}")

            response_text = await self._llm.complete(messages)
            logger.debug(f"LLM response (iter {iteration}): {response_text[:200]}")

            # Check for Final Answer first
            final = self._parse_final_answer(response_text)
            if final:
                logger.info(f"ReAct: Final Answer reached after {iteration} iteration(s)")
                yield final
                return

            # Check for Action
            action_name, action_args = self._parse_action(response_text)
            if action_name:
                logger.info(f"ReAct: executing tool '{action_name}'")
                observation = self._registry.execute(action_name, action_args or {})

                # Append the assistant's reasoning + the observation as a new user turn
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\nContinue your reasoning.",
                })
                continue

            # No action and no Final Answer — treat the whole response as the answer
            logger.info("ReAct: no structured output detected, returning response directly")
            yield response_text
            return

        # Exhausted iterations — stream a final call
        logger.warning(f"ReAct: max iterations ({self._max_iter}) reached, requesting final answer")
        messages.append({
            "role": "user",
            "content": "Please provide your Final Answer now based on the information gathered.",
        })
        async for chunk in self._llm.stream(messages):
            yield chunk
