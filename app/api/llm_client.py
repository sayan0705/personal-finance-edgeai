"""Async HTTP client for the BanyanTree fine-tuned LLM API."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
from loguru import logger

from app.api.config import get


class LLMClient:
    """Async client for the OpenAI-compatible fine-tuned LLM endpoint.

    Reads connection details from config (with env-var override):
    - ``llm.base_url``  → LLM_BASE_URL
    - ``llm.api_key``   → LLM_API_KEY
    - ``llm.model``     → LLM_MODEL
    """

    def __init__(self) -> None:
        self._base_url: str = get("llm.base_url")
        self._api_key: str = get("llm.api_key")
        self._model: str = get("llm.model")
        self._max_tokens: int = int(get("llm.max_tokens", 512))
        self._temperature: float = float(get("llm.temperature", 0.6))
        self._timeout: float = float(get("llm.timeout_seconds", 120))
        self._max_retries: int = int(get("llm.max_retries", 3))

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _build_body(
        self,
        messages: list[dict],
        stream: bool,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
            "stream": stream,
        }

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Non-streaming completion — returns full response text.

        Args:
            messages: List of message dicts with ``role`` and ``content``.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Returns:
            Assistant message content string.

        Raises:
            httpx.HTTPStatusError: On non-2xx response after retries.
        """
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        body = self._build_body(messages, stream=False, max_tokens=max_tokens, temperature=temperature)

        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, headers=self._headers(), json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    logger.debug(f"LLM complete: {len(content)} chars")
                    return content
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning(f"LLM request attempt {attempt}/{self._max_retries} failed: {exc}")
                if attempt == self._max_retries:
                    raise
        return ""

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Streaming completion — yields text chunks as they arrive.

        Args:
            messages: List of message dicts.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Yields:
            Text chunks from the assistant response.
        """
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        body = self._build_body(messages, stream=True, max_tokens=max_tokens, temperature=temperature)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, headers=self._headers(), json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                        text = delta.get("content") or ""
                        if text:
                            yield text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
