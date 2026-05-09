"""Unified model interface — wraps llama-cpp-python (standalone) or FinEdgeOrchestrator.

Standalone mode: evaluates the GGUF model directly via llama-cpp-python.
                 Works NOW before the agent stack (agents/, rag/) is built.

Orchestrator mode: delegates to FinEdgeOrchestrator when agents/ is present.
                   Enables full trajectory evaluation including tool calls and RAG.

All eval runners import ModelAdapter — never import llama-cpp directly in runners.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class GenerationResult:
    """Result of a single model generation call."""

    response: str
    ttft_ms: float          # time to first token (ms) — approximated in standalone mode
    total_ms: float         # end-to-end latency (ms)
    tokens_generated: int   # number of tokens in the response
    tool_calls: list[dict] = field(default_factory=list)  # empty in standalone mode
    model_mode: str = "standalone"  # "standalone" or "orchestrator"


# ─── Model Adapter ─────────────────────────────────────────────────────────────


class ModelAdapter:
    """Unified model interface for eval pipeline.

    Usage:
        adapter = ModelAdapter(config)
        adapter.load()
        result = adapter.generate("What is my tax for 10L?")
        adapter.unload()

    Config keys (from eval/configs/eval_config.yaml):
        model.path: Path to GGUF model file
        model.n_ctx: Context length (default 4096)
        model.n_threads: CPU threads (default 8)
        model.n_gpu_layers: GPU layers to offload (default 0)
        model.max_new_tokens: Max tokens to generate (default 512)
        model.temperature: Sampling temperature (default 0.7)
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._model_config = config.get("model", {})
        self._model = None          # llama-cpp Llama instance
        self._orchestrator = None   # FinEdgeOrchestrator instance
        self._mode = self._detect_mode()
        logger.info("ModelAdapter initialized in '{}' mode", self._mode)

    def _detect_mode(self) -> str:
        """Detect mode from config, then fall back to auto-detection.

        Priority:
          1. model.type = "api"          → "api"
          2. agents.orchestrator present → "orchestrator"
          3. default                     → "standalone"
        """
        if self._model_config.get("type", "").lower() == "api":
            return "api"
        try:
            import importlib.util
            spec = importlib.util.find_spec("agents.orchestrator")
            if spec is not None:
                return "orchestrator"
        except (ImportError, ModuleNotFoundError, ValueError):
            pass
        return "standalone"

    def load(self) -> None:
        """Load model or orchestrator. Must call before generate()."""
        if self._mode == "api":
            self._load_api()
        elif self._mode == "orchestrator":
            self._load_orchestrator()
        else:
            self._load_llama_cpp()

    def _load_api(self) -> None:
        """Validate API config. No persistent connection needed for HTTP."""
        api_base = self._model_config.get("api_base", "")
        api_model = self._model_config.get("api_model", "")
        if not api_base:
            raise ValueError("model.api_base not set in config (required for type=api)")
        if not api_model:
            raise ValueError("model.api_model not set in config (required for type=api)")
        logger.info("API mode: endpoint={} model={}", api_base, api_model)

    def _load_llama_cpp(self) -> None:
        """Load GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise ImportError(
                "llama-cpp-python required for standalone mode. "
                "Install: pip install llama-cpp-python"
            ) from exc

        model_path = self._model_config.get("path", "")
        if not model_path:
            raise ValueError("model.path not set in config")

        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GGUF model not found: {model_path}\n"
                "Download Qwen3-8B Q4_K_M from HuggingFace or run make quantize"
            )

        logger.info("Loading GGUF model: {}", model_path)
        t0 = time.perf_counter()
        self._model = Llama(
            model_path=model_path,
            n_ctx=self._model_config.get("n_ctx", 4096),
            n_threads=self._model_config.get("n_threads", 8),
            n_gpu_layers=self._model_config.get("n_gpu_layers", 0),
            verbose=False,
            logits_all=False,
        )
        load_ms = (time.perf_counter() - t0) * 1000
        logger.info("Model loaded in {:.0f} ms (standalone mode)", load_ms)

    def _load_orchestrator(self) -> None:
        """Load the FinEdgeOrchestrator from agents/ package."""
        from agents.orchestrator import FinEdgeOrchestrator
        self._orchestrator = FinEdgeOrchestrator(self._config)
        logger.info("FinEdgeOrchestrator loaded (orchestrator mode)")

    def generate(
        self,
        query: str,
        system_prompt: str = "",
    ) -> GenerationResult:
        """Generate a response for the given query.

        Args:
            query: User query text.
            system_prompt: Optional system prompt (overrides default if provided).

        Returns:
            GenerationResult with response text and timing metadata.
        """
        if self._mode == "api":
            return self._generate_api(query, system_prompt)
        if self._mode == "orchestrator":
            return self._generate_orchestrator(query, system_prompt)
        return self._generate_standalone(query, system_prompt)

    def _generate_standalone(self, query: str, system_prompt: str) -> GenerationResult:
        """Generate via llama-cpp-python in standalone mode."""
        if self._model is None:
            raise RuntimeError("Call load() before generate()")

        default_system = (
            "You are FinEdge, an expert personal finance advisor specializing in Indian "
            "personal finance including income tax, mutual funds, insurance, banking, and "
            "investments. Always provide accurate, actionable advice based on current Indian "
            "regulations. Recommend consulting a SEBI-registered advisor for large investments."
        )

        messages = [
            {"role": "system", "content": system_prompt or default_system},
            {"role": "user", "content": query},
        ]

        t0 = time.perf_counter()
        output = self._model.create_chat_completion(
            messages=messages,
            max_tokens=self._model_config.get("max_new_tokens", 512),
            temperature=self._model_config.get("temperature", 0.7),
            top_p=self._model_config.get("top_p", 0.9),
        )
        total_ms = (time.perf_counter() - t0) * 1000

        response_text = output["choices"][0]["message"]["content"] or ""
        tokens_generated = output.get("usage", {}).get("completion_tokens", len(response_text.split()))

        return GenerationResult(
            response=response_text,
            ttft_ms=total_ms,  # llama-cpp non-streaming doesn't expose TTFT separately
            total_ms=round(total_ms, 1),
            tokens_generated=tokens_generated,
            model_mode="standalone",
        )

    def _generate_api(self, query: str, system_prompt: str) -> GenerationResult:
        """Generate via a remote OpenAI-compatible /v1/chat/completions endpoint."""
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx not installed — run: pip install httpx") from exc

        default_system = (
            "You are FinEdge, an expert personal finance advisor specializing in Indian "
            "personal finance including income tax, mutual funds, insurance, banking, and "
            "investments. Always provide accurate, actionable advice based on current Indian "
            "regulations. Recommend consulting a SEBI-registered advisor for large investments."
        )

        api_base = self._model_config.get("api_base", "").rstrip("/")
        api_key = self._model_config.get("api_key", "xxxx")
        api_model = self._model_config.get("api_model", "")
        timeout = float(self._model_config.get("api_timeout", 120))

        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": api_model,
            "messages": [
                {"role": "system", "content": system_prompt or default_system},
                {"role": "user", "content": query},
            ],
            "max_tokens": self._model_config.get("max_new_tokens", 512),
            "temperature": self._model_config.get("temperature", 0.7),
            "top_p": self._model_config.get("top_p", 0.9),
        }

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(f"{api_base}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("API request failed: {} — {}", exc.response.status_code, exc.response.text)
            raise
        except httpx.RequestError as exc:
            logger.error("API connection error: {}", exc)
            raise

        total_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()

        response_text = data["choices"][0]["message"]["content"] or ""
        tokens_generated = data.get("usage", {}).get("completion_tokens", len(response_text.split()))

        return GenerationResult(
            response=response_text,
            ttft_ms=total_ms,  # non-streaming: no separate TTFT available
            total_ms=round(total_ms, 1),
            tokens_generated=tokens_generated,
            model_mode="api",
        )

    def _generate_orchestrator(self, query: str, system_prompt: str) -> GenerationResult:
        """Generate via FinEdgeOrchestrator (full agent + RAG + tools)."""
        t0 = time.perf_counter()
        try:
            result = self._orchestrator.generate(
                query=query,
                system_prompt=system_prompt or None,
            )
        except Exception as exc:
            logger.error("Orchestrator generation failed: {}", exc)
            return GenerationResult(
                response=f"[ERROR: {exc}]",
                ttft_ms=0.0,
                total_ms=round((time.perf_counter() - t0) * 1000, 1),
                tokens_generated=0,
                model_mode="orchestrator",
            )

        total_ms = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            response=result.get("response", ""),
            ttft_ms=result.get("ttft_ms", total_ms),
            total_ms=round(total_ms, 1),
            tokens_generated=result.get("tokens_generated", 0),
            tool_calls=result.get("tool_calls", []),
            model_mode="orchestrator",
        )

    def is_agent_mode(self) -> bool:
        """Return True if using the full orchestrator stack with tools."""
        return self._mode == "orchestrator"

    def is_api_mode(self) -> bool:
        """Return True if routing inference to a remote OpenAI-compatible API."""
        return self._mode == "api"

    def unload(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("GGUF model unloaded")
        if self._orchestrator is not None:
            del self._orchestrator
            self._orchestrator = None
            logger.info("Orchestrator unloaded")

    def __enter__(self) -> "ModelAdapter":
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload()
