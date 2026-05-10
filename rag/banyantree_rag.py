"""Adapter around the original BanyanTree RAG implementation.

The goal of this module is not to reimplement RAG. It imports and owns the
original ``FINANCIAL_HIERARCHICAL_LIGHT_RAG`` lifecycle from
``src/banyanTreev3_agentic.py`` while making it safe for the Docker API.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from app.api.config import get


def _configure_original_runtime_env() -> None:
    """Map container API env vars to the original BanyanTree env vars."""

    os.environ.setdefault("BANYANTREE_SKIP_MCP_BOOTSTRAP", "1")
    os.environ.setdefault("BANYANTREE_LLM_PROVIDER", "api")
    os.environ.setdefault("BANYANTREE_API_MODEL", os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B"))
    os.environ.setdefault("BANYANTREE_API_BASE_URL", os.environ.get("LLM_BASE_URL", "http://95.253.220.115:63084/v1"))
    os.environ.setdefault("BANYANTREE_API_KEY", os.environ.get("LLM_API_KEY", "BANYAN_TREE_LLM_API_KEY"))
    os.environ.setdefault("BANYANTREE_MCP_BASE", "http://mcp-tools:8010")
    os.environ.setdefault("BANYANTREE_DATA_DIR", "/app/data")
    os.environ.setdefault("BANYANTREE_FINANCIAL_KG_ROOT", "/app/data/financial_kg")
    os.environ.setdefault("BANYANTREE_RAW_DOCS_DIR", "/app/data/financial_kg/raw_docs")
    os.environ.setdefault(
        "BANYANTREE_SEED_DOCS_PATH",
        "/app/data/financial_kg/raw_docs/seed/personal_finance_seed.json",
    )
    os.environ.setdefault(
        "BANYANTREE_PAGEINDEX_FLATTENED_DOCS_PATH",
        "/app/data/financial_kg/raw_docs/pageindex/pageindex_flattened_docs.json",
    )


@dataclass
class BanyanTreeRAGResult:
    answer: str
    raw: dict[str, Any]


class BanyanTreeOriginalRAGService:
    """Lazy singleton service for the original BanyanTree RAG engine."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rag: Any | None = None
        self._module: Any | None = None

    def _load_module(self) -> Any:
        if self._module is not None:
            return self._module
        _configure_original_runtime_env()
        self._module = self._load_original_rag_sections()
        return self._module

    def _load_original_rag_sections(self) -> Any:
        """Execute the original BanyanTree RAG code without MCP bootstrap.

        ``src/banyanTreev3_agentic.py`` contains the original design and also a
        top-level notebook-style MCP server bootstrap. Importing it directly
        would start/kill port 8000. To preserve the original RAG while making it
        container-safe, we execute:

        - project path/doc loading setup from the top of the file
        - dependency imports and original RAG/sentiment/query classes

        and skip:

        - MCP server code generation/startup tests
        - demo entrypoint
        """

        source_path = Path(__file__).resolve().parents[1] / "src" / "banyanTreev3_agentic.py"
        text = source_path.read_text(encoding="utf-8")

        server_marker = "# SINGLE CANONICAL MCP SERVER"
        imports_marker = "# LOCAL DEPENDENCY CHECK + IMPORTS"
        demo_marker = "# DEMO"
        header_end = text.index(server_marker)
        imports_start = text.index(imports_marker)
        demo_start = text.index(demo_marker)
        safe_source = text[:header_end] + "\n\n" + text[imports_start:demo_start]

        module_name = "_banyantree_original_rag_runtime"
        module = types.ModuleType(module_name)
        module.__file__ = str(source_path)
        module.__package__ = ""
        sys.modules[module_name] = module
        exec(compile(safe_source, str(source_path), "exec"), module.__dict__)

        async def _http_only_mcp_call_one(client_self, tool_call: dict):
            return await client_self._call_mcp(tool_call)

        # The notebook version mixed HTTP MCP calls with a few in-process
        # fallbacks. In Docker we want faithful service separation, so every
        # selected tool goes through the dedicated mcp-tools container.
        if hasattr(module, "BanyanTreeMCPToolClient"):
            module.BanyanTreeMCPToolClient._call_one = _http_only_mcp_call_one
        return module

    def _ensure_ready(self) -> Any:
        if self._rag is not None:
            return self._rag
        with self._lock:
            if self._rag is not None:
                return self._rag

            module = self._load_module()
            logger.info("Initialising original FINANCIAL_HIERARCHICAL_LIGHT_RAG")
            rag = module.FINANCIAL_HIERARCHICAL_LIGHT_RAG(kg_db_path="finsage_final_kg")

            # Keep the original flow: query_agentic uses the original
            # BanyanTreeMCPToolClient/MCPToolClient over HTTP. In Docker the MCP
            # server runs as the separate `mcp-tools` service on port 8010.

            docs = module.load_financial_docs()
            if docs and not getattr(rag, "documents", []):
                logger.info(f"Ingesting {len(docs)} BanyanTree seed/PageIndex docs into original RAG")
                rag.ingest_financial_documents(docs)
                rag.build_raptor_root()
                rag.build_knowledge_graph_from_documents()
            elif docs and len(getattr(rag, "documents", [])) < len(docs):
                logger.info("Existing RAG DB found; using persisted indices and KG")

            self._rag = rag
            return self._rag

    async def query(self, question: str, k: int = 8) -> BanyanTreeRAGResult:
        rag = await asyncio.to_thread(self._ensure_ready)
        result = await rag.query_agentic(question, k=k)
        return BanyanTreeRAGResult(answer=str(result.get("answer", "")), raw=result)


_SERVICE: BanyanTreeOriginalRAGService | None = None


def get_banyantree_rag_service() -> BanyanTreeOriginalRAGService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = BanyanTreeOriginalRAGService()
    return _SERVICE
