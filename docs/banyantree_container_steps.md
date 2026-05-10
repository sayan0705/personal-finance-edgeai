# BanyanTree Container Flow

This repo preserves the original BanyanTree RAG design and only changes the
folder/container structure.

## Active Runtime Flow

```text
OpenWebUI / API query
  -> app/api/main.py
  -> app/api/orchestrator.py
  -> agents/orchestrator.py
  -> rag/banyantree_rag.py
  -> original FINANCIAL_HIERARCHICAL_LIGHT_RAG.query_agentic(...)
      -> guardrail
      -> Qwen sentiment analysis
      -> workflow router
      -> HTTP MCP tool calls to mcp-tools:8010
      -> original LightRAG retrieval
      -> KG local search
      -> PageIndex/Qwen extracted docs already ingested into the same RAG corpus
      -> grounded answer generation
```

## Original RAG Location

The original implementation remains in:

```text
src/banyanTreev3_agentic.py
```

The class used is:

```text
FINANCIAL_HIERARCHICAL_LIGHT_RAG
```

The container adapter is:

```text
rag/banyantree_rag.py
```

It loads only the original RAG sections from `src/banyanTreev3_agentic.py` and
skips the notebook-style top-level MCP server bootstrap. Tool calls still go
through the original HTTP MCP client, but the MCP server is now a dedicated
container on port `8010`.

## MCP Tools Server

The MCP server runs from:

```text
src/finsage_mcp_server.py
```

Docker service:

```text
mcp-tools
```

Internal URL used by the API container:

```text
http://mcp-tools:8010
```

Host URL for debugging:

```text
http://localhost:8010/health
http://localhost:8010/tools
```

## KG + PageIndex Data

Input docs are loaded by the original `load_financial_docs()` logic from:

```text
data/financial_kg/raw_docs/seed/personal_finance_seed.json
data/financial_kg/raw_docs/pageindex/pageindex_flattened_docs.json
```

Original RAG persisted artifacts are written under:

```text
data/financial_kg/lightrag
data/financial_kg/graph
```

PDF inputs remain here:

```text
data/financial_kg/pageindex/inputs
```

## Models

Chat/generation, sentiment routing, and tool selection use API mode:

```text
LLM_MODEL=Qwen/Qwen3-8B
BANYANTREE_LLM_PROVIDER=api
BANYANTREE_API_MODEL=Qwen/Qwen3-8B
```

Original RAG embeddings use:

```text
BANYANTREE_EMBEDDING_MODEL=BAAI/bge-m3
```

Offline PDF extraction uses:

```text
BANYANTREE_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
scripts/import_pdf_qwen_docs.py
```

## Container Services

`docker-compose.yml` starts:

```text
 mcp-tools
api
openwebui
```

No ChromaDB service is used. The original BanyanTree RAG uses its own FAISS,
BM25, RAPTOR/community summaries, and NetworkX KG artifacts in `data/financial_kg`.
