# CLAUDE.md — Implementation Guide for Claude Code

> This file provides Claude Code with all the context, conventions, and implementation instructions needed to build the **personal-finance-edge-ai** project correctly.

---

## 🎯 Project Summary

**personal-finance-edge-ai** is an end-to-end pipeline to build a privacy-first, on-device AI assistant for Indian personal finance. The pipeline covers:

1. Indian finance dataset curation (public + custom)
2. SLM selection from candidates (Phi-3, Gemma, Qwen2, SmolLM2, TinyLlama)
3. Fine-tuning with LoRA/QLoRA (HuggingFace PEFT + TRL)
4. Benchmarking (public FinanceBench + custom India finance benchmark)
5. Edge optimization (GGUF quantization via llama.cpp)
6. RAG pipeline (FAISS + semantic chunking + reranking)
7. Agentic workflow (ReAct agent with Indian finance tools)
8. App layer (CLI, FastAPI, Gradio)

All inference must run **on-device** — no cloud API calls during inference.

---

## 📁 Repo Layout (implement in this order)

```
personal-finance-edge-ai/
├── src/data/           ← Step 1: Data ingestion & preprocessing
├── slm/selection/      ← Step 2: SLM comparison scripts
├── slm/fine-tuning/    ← Step 3: LoRA/QLoRA training
├── benchmarks/         ← Step 4: Evaluation framework
├── edge/               ← Step 5: Quantization & edge runtimes
├── rag/                ← Step 6: RAG pipeline
├── agents/             ← Step 7: Agentic workflow
└── app/                ← Step 8: CLI + API + UI
```

---

## 🔧 Coding Conventions

- **Language**: Python 3.10+
- **Formatting**: `black` + `isort` (line length 100)
- **Linting**: `ruff`
- **Type hints**: Required on all public functions
- **Docstrings**: Google-style docstrings on all classes and public methods
- **Config**: All hyperparameters and paths in `configs/*.yaml`, never hardcoded
- **Logging**: Use `loguru` (`from loguru import logger`), not `print()`
- **Error handling**: Use specific exception types, always log errors before raising
- **Tests**: Write `pytest` unit tests for every module in `tests/unit/`

---

## ⚙️ Config Files to Create

### `configs/app_config.yaml`
```yaml
app:
  name: personal-finance-edge-ai
  version: "0.1.0"
  log_level: INFO

model:
  base_model: "Qwen/Qwen2-1.5B-Instruct"
  edge_model_path: "models/finedge-Q4_K_M.gguf"
  context_length: 4096
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9

rag:
  enabled: true
  vector_store: "faiss"
  index_path: "data/processed/rag-corpus/faiss_index"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  retrieval_k: 10
  rerank_top_n: 3
  chunk_size: 512
  chunk_overlap: 64

agent:
  enabled: true
  max_iterations: 5
  tools: ["tax_calculator", "sip_calculator", "budget_analyzer", "loan_advisor"]

memory:
  short_term_window: 10
  long_term_path: "data/user_profile.json"
```

### `configs/training_config.yaml`
```yaml
training:
  output_dir: "models/adapters"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  weight_decay: 0.001
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  max_seq_length: 2048
  fp16: false
  bf16: true
  logging_steps: 10
  eval_steps: 100
  save_steps: 500
  save_total_limit: 3
  load_best_model_at_end: true
  report_to: "wandb"
```

### `configs/lora_config.yaml`
```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

qlora:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
```

### `configs/dataset_config.yaml`
```yaml
dataset:
  train_path: "data/processed/train"
  eval_path: "data/processed/eval"
  test_path: "data/processed/test"
  text_column: "text"
  split_ratios:
    train: 0.80
    eval: 0.10
    test: 0.10

sources:
  public:
    - name: "sebi_circulars"
      path: "data/raw/public/sebi"
      format: "pdf"
    - name: "rbi_guidelines"
      path: "data/raw/public/rbi"
      format: "pdf"
    - name: "income_tax"
      path: "data/raw/public/income-tax"
      format: "html"
  custom:
    - name: "finance_qa"
      path: "data/raw/custom/qa-pairs"
      format: "jsonl"
    - name: "conversations"
      path: "data/raw/custom/conversations"
      format: "jsonl"
```

---

## 📦 Module Implementation Instructions

### `src/data/ingestion.py`
- Implement `DataIngestionPipeline` class
- Methods: `load_pdf()`, `load_html()`, `load_jsonl()`, `load_all()`
- Use `pypdf` for PDFs, `beautifulsoup4` for HTML
- Output: list of dicts with keys `{source, content, metadata}`
- Handle encoding errors gracefully

### `src/data/preprocessing.py`
- Implement `DataPreprocessor` class
- Convert raw documents to ChatML instruction format
- Methods: `clean_text()`, `to_chatml()`, `split_dataset()`, `save()`
- ChatML format:
```python
{
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```
- System prompt: "You are FinEdge, an expert personal finance advisor specializing in Indian personal finance including income tax, mutual funds, insurance, banking, and investments. Always provide accurate, actionable advice based on current Indian regulations."

### `slm/selection/compare.py`
- Implement `SLMComparator` class
- Load each candidate model via HuggingFace
- Evaluate on 50-question finance sample
- Measure: accuracy, tokens/sec, RAM usage
- Output markdown comparison table to `slm/selection/selection-report.md`

### `slm/fine-tuning/train.py`
- Use `SFTTrainer` from TRL library
- Load configs from `configs/training_config.yaml` and `configs/lora_config.yaml`
- Apply QLoRA via `BitsAndBytesConfig`
- Apply LoRA via `LoraConfig` from PEFT
- Save adapter checkpoints to `models/adapters/`
- Log metrics to W&B

### `slm/fine-tuning/merge_adapters.py`
- Load base model + LoRA adapters
- Use `model.merge_and_unload()` from PEFT
- Save merged model to `models/finedge-merged/`
- Verify model loads correctly after merge

### `edge/optimization/export_gguf.py`
- Use `llama.cpp` convert scripts
- Support quantization levels: Q4_K_M, Q5_K_M, Q8_0
- Verify exported model runs with llama-cpp-python
- Log model size before/after quantization

### `rag/ingestion/loader.py`
- Implement `DocumentLoader` class
- Support: PDF, HTML, plain text, JSONL
- Output: `List[Document]` with `page_content` and `metadata`

### `rag/ingestion/chunker.py`
- Implement `SemanticChunker` class
- Use `RecursiveCharacterTextSplitter` from LangChain
- Default: `chunk_size=512, chunk_overlap=64`
- Preserve source metadata through chunks

### `rag/ingestion/embedder.py`
- Implement `EmbeddingIndexer` class
- Use `sentence-transformers/all-MiniLM-L6-v2`
- Build FAISS index and save to `data/processed/rag-corpus/faiss_index`
- Support incremental index updates

### `rag/retrieval/retriever.py`
- Implement `HybridRetriever` class
- Combine FAISS vector search + BM25 keyword search
- Reciprocal Rank Fusion (RRF) for result merging
- Method: `retrieve(query: str, k: int) -> List[Document]`

### `rag/retrieval/reranker.py`
- Implement `CrossEncoderReranker` class
- Use `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Method: `rerank(query: str, docs: List[Document], top_n: int) -> List[Document]`

### `rag/pipeline.py`
- Implement `RAGPipeline` class
- Orchestrate: retrieve → rerank → format context
- Method: `query(user_query: str) -> str` (returns formatted context string)

### `agents/tools/tax_calculator.py`
- Implement `TaxCalculator` tool
- Support FY 2024-25 Indian tax slabs (old + new regime)
- Inputs: `annual_income`, `regime`, `deductions_80c`, `deductions_80d`
- Output: `{tax_payable, effective_rate, tax_breakdown}`
- New regime slabs (FY 2024-25):
  - 0–3L: 0%, 3–7L: 5%, 7–10L: 10%, 10–12L: 15%, 12–15L: 20%, >15L: 30%

### `agents/tools/sip_calculator.py`
- Implement `SIPCalculator` tool
- Methods: `calculate_sip_returns()`, `calculate_lumpsum()`, `calculate_step_up_sip()`
- Inputs: `monthly_amount`, `annual_rate`, `years`
- Output: `{total_invested, total_returns, maturity_amount, xirr}`

### `agents/tools/loan_advisor.py`
- Implement `LoanAdvisor` tool
- Methods: `calculate_emi()`, `compare_loans()`, `amortization_schedule()`
- Inputs: `principal`, `annual_rate`, `tenure_months`
- Output: `{emi, total_interest, total_payment, amortization_schedule}`

### `agents/planner/react_agent.py`
- Implement `FinanceReActAgent` class
- Follow ReAct (Reason + Act) pattern
- Parse `Thought:`, `Action:`, `Action Input:`, `Observation:` from LLM output
- Tool dispatch via registry pattern
- Max iterations from config

### `agents/memory/short_term.py`
- Implement `ConversationMemory` class
- Sliding window buffer of last N messages
- Method: `add_message()`, `get_context()`, `clear()`

### `agents/memory/long_term.py`
- Implement `UserProfileMemory` class
- Store: income_bracket, risk_appetite, financial_goals, preferred_language
- Persist to JSON file on disk
- Method: `update_profile()`, `load_profile()`, `get_profile_context()`

### `agents/orchestrator.py`
- Implement `FinEdgeOrchestrator` class
- Route query to: RAG-only / Agent-only / RAG+Agent
- Build final prompt with: system_prompt + user_profile + rag_context + tool_results + conversation_history
- Call edge SLM for final generation
- Stream response tokens

### `app/cli/chat.py`
- Rich terminal UI with streaming output
- Commands: `/clear`, `/profile`, `/tools`, `/help`, `/exit`
- Show tool calls and RAG sources when verbose mode on

### `app/api/main.py`
- FastAPI app with CORS
- Routes: `POST /chat`, `GET /health`, `POST /index`
- Streaming response via `StreamingResponse`
- Request/response schemas in `app/api/schemas.py`

### `app/ui/gradio_app.py`
- Gradio `ChatInterface` with streaming
- Show retrieved RAG sources in collapsible panel
- Show agent tool calls in separate panel
- Dark theme, mobile-friendly

---

## 🧪 Testing Requirements

For each module, create corresponding test in `tests/unit/`:

```python
# Example: tests/unit/test_tax_calculator.py
def test_new_regime_tax_basic():
    calc = TaxCalculator()
    result = calc.calculate(annual_income=1000000, regime="new")
    assert result["tax_payable"] == 112500  # Expected for 10L new regime

def test_old_regime_with_deductions():
    calc = TaxCalculator()
    result = calc.calculate(
        annual_income=1000000,
        regime="old",
        deductions_80c=150000,
        deductions_80d=25000
    )
    assert result["tax_payable"] < 112500  # Less than new regime
```

---

## 🚫 Important Constraints

1. **No cloud inference** — all inference via `llama-cpp-python` or `onnxruntime` locally
2. **No hardcoded values** — all hyperparameters in `configs/*.yaml`
3. **No `print()` statements** — use `loguru` logger only
4. **Indian tax slabs** must be for FY 2024-25 (updated)
5. **SEBI/RBI data** must cite source and date in metadata
6. **Hallucination guard** — always include source citations in RAG responses
7. **Hindi support** — keep Devanagari encoding in mind for text processing

---

## 📝 Prompts to Use

### System Prompt (`agents/prompts/system_prompt.txt`)
```
You are FinEdge, an expert personal finance advisor specializing in Indian personal finance. You have deep knowledge of:
- Indian income tax (IT Act, 80C, 80D, new vs old regime)
- Investment products (mutual funds, SIP, ELSS, PPF, NPS, FD, bonds)
- SEBI and RBI regulations
- Insurance (term, health, ULIP)
- Banking (loans, EMI, CIBIL score)
- NSE/BSE markets

Always:
- Give specific, actionable advice for the Indian context
- Cite relevant sections (e.g., "Under Section 80C...")
- Mention current limits and thresholds (FY 2024-25)
- Clarify if regulations may have changed
- Recommend consulting a SEBI-registered advisor for large investments

Never:
- Hallucinate specific returns or guaranteed profits
- Give advice without knowing user's tax bracket and risk profile
- Recommend specific stocks or mutual fund schemes by name
```

### Tool Use Prompt (`agents/prompts/tool_use_prompt.txt`)
```
You have access to the following tools:
{tool_descriptions}

To use a tool, respond in this format:
Thought: [your reasoning about what to do]
Action: [tool_name]
Action Input: [JSON input for the tool]

After receiving the tool result (Observation), continue reasoning.
When you have enough information, respond with:
Final Answer: [your complete response to the user]
```

---

## 🔄 Implementation Order for Claude Code

Follow this sequence to avoid import errors and dependency issues:

1. `configs/` — all YAML config files
2. `src/data/ingestion.py` + `preprocessing.py`
3. `slm/fine-tuning/dataset_loader.py`
4. `slm/fine-tuning/train.py` + `merge_adapters.py`
5. `benchmarks/metrics/` — all metric modules
6. `benchmarks/runners/` — benchmark runners
7. `edge/optimization/` — quantization scripts
8. `edge/runtimes/llama_cpp/run_inference.py`
9. `rag/ingestion/` — loader, chunker, embedder
10. `rag/vector_store/` — FAISS and Chroma stores
11. `rag/retrieval/` — retriever, reranker, query expander
12. `rag/pipeline.py`
13. `agents/tools/` — all tool implementations
14. `agents/memory/` — short and long term memory
15. `agents/planner/react_agent.py`
16. `agents/orchestrator.py`
17. `app/api/` — FastAPI server
18. `app/cli/chat.py`
19. `app/ui/gradio_app.py`
20. `tests/` — unit and integration tests
