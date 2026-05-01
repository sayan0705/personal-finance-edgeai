# 🏗️ System Architecture — personal-finance-edge-ai

> This document describes the complete system design, data flow, component interactions, and technical decisions for the personal-finance-edge-ai edge AI personal finance assistant.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                         │
│              CLI Chat │ FastAPI Server │ Gradio UI                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                          │
│                    agents/orchestrator.py                           │
│         (Routes query → RAG and/or Agent and/or Direct SLM)        │
└────────────┬─────────────────────────┬──────────────────────────────┘
             │                         │
             ▼                         ▼
┌────────────────────┐     ┌───────────────────────────┐
│    RAG PIPELINE    │     │      AGENTIC WORKFLOW      │
│  rag/pipeline.py   │     │   agents/orchestrator.py  │
│                    │     │                            │
│  • Retriever       │     │  • ReAct Agent             │
│  • Reranker        │     │  • Tool Executor           │
│  • Query Expander  │     │  • Memory (short + long)   │
└────────┬───────────┘     └────────────┬───────────────┘
         │                              │
         └──────────────┬───────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EDGE SLM INFERENCE ENGINE                        │
│                   edge/runtimes/llama_cpp/                          │
│              (Fine-tuned + Quantized SLM on-device)                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End ML Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│              │    │              │    │              │    │              │
│   DATASET    │───▶│     SLM      │───▶│  FINE-TUNE   │───▶│  BENCHMARK   │
│  CURATION    │    │  SELECTION   │    │  (LoRA/QLoRA)│    │  EVALUATION  │
│              │    │              │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                    │
                    ┌───────────────────────────────────────────────┘
                    ▼
             ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
             │              │    │              │    │              │
             │    EDGE      │───▶│     RAG      │───▶│   AGENTIC    │
             │ OPTIMIZATION │    │   PIPELINE   │    │  WORKFLOW    │
             │ (GGUF/ONNX)  │    │              │    │              │
             └──────────────┘    └──────────────┘    └──────────────┘
```

---

## 3. Component Deep Dive

### 3.1 Dataset Curation

```
data/
├── raw/public/
│   ├── sebi/        ← PDFs, circulars scraped from sebi.gov.in
│   ├── rbi/         ← Monetary policy, guidelines from rbi.org.in
│   ├── income-tax/  ← IT slabs, 80C rules from incometax.gov.in
│   └── nse-bse/     ← Market data, company info
└── raw/custom/
    ├── qa-pairs/    ← Manually crafted finance Q&A (Hindi + English)
    ├── conversations/ ← Synthetic multi-turn finance conversations
    └── scenarios/   ← Real-world Indian finance scenarios
```

**Processing Pipeline:**
1. Raw document ingestion (PDF, HTML, text)
2. Cleaning and deduplication
3. Instruction-format conversion (Alpaca / ChatML)
4. Train / eval / test split (80/10/10)
5. Tokenization and dataset card generation

**Dataset Format (ChatML):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a personal finance advisor specializing in Indian finance..."},
    {"role": "user", "content": "What is the tax benefit under Section 80C?"},
    {"role": "assistant", "content": "Under Section 80C of the Income Tax Act, you can claim deductions up to ₹1.5 lakh per financial year..."}
  ]
}
```

---

### 3.2 SLM Selection

**Evaluation Criteria:**

| Criterion | Weight | Notes |
|---|---|---|
| Inference speed (tokens/sec) | 25% | Measured on target edge device |
| RAM footprint (MB) | 20% | Post-quantization |
| Finance domain accuracy | 30% | On custom India finance eval set |
| Multilingual support (Hindi) | 15% | Code-switching capability |
| Fine-tuning ease | 10% | PEFT compatibility |

**Candidate Models:**

| Model | HuggingFace ID |
|---|---|
| Phi-3 Mini | `microsoft/Phi-3-mini-4k-instruct` |
| Gemma 2B-IT | `google/gemma-2b-it` |
| Qwen2-1.5B | `Qwen/Qwen2-1.5B-Instruct` |
| SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| TinyLlama-1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

---

### 3.3 Fine-Tuning Architecture

```
Base SLM (frozen weights)
        │
        ▼
  LoRA Adapters injected into:
  • Attention Q, K, V, O projections
  • MLP layers (optional)
        │
        ▼
  QLoRA (4-bit NF4 quantization of base)
  + LoRA adapters trained in fp16/bf16
        │
        ▼
  Adapter merge → Full fine-tuned model
```

**Training Config:**
```yaml
# configs/lora_config.yaml
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
bias: "none"
task_type: "CAUSAL_LM"

# QLoRA
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_compute_dtype: "bfloat16"
```

---

### 3.4 Benchmarking Framework

**Two-track evaluation:**

```
Track 1: Public Benchmarks
  └── FinanceBench (public finance QA)
  └── MMLU Finance subset
  └── Standard perplexity on held-out finance corpus

Track 2: Custom Indian Finance Benchmark
  └── Tax calculation accuracy
  └── Regulatory knowledge (SEBI/RBI)
  └── Product comparison (MF, insurance, FD)
  └── Hindi-English code-switch accuracy
  └── Hallucination rate on India-specific claims
```

**Edge Performance Metrics:**
- Tokens per second (TPS)
- Time to first token (TTFT)
- Peak RAM usage (MB)
- CPU/GPU utilization (%)
- Power consumption (mW) — for battery-powered devices

---

### 3.5 Edge Optimization Pipeline

```
Fine-tuned Model (fp16/bf32)
        │
        ├──▶ GGUF Export (llama.cpp)
        │       ├── Q4_K_M  (recommended, best quality/size tradeoff)
        │       ├── Q5_K_M  (higher quality)
        │       └── Q8_0    (near lossless)
        │
        ├──▶ ONNX Export + ORT Quantization
        │       └── INT8 dynamic quantization
        │
        └──▶ MLX Format (Apple Silicon Macs)
```

**Target Edge Devices:**

| Device | Runtime | Quantization | Expected TPS |
|---|---|---|---|
| Raspberry Pi 4 (8GB) | llama.cpp | Q4_K_M | ~3-8 |
| NVIDIA Jetson Nano | llama.cpp + CUDA | Q4_K_M | ~15-25 |
| MacBook (Apple Silicon) | MLX | fp16 | ~30-60 |
| Android (8GB RAM) | llama.cpp via JNI | Q4_K_M | ~5-10 |
| PC/Laptop (CPU-only) | llama.cpp | Q4_K_M | ~10-20 |

---

### 3.6 RAG Pipeline

```
INGESTION (Offline)                    RETRIEVAL (Online)
──────────────────────                 ─────────────────────────
Raw Docs (PDF/HTML/Text)               User Query
        │                                      │
        ▼                                      ▼
   Document Loader                      Query Expansion
   (rag/ingestion/loader.py)            (finance terminology)
        │                                      │
        ▼                                      ▼
  Semantic Chunking                    Vector Similarity Search
  (~512 tokens, overlap 64)            (FAISS / ChromaDB)
        │                                      │
        ▼                                      ▼
  Embedding Model                       BM25 Keyword Search
  (edge-optimized, e.g.                        │
   all-MiniLM-L6-v2)                           ▼
        │                               Hybrid Fusion (RRF)
        ▼                                      │
  Vector Store Index                           ▼
  (FAISS local file)                    Cross-Encoder Reranking
                                               │
                                               ▼
                                        Top-K Chunks → SLM Context
```

**RAG Configuration:**
```yaml
chunk_size: 512
chunk_overlap: 64
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
vector_store: "faiss"  # or "chroma"
retrieval_k: 10
rerank_top_n: 3
hybrid_search: true
bm25_weight: 0.3
vector_weight: 0.7
```

---

### 3.7 Agentic Workflow

**ReAct Pattern:**
```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│              ReAct Agent                │
│                                         │
│  Thought: Analyze query intent          │
│  Action: Select appropriate tool        │
│  Observation: Get tool result           │
│  Thought: Evaluate result               │
│  ... (repeat until answer ready)        │
│  Final Answer: Synthesize response      │
└─────────────────────────────────────────┘
    │              │              │
    ▼              ▼              ▼
tax_calculator  sip_calc    stock_fetcher
loan_advisor  budget_analyzer  rag_search
```

**Memory Architecture:**
```
Short-term Memory:
  └── Sliding window conversation buffer (last N turns)
  └── Stored in-process (list of messages)

Long-term Memory:
  └── User profile (risk appetite, income bracket, goals)
  └── Stored as JSON on-device
  └── Retrieved at session start
```

---

## 4. Data Flow — Query to Response

```
1. User types: "Should I invest in PPF or ELSS for tax saving?"

2. Orchestrator classifies query:
   → Needs RAG (regulatory knowledge)
   → Needs Agent tool (tax_calculator, sip_calculator)

3. RAG retrieves:
   → PPF rules (from RBI/Finance Ministry docs)
   → ELSS fund details (from SEBI circulars)
   → 80C deduction rules (from income-tax corpus)

4. Agent executes:
   → tax_calculator(income=user_profile.income, regime="new")
   → sip_calculator(amount=150000, years=3, rate=12.0)

5. SLM generates response:
   → Context = [RAG chunks] + [Tool outputs] + [Conversation history]
   → Prompt = system_prompt + context + user_query
   → Inference via llama.cpp (on-device, Q4_K_M)

6. Response streamed to user interface
```

---

## 5. Security & Privacy Design

| Concern | Solution |
|---|---|
| Financial data leakage | All inference on-device, zero cloud calls |
| User profile storage | Local encrypted JSON |
| Model weights | Stored locally, no telemetry |
| API server (optional) | Local network only, no external exposure |
| Sensitive query logging | Disabled by default |

---

## 6. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Fine-tuning | HuggingFace PEFT + TRL | Industry standard LoRA/QLoRA |
| Edge inference | llama.cpp | Best CPU performance, GGUF support |
| Embedding | sentence-transformers | Lightweight, edge-compatible |
| Vector store | FAISS (edge) / ChromaDB (dev) | FAISS: no server needed |
| Agent framework | LangChain / custom ReAct | Flexibility + tool integration |
| API | FastAPI | Async, lightweight |
| UI | Gradio | Rapid prototyping |
| Benchmarking | lm-evaluation-harness | Standard eval framework |
| Experiment tracking | MLflow / Weights & Biases | Fine-tuning experiment logging |

---

## 7. Key Design Decisions

**Why Edge (not Cloud)?**
- User financial data is sensitive — must not leave the device
- Works offline — no internet dependency
- Lower latency — no network round-trip
- Cost-free inference after deployment

**Why SLM (not LLM)?**
- Edge devices have limited RAM (4-8GB)
- SLMs at 1-4B params can be quantized to fit
- Domain fine-tuning closes the quality gap for specific tasks

**Why LoRA/QLoRA?**
- Full fine-tuning of SLMs requires significant GPU memory
- LoRA trains <1% of parameters, retains base model knowledge
- QLoRA enables fine-tuning on consumer GPUs (16GB VRAM)

**Why RAG + Fine-tuning (not just one)?**
- Fine-tuning: teaches the model *how to reason* about Indian finance
- RAG: provides *current, accurate facts* (regulations change frequently)
- Together: reliable, grounded, context-aware responses

---

## 8. Future Extensions

- Voice interface (Whisper on-device for STT)
- Multilingual support (Hindi-first fine-tuning)
- Document understanding (upload salary slip, investment statement)
- Portfolio tracker integration
- Local notification agent (SIP reminder, tax deadline alerts)
