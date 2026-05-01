# 🧠 personal-finance-edge-ai

> **Personal Finance Edge AI Assistant** — A domain-fine-tuned Small Language Model (SLM) running on-device with RAG and agentic workflows, purpose-built for Indian personal finance.

---

## 📌 Project Overview

`personal-finance-edge-ai` is an end-to-end pipeline to build, fine-tune, benchmark, and deploy a privacy-first AI financial assistant that runs entirely on edge devices — no cloud, no data leakage.

Built specifically for the **Indian personal finance context**: income tax slabs, SEBI regulations, RBI guidelines, SIP/mutual funds, NSE/BSE, EPF, PPF, NPS, and more.

---

## 🏗️ Pipeline Architecture

```
Dataset Curation → SLM Selection → Fine-Tuning → Benchmarking → Edge Optimization → RAG + Agents → App
```

| Stage | Description | Key Tools |
|---|---|---|
| **Dataset Curation** | Public (SEBI, RBI, IT) + custom Indian finance Q&A | Python, Pandas, LabelStudio |
| **SLM Selection** | Evaluate Phi-3, Gemma, Qwen2, SmolLM2, TinyLlama | HuggingFace, Ollama |
| **Fine-Tuning** | LoRA / QLoRA on Indian finance dataset | PEFT, TRL, bitsandbytes |
| **Benchmarking** | Public + custom Indian finance benchmarks | EleutherAI lm-eval |
| **Edge Optimization** | INT4/INT8 quantization, GGUF/ONNX export | llama.cpp, ONNX Runtime |
| **RAG Pipeline** | Semantic chunking + local vector store | LangChain, FAISS, ChromaDB |
| **Agentic Workflow** | ReAct agent with finance tools | LangChain Agents, custom tools |
| **App** | CLI + FastAPI + Gradio UI | FastAPI, Gradio |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for fine-tuning, optional)
- 8GB+ RAM (edge inference)
- Git LFS (for model artifacts)

### Setup

```bash
# Clone the repo
git clone https://github.com/your-username/personal-finance-edge-ai.git
cd personal-finance-edge-ai

# Setup environment
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# Install dependencies
pip install -r requirements.txt          # General
pip install -r requirements-gpu.txt     # For fine-tuning (GPU required)
pip install -r requirements-edge.txt    # For edge deployment
```

### Run the Assistant (CLI)

```bash
python app/cli/chat.py --model models/finedge-q4.gguf --rag
```

### Run the API Server

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

### Launch Gradio UI

```bash
python app/ui/gradio_app.py
```

---

## 📁 Repository Structure

```
personal-finance-edge-ai/
├── data/                     # Datasets (raw, processed, benchmarks)
├── slm/                      # SLM selection, fine-tuning, registry
├── benchmarks/               # Benchmark runners, metrics, results
├── edge/                     # Quantization, export, edge runtimes
├── rag/                      # RAG ingestion, vector store, retrieval
├── agents/                   # Tools, planner, memory, orchestrator
├── app/                      # CLI, FastAPI, Gradio interfaces
├── tests/                    # Unit and integration tests
├── notebooks/                # Exploration and experiment notebooks
├── scripts/                  # Shell scripts for pipeline automation
├── configs/                  # App and logging configs
└── docs/                     # Architecture, design, and guides
```

> See [docs/architecture.md](docs/architecture.md) for detailed system design.

---

## 🤖 Supported SLM Candidates

| Model | Params | Context | Edge-Friendly | Multilingual |
|---|---|---|---|---|
| **Phi-3 Mini** | 3.8B | 128K | ✅ | ⚠️ |
| **Gemma 2B-IT** | 2B | 8K | ✅ | ✅ |
| **Qwen2-1.5B** | 1.5B | 32K | ✅ | ✅ Hindi |
| **SmolLM2-1.7B** | 1.7B | 8K | ✅✅ | ⚠️ |
| **TinyLlama-1.1B** | 1.1B | 2K | ✅✅ | ⚠️ |

> Final selection documented in [slm/selection/selection-report.md](slm/selection/selection-report.md)

---

## 🇮🇳 Indian Finance Domain Coverage

- **Tax**: Income tax slabs, 80C/80D deductions, TDS, ITR filing
- **Investments**: Mutual funds, SIP, ELSS, PPF, NPS, FD, bonds
- **Regulations**: SEBI circulars, RBI monetary policy, IRDAI
- **Banking**: Loan types, EMI calculation, credit score, CIBIL
- **Markets**: NSE/BSE basics, indices, stock fundamentals
- **Insurance**: Term, health, ULIP comparison

---

## 🔁 Makefile Commands

```bash
make setup          # Install all dependencies
make data           # Download and process datasets
make finetune       # Run fine-tuning pipeline
make benchmark      # Run all benchmarks
make quantize       # Export quantized edge model
make rag-index      # Build RAG vector index
make serve          # Start API server
make test           # Run all tests
make lint           # Run code linters
```

---

## 📊 Benchmarks

| Benchmark | Baseline (Base SLM) | Fine-tuned | Delta |
|---|---|---|---|
| FinanceBench (public) | TBD | TBD | TBD |
| Custom India Finance Bench | TBD | TBD | TBD |
| Edge Latency (tokens/sec) | TBD | TBD | TBD |
| RAM Usage (MB) | TBD | TBD | TBD |

> Full results in [benchmarks/reports/benchmark_report.md](benchmarks/reports/benchmark_report.md)

---

## 🧩 Agentic Tools

| Tool | Description |
|---|---|
| `tax_calculator` | Income tax computation with Indian slabs |
| `sip_calculator` | SIP returns, CAGR, step-up SIP |
| `budget_analyzer` | Monthly income/expense breakdown |
| `loan_advisor` | EMI calculator, loan type comparisons |
| `stock_fetcher` | NSE/BSE live data via public APIs |

---

## 🔒 Privacy & Edge-First Design

- ✅ All inference runs **on-device** — no data sent to cloud
- ✅ Vector store is **local** (FAISS / ChromaDB)
- ✅ No external API calls during inference
- ✅ User financial data stays on the device

---

## 🗺️ Roadmap

- [ ] Dataset curation (public + custom)
- [ ] SLM selection & comparison
- [ ] Fine-tuning with LoRA/QLoRA
- [ ] Custom Indian finance benchmark
- [ ] Edge quantization (GGUF INT4)
- [ ] RAG pipeline with FAISS
- [ ] ReAct agent with finance tools
- [ ] CLI app
- [ ] FastAPI server
- [ ] Gradio demo UI
- [ ] Android / Raspberry Pi deployment

---

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines and open an issue before submitting a PR.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co) for model hub and PEFT
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for edge inference
- [LangChain](https://langchain.com) for RAG and agent frameworks
- SEBI, RBI, Income Tax India for public financial data
