# ============================================================
# personal-finance-edge-ai Makefile
# Edge AI Personal Finance Assistant
# ============================================================

.PHONY: help setup setup-gpu setup-edge data finetune merge benchmark \
        quantize rag-index serve ui test lint format clean

# ── Default ─────────────────────────────────────────────────
.DEFAULT_GOAL := help

# ── Variables ───────────────────────────────────────────────
PYTHON       := python3
PIP          := pip3
MODEL_DIR    := models
DATA_DIR     := data
CONFIG_DIR   := configs
LOG_DIR      := logs

TRAIN_CONFIG    := $(CONFIG_DIR)/training_config.yaml
LORA_CONFIG     := $(CONFIG_DIR)/lora_config.yaml
DATASET_CONFIG  := $(CONFIG_DIR)/dataset_config.yaml
APP_CONFIG      := $(CONFIG_DIR)/app_config.yaml

BASE_MODEL      ?= Qwen/Qwen2-1.5B-Instruct
QUANTIZATION    ?= Q4_K_M
EDGE_MODEL      ?= $(MODEL_DIR)/finedge-$(QUANTIZATION).gguf
PORT            ?= 8000

# ── Help ─────────────────────────────────────────────────────
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════╗"
	@echo "║          personal-finance-edge-ai — Make Commands              ║"
	@echo "╚══════════════════════════════════════════════════╝"
	@echo ""
	@echo "  SETUP"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make setup           Install general dependencies"
	@echo "  make setup-gpu       Install GPU fine-tuning deps"
	@echo "  make setup-edge      Install edge runtime deps"
	@echo ""
	@echo "  DATA"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make data            Download + process all datasets"
	@echo "  make data-public     Download public Indian finance datasets"
	@echo "  make data-process    Process raw data into training format"
	@echo "  make rag-index       Build RAG vector index from corpus"
	@echo ""
	@echo "  MODEL"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make slm-compare     Run SLM candidate comparison"
	@echo "  make finetune        Run LoRA/QLoRA fine-tuning"
	@echo "  make merge           Merge LoRA adapters into base model"
	@echo "  make quantize        Export quantized edge model (GGUF)"
	@echo "  make quantize-onnx   Export ONNX model for ORT inference"
	@echo ""
	@echo "  EVALUATION"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make benchmark       Run all benchmarks"
	@echo "  make bench-public    Run public finance benchmarks"
	@echo "  make bench-custom    Run custom India finance benchmark"
	@echo "  make bench-edge      Run edge performance profiling"
	@echo ""
	@echo "  APP"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make serve           Start FastAPI server"
	@echo "  make ui              Launch Gradio UI"
	@echo "  make chat            Start CLI chat session"
	@echo ""
	@echo "  QUALITY"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make test            Run all tests"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make test-integration  Run integration tests"
	@echo "  make lint            Run linters (ruff, mypy)"
	@echo "  make format          Auto-format code (black, isort)"
	@echo ""
	@echo "  UTILS"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make clean           Remove generated artifacts"
	@echo "  make clean-data      Remove processed data"
	@echo "  make clean-models    Remove downloaded models"
	@echo ""

# ── Setup ────────────────────────────────────────────────────
setup:
	@echo "📦 Installing general dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Setup complete."

setup-gpu:
	@echo "🖥️  Installing GPU fine-tuning dependencies..."
	$(PIP) install -r requirements-gpu.txt
	@echo "✅ GPU setup complete."

setup-edge:
	@echo "⚡ Installing edge runtime dependencies..."
	$(PIP) install -r requirements-edge.txt
	@echo "✅ Edge setup complete."

# ── Data ─────────────────────────────────────────────────────
data: data-public data-process
	@echo "✅ Dataset pipeline complete."

data-public:
	@echo "📥 Downloading public Indian finance datasets..."
	@mkdir -p $(DATA_DIR)/raw/public
	bash scripts/download_datasets.sh
	@echo "✅ Public datasets downloaded."

data-process:
	@echo "⚙️  Processing raw data into training format..."
	@mkdir -p $(DATA_DIR)/processed/train $(DATA_DIR)/processed/eval
	$(PYTHON) src/data/preprocessing.py \
		--input $(DATA_DIR)/raw \
		--output $(DATA_DIR)/processed \
		--config $(DATASET_CONFIG)
	@echo "✅ Data processing complete."

rag-index:
	@echo "🗂️  Building RAG vector index..."
	$(PYTHON) rag/ingestion/loader.py --source $(DATA_DIR)/processed/rag-corpus
	$(PYTHON) rag/ingestion/chunker.py
	$(PYTHON) rag/ingestion/embedder.py
	@echo "✅ RAG index built."

# ── SLM Selection ─────────────────────────────────────────────
slm-compare:
	@echo "🔍 Running SLM candidate comparison..."
	$(PYTHON) slm/selection/compare.py \
		--config $(CONFIG_DIR)/app_config.yaml \
		--output slm/selection/selection-report.md
	@echo "✅ SLM comparison complete. See slm/selection/selection-report.md"

# ── Fine-Tuning ───────────────────────────────────────────────
finetune:
	@echo "🔥 Starting LoRA/QLoRA fine-tuning..."
	@mkdir -p $(LOG_DIR) $(MODEL_DIR)
	$(PYTHON) slm/fine-tuning/train.py \
		--train-config $(TRAIN_CONFIG) \
		--lora-config $(LORA_CONFIG) \
		--dataset-config $(DATASET_CONFIG) \
		--base-model $(BASE_MODEL) \
		--output-dir $(MODEL_DIR)/adapters
	@echo "✅ Fine-tuning complete. Adapter saved to $(MODEL_DIR)/adapters"

merge:
	@echo "🔗 Merging LoRA adapters into base model..."
	$(PYTHON) slm/fine-tuning/merge_adapters.py \
		--base-model $(BASE_MODEL) \
		--adapter-path $(MODEL_DIR)/adapters \
		--output-path $(MODEL_DIR)/finedge-merged
	@echo "✅ Merged model saved to $(MODEL_DIR)/finedge-merged"

# ── Edge Optimization ────────────────────────────────────────
quantize:
	@echo "⚡ Quantizing model to GGUF ($(QUANTIZATION))..."
	$(PYTHON) edge/optimization/export_gguf.py \
		--model-path $(MODEL_DIR)/finedge-merged \
		--output $(EDGE_MODEL) \
		--quantization $(QUANTIZATION)
	@echo "✅ Edge model saved: $(EDGE_MODEL)"

quantize-onnx:
	@echo "⚡ Exporting to ONNX + INT8 quantization..."
	$(PYTHON) edge/optimization/export_onnx.py \
		--model-path $(MODEL_DIR)/finedge-merged \
		--output $(MODEL_DIR)/finedge-onnx
	@echo "✅ ONNX model saved to $(MODEL_DIR)/finedge-onnx"

profile-edge:
	@echo "📊 Profiling edge inference performance..."
	$(PYTHON) edge/profiler/profile_edge.py \
		--model $(EDGE_MODEL) \
		--runtime llama_cpp
	@echo "✅ Edge profiling complete."

# ── Benchmarking ─────────────────────────────────────────────
benchmark: bench-public bench-custom bench-edge
	@echo "✅ All benchmarks complete. See benchmarks/reports/"

bench-public:
	@echo "📐 Running public finance benchmarks..."
	@mkdir -p benchmarks/results/fine-tuned
	$(PYTHON) benchmarks/runners/run_public_bench.py \
		--model $(EDGE_MODEL) \
		--output benchmarks/results/fine-tuned/public_results.json
	@echo "✅ Public benchmark complete."

bench-custom:
	@echo "📐 Running custom Indian finance benchmark..."
	$(PYTHON) benchmarks/runners/run_custom_bench.py \
		--model $(EDGE_MODEL) \
		--dataset $(DATA_DIR)/benchmarks/custom-india-bench \
		--output benchmarks/results/fine-tuned/custom_results.json
	@echo "✅ Custom benchmark complete."

bench-edge:
	@echo "📐 Running edge performance benchmarks..."
	$(PYTHON) benchmarks/runners/run_edge_bench.py \
		--model $(EDGE_MODEL) \
		--output benchmarks/results/fine-tuned/edge_perf.json
	@echo "✅ Edge performance benchmark complete."

# ── App ──────────────────────────────────────────────────────
serve:
	@echo "🚀 Starting FastAPI server on port $(PORT)..."
	uvicorn app.api.main:app \
		--host 0.0.0.0 \
		--port $(PORT) \
		--reload

ui:
	@echo "🎨 Launching Gradio UI..."
	$(PYTHON) app/ui/gradio_app.py

chat:
	@echo "💬 Starting CLI chat..."
	$(PYTHON) app/cli/chat.py \
		--model $(EDGE_MODEL) \
		--rag \
		--config $(APP_CONFIG)

# ── Tests ─────────────────────────────────────────────────────
test: test-unit test-integration
	@echo "✅ All tests passed."

test-unit:
	@echo "🧪 Running unit tests..."
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

test-integration:
	@echo "🧪 Running integration tests..."
	$(PYTHON) -m pytest tests/integration/ -v --tb=short

# ── Lint & Format ────────────────────────────────────────────
lint:
	@echo "🔍 Running linters..."
	ruff check src/ rag/ agents/ app/ benchmarks/ edge/
	mypy src/ --ignore-missing-imports
	@echo "✅ Lint complete."

format:
	@echo "✨ Formatting code..."
	black src/ rag/ agents/ app/ benchmarks/ edge/ tests/
	isort src/ rag/ agents/ app/ benchmarks/ edge/ tests/
	@echo "✅ Format complete."

# ── Clean ─────────────────────────────────────────────────────
clean: clean-pyc
	@echo "🧹 Cleaned build artifacts."

clean-pyc:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

clean-data:
	@echo "⚠️  Removing processed data..."
	rm -rf $(DATA_DIR)/processed/
	@echo "✅ Processed data removed."

clean-models:
	@echo "⚠️  Removing downloaded model artifacts..."
	rm -rf $(MODEL_DIR)/
	@echo "✅ Model artifacts removed."

# ── Full Pipeline ────────────────────────────────────────────
pipeline: setup data finetune merge quantize rag-index benchmark
	@echo ""
	@echo "╔══════════════════════════════════════════════════╗"
	@echo "║    ✅  Full pipeline complete!                   ║"
	@echo "║    Run 'make chat' to start the assistant        ║"
	@echo "╚══════════════════════════════════════════════════╝"
