#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x ".venv/bin/python" ]; then
    echo "Local virtual environment not found. Run setup first:"
    echo "  bash scripts/setup_linux.sh"
    exit 1
fi

set -a
if [ -f ".env" ]; then
    # shellcheck disable=SC1091
    source .env
fi
set +a

export PYTHONIOENCODING=utf-8

.venv/bin/python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

.venv/bin/python src/banyanTreev3_agentic.py
