#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY'
import sys
if not (sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12):
    raise SystemExit(
        f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported. "
        "Use Python 3.10, 3.11, or 3.12."
    )
PY

if [ ! -d ".venv" ]; then
    "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --timeout 120 --retries 10 "numpy<2.0.0"
python -m pip install --timeout 120 --retries 10 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --timeout 120 --retries 10 -r requirements.txt
python -m spacy download en_core_web_sm
python -m playwright install --with-deps chromium

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
fi

echo "Setup complete. Run: bash scripts/run_linux.sh"
