#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PAGEINDEX_REPO_DIR="${BANYANTREE_PAGEINDEX_REPO_DIR:-external/PageIndex}"

if [ ! -d "$PAGEINDEX_REPO_DIR" ]; then
    git clone https://github.com/VectifyAI/PageIndex.git "$PAGEINDEX_REPO_DIR"
fi

python3 -m pip install --user --upgrade pip

# PageIndex currently has a conflicting python-dotenv pin in requirements.txt:
# litellm==1.83.7 requires python-dotenv==1.0.1, while the repo pins 1.2.2.
# Install a compatible set explicitly for Kaggle/offline indexing.
python3 -m pip install --user \
    "litellm==1.83.7" \
    "python-dotenv==1.0.1" \
    "pymupdf==1.26.4" \
    "PyPDF2==3.0.1" \
    "pyyaml==6.0.2"

echo "PageIndex Kaggle dependencies installed."
echo "Next: python3 scripts/run_pageindex_indexing.py"
