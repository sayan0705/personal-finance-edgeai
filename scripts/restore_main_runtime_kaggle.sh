#!/usr/bin/env bash
set -euo pipefail

# Restore packages that PageIndex/LiteLLM may downgrade or upgrade in ways that
# break the main BanyanTree runtime on Kaggle.
python3 -m pip install --user \
    "tokenizers==0.19.1" \
    "click>=8.2.1" \
    "pydantic<=2.12.3,>=2.0" \
    "python-dotenv==1.2.2"

echo "Main Kaggle runtime compatibility packages restored."
