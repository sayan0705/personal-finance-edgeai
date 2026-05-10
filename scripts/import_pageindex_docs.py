from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pageindex_adapter import PageIndexAdapter


DATA_DIR = Path(os.environ.get("BANYANTREE_DATA_DIR", PROJECT_ROOT / "data")).resolve()
FINANCIAL_KG_ROOT = Path(os.environ.get("BANYANTREE_FINANCIAL_KG_ROOT", DATA_DIR / "financial_kg")).resolve()
RAW_DOCS_DIR = Path(os.environ.get("BANYANTREE_RAW_DOCS_DIR", FINANCIAL_KG_ROOT / "raw_docs")).resolve()
PAGEINDEX_INPUT_DIR = Path(os.environ.get("BANYANTREE_PAGEINDEX_INPUT_DIR", FINANCIAL_KG_ROOT / "pageindex" / "inputs")).resolve()
PAGEINDEX_OUTPUT_DIR = Path(os.environ.get("BANYANTREE_PAGEINDEX_OUTPUT_DIR", FINANCIAL_KG_ROOT / "pageindex" / "outputs")).resolve()
PAGEINDEX_STRUCTURES_DIR = Path(os.environ.get("BANYANTREE_PAGEINDEX_STRUCTURES_DIR", FINANCIAL_KG_ROOT / "pageindex" / "structures")).resolve()
PAGEINDEX_FLATTENED_DOCS_PATH = Path(
    os.environ.get("BANYANTREE_PAGEINDEX_FLATTENED_DOCS_PATH", RAW_DOCS_DIR / "pageindex" / "pageindex_flattened_docs.json")
).resolve()


def main() -> int:
    adapter = PageIndexAdapter(
        input_dir=PAGEINDEX_INPUT_DIR,
        output_dir=PAGEINDEX_OUTPUT_DIR,
        structures_dir=PAGEINDEX_STRUCTURES_DIR,
    )
    flattened = []
    structure_paths = sorted(adapter.structures_dir.glob("*.json"))

    for structure_path in structure_paths:
        docs = adapter.flatten_structure_file(structure_path)
        for doc in docs:
            doc.setdefault("structure_path", str(structure_path))
        flattened.extend(docs)

    PAGEINDEX_FLATTENED_DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PAGEINDEX_FLATTENED_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(flattened, f, indent=2, ensure_ascii=False)

    print(f"Imported {len(flattened)} flattened PageIndex docs from {len(structure_paths)} structure file(s)")
    print(f"Wrote: {PAGEINDEX_FLATTENED_DOCS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
