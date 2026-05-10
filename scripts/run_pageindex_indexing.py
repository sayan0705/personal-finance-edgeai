from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pageindex_adapter import PageIndexAdapter


DATA_DIR = Path(os.environ.get("BANYANTREE_DATA_DIR", PROJECT_ROOT / "data")).resolve()
FINANCIAL_KG_ROOT = Path(os.environ.get("BANYANTREE_FINANCIAL_KG_ROOT", DATA_DIR / "financial_kg")).resolve()
PAGEINDEX_INPUT_DIR = Path(os.environ.get("BANYANTREE_PAGEINDEX_INPUT_DIR", FINANCIAL_KG_ROOT / "pageindex" / "inputs")).resolve()
PAGEINDEX_OUTPUT_DIR = Path(os.environ.get("BANYANTREE_PAGEINDEX_OUTPUT_DIR", FINANCIAL_KG_ROOT / "pageindex" / "outputs")).resolve()
PAGEINDEX_STRUCTURES_DIR = Path(os.environ.get("BANYANTREE_PAGEINDEX_STRUCTURES_DIR", FINANCIAL_KG_ROOT / "pageindex" / "structures")).resolve()
PAGEINDEX_MODEL = os.environ.get("BANYANTREE_PAGEINDEX_MODEL", "gpt-4o-mini")
PAGEINDEX_RUN_MODEL = PAGEINDEX_MODEL.removeprefix("anthropic/")
PAGEINDEX_REPO_DIR = Path(os.environ.get("BANYANTREE_PAGEINDEX_REPO_DIR", PROJECT_ROOT / "external" / "PageIndex")).resolve()


def _pageindex_runner() -> Path | None:
    runner = PAGEINDEX_REPO_DIR / "run_pageindex.py"
    return runner if runner.exists() else None


def _print_manual_instructions(adapter: PageIndexAdapter, docs: list[Path]) -> None:
    print("PageIndex repo clone not found.")
    print()
    print("Clone PageIndex and install its dependencies only when you want to index documents:")
    print(f"  git clone https://github.com/VectifyAI/PageIndex.git {PAGEINDEX_REPO_DIR}")
    print(f"  python -m pip install -r {PAGEINDEX_REPO_DIR / 'requirements.txt'}")
    print()
    print("Recommended indexing LLM for PageIndex:")
    print(f"  BANYANTREE_PAGEINDEX_MODEL={PAGEINDEX_MODEL}")
    print("  OPENAI_API_KEY=your_api_key  # or any LiteLLM-supported provider key")
    print()
    print(f"Input documents directory: {adapter.input_dir}")
    print(f"Expected structure output directory: {adapter.structures_dir}")
    print()
    if docs:
        print("Documents waiting for PageIndex indexing:")
        for doc in docs:
            print(f"  - {doc}")
    else:
        print("No supported input docs found. Add .pdf, .md, .markdown, or .txt files first.")


def main() -> int:
    adapter = PageIndexAdapter(
        input_dir=PAGEINDEX_INPUT_DIR,
        output_dir=PAGEINDEX_OUTPUT_DIR,
        structures_dir=PAGEINDEX_STRUCTURES_DIR,
    )
    docs = adapter.discover_documents()
    runner = _pageindex_runner()
    if runner is None:
        _print_manual_instructions(adapter, docs)
        return 0

    if not docs:
        print(f"No PageIndex input docs found in {adapter.input_dir}")
        return 0

    print(f"Found PageIndex runner: {runner}")
    print(f"Indexing model: {PAGEINDEX_RUN_MODEL}")
    print()

    for doc in docs:
        structure_path = adapter.structure_path_for(doc)
        if structure_path.exists():
            print(f"Skipping already-indexed doc: {doc}")
            continue
        if doc.suffix.lower() == ".pdf":
            input_args = ["--pdf_path", str(doc)]
        elif doc.suffix.lower() in {".md", ".markdown"}:
            input_args = ["--md_path", str(doc)]
        else:
            print(f"Skipping unsupported PageIndex direct input type for runner: {doc}")
            continue
        candidate_command = [
            sys.executable,
            str(runner),
            *input_args,
            "--model",
            PAGEINDEX_RUN_MODEL,
            "--if-add-node-text",
            "yes",
            "--if-add-node-summary",
            "yes",
            "--if-add-doc-description",
            "yes",
        ]
        print("Running:")
        print("  " + " ".join(candidate_command))
        result = subprocess.run(candidate_command, cwd=str(PAGEINDEX_REPO_DIR), check=False)
        if result.returncode != 0:
            print(f"PageIndex command failed for {doc} with exit code {result.returncode}")
            return result.returncode
        generated_path = PAGEINDEX_REPO_DIR / "results" / f"{doc.stem}_structure.json"
        if not generated_path.exists():
            print(f"PageIndex completed, but expected result was not found: {generated_path}")
            return 1
        structure_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_path, structure_path)
        print(f"Copied PageIndex structure to: {structure_path}")

    print()
    print("After structures are created, run:")
    print("  python scripts/import_pageindex_docs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
