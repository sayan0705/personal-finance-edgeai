# Financial KG and PageIndex Workflow

This project keeps the financial knowledge base separate from code so documents can be updated without editing `src/banyanTreev3_agentic.py`.

## Folder Layout

```text
data/financial_kg/
  raw_docs/
    seed/
      personal_finance_seed.json
    pageindex/
      pageindex_flattened_docs.json
  pageindex/
    inputs/
    outputs/
    structures/
  lightrag/
    documents.json
    titles.json
    document_metadata.json
    document_embeddings.npy
    faiss_index.index
    communities.npy
    community_summaries.json
  graph/
    knowledge_graph.json
    entity_contexts.json
    entity_cooccurrence.json
```

## Seed Docs

Short curated finance docs live here:

```text
data/financial_kg/raw_docs/seed/personal_finance_seed.json
```

Use this for stable concepts such as:

```text
budgeting
emergency fund
PPF/NPS/ELSS
tax deductions
retirement planning
investment risk warnings
```

## PageIndex Inputs

Place long documents here:

```text
data/financial_kg/pageindex/inputs/
```

Good candidates:

```text
annual reports
investor presentations
earnings call transcripts
tax PDFs
SEBI/RBI investor guidance
mutual fund factsheets
Markdown research notes
```

Do not store live prices as KG truth. Live market prices should continue to come from MCP tools.

## Optional PageIndex Install

PageIndex is an offline ingestion dependency. Do not install it during every normal app run.

Install only when indexing documents:

```powershell
git clone https://github.com/VectifyAI/PageIndex.git external/PageIndex
python -m pip install -r external/PageIndex/requirements.txt
```

On Kaggle, PageIndex dependency pins may conflict. Use the helper instead:

```bash
bash scripts/setup_pageindex_kaggle.sh
```

Recommended summarisation/indexing LLM:

```text
BANYANTREE_PAGEINDEX_MODEL=gpt-4o-mini
```

For Anthropic/Claude, set for the Kaggle session:

```bash
export ANTHROPIC_API_KEY="your_key"
export BANYANTREE_PAGEINDEX_MODEL="anthropic/claude-sonnet-4-5-20250929"
```

The main app can still use local `Qwen/Qwen2.5-7B-Instruct` for query answering.

## Index Long Docs

Run:

```powershell
python .\scripts\run_pageindex_indexing.py
```

This runner is intentionally conservative. If PageIndex is installed but expects different CLI flags, copy the printed suggested command and adjust it manually.

PageIndex structure JSON should end up here:

```text
data/financial_kg/pageindex/structures/
```

## Flatten PageIndex Structures

Run:

```powershell
python .\scripts\import_pageindex_docs.py
```

This writes:

```text
data/financial_kg/raw_docs/pageindex/pageindex_flattened_docs.json
```

## Qwen PDF Import Fallback

If API-based PageIndex hits rate limits, use the local Qwen PDF importer instead:

```bash
python3 -m pip install --user "tokenizers==0.19.1"
python3 scripts/import_pdf_qwen_docs.py --pages-per-chunk 2
```

This reads PDFs from:

```text
data/financial_kg/pageindex/inputs/
```

and writes the same flattened output file:

```text
data/financial_kg/raw_docs/pageindex/pageindex_flattened_docs.json
```

The output uses `source_type=qwen_pdf` and includes page ranges plus `section_path` metadata.

After PageIndex/LiteLLM experiments on Kaggle, restore the main runtime:

```bash
bash scripts/restore_main_runtime_kaggle.sh
```

The flattened docs include metadata:

```json
{
  "title": "Document section title",
  "content": "Section text or summary",
  "source_type": "pageindex",
  "doc_id": "source document id",
  "source_path": "original file path",
  "node_id": "PageIndex node id",
  "section_path": ["Chapter", "Section"],
  "page_start": 1,
  "page_end": 3
}
```

## Run Main App

The main app loads:

```text
seed docs + pageindex flattened docs
```

Then it builds/saves:

```text
data/financial_kg/lightrag/
data/financial_kg/graph/
```

On Kaggle:

```bash
cd /kaggle/working/banyantree_vscode_local
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 python3 -u src/banyanTreev3_agentic.py 2>&1 | tee logs/kaggle_run.log
```

## Git Workflow

Usually commit:

```text
data/financial_kg/raw_docs/seed/*.json
data/financial_kg/raw_docs/pageindex/*.json
data/financial_kg/pageindex/structures/*.json
src/
scripts/
docs/
```

Usually do not commit:

```text
.venv/
data/hf_cache/
data/sentence_transformers/
data/reranker/
logs/
large PDFs if the repo should stay small
```

For large source PDFs, prefer Git LFS or keep them outside Git and regenerate flattened docs when needed.
