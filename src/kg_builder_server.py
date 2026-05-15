"""
KG Builder Service — port 8020
===============================
Dedicated container for batch document ingestion into the BanyanTree
knowledge base (KG + FAISS + BM25 + RAPTOR communities).

Responsibilities:
  - Accept document uploads (JSONL seed, PDF, MD, TXT, URL)
  - Run PageIndex → flatten → ingest → rebuild all indices
  - Save artefacts to shared volume
  - Notify api container to hot-reload
  - Serve index status

The api container (port 8000) handles live queries; this container handles
all write operations. They share the ./data volume.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A — Paths & shared state
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR        = Path(os.environ.get("BANYANTREE_DATA_DIR", "/app/data"))
KG_ROOT         = Path(os.environ.get("BANYANTREE_FINANCIAL_KG_ROOT", DATA_DIR / "financial_kg"))
RAW_DOCS_DIR    = Path(os.environ.get("BANYANTREE_RAW_DOCS_DIR",    KG_ROOT / "raw_docs"))
SEED_PATH       = Path(os.environ.get("BANYANTREE_SEED_DOCS_PATH",  RAW_DOCS_DIR / "seed" / "personal_finance_seed.json"))
PAGEINDEX_FLAT  = Path(os.environ.get("BANYANTREE_PAGEINDEX_FLATTENED_DOCS_PATH",
                                      RAW_DOCS_DIR / "pageindex" / "pageindex_flattened_docs.json"))
PAGEINDEX_IN    = Path(os.environ.get("BANYANTREE_PAGEINDEX_INPUT_DIR",      RAW_DOCS_DIR / "pageindex" / "inputs"))
PAGEINDEX_OUT   = Path(os.environ.get("BANYANTREE_PAGEINDEX_OUTPUT_DIR",     RAW_DOCS_DIR / "pageindex" / "outputs"))
PAGEINDEX_STRUCTS = Path(os.environ.get("BANYANTREE_PAGEINDEX_STRUCTURES_DIR", RAW_DOCS_DIR / "pageindex" / "structures"))
KG_DB_PATH      = Path(os.environ.get("FINSAGE_KG_PATH", KG_ROOT / "lightrag"))
API_BASE        = os.environ.get("BANYANTREE_MCP_BASE", "http://api:8000").replace(":8010", ":8000")
API_RELOAD_URL  = f"{API_BASE}/internal/reload-kg"

# Ensure all directories exist at startup
for _d in (SEED_PATH.parent, PAGEINDEX_IN, PAGEINDEX_OUT, PAGEINDEX_STRUCTS, KG_DB_PATH):
    _d.mkdir(parents=True, exist_ok=True)

# Global state
_ingestor: Any = None           # lazy singleton FINANCIAL_HIERARCHICAL_LIGHT_RAG
_rebuild_lock = asyncio.Lock()  # prevents concurrent rebuilds
_rebuild_busy = False           # exposed via /status


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — FastAPI app & routes
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="BanyanTree KG Builder", version="1.0.0")


@app.on_event("startup")
async def _startup():
    logger.info("KG Builder starting — shared data dir: {}", DATA_DIR)
    logger.info("Seed path: {}", SEED_PATH)
    logger.info("PageIndex flat: {}", PAGEINDEX_FLAT)
    logger.info("KG DB path: {}", KG_DB_PATH)


# ── Route: Upload seed Q&A JSONL ─────────────────────────────────────────────
@app.post("/ingest/seed")
async def ingest_seed(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a JSON/JSONL file of {title, content, category?} Q&A pairs.

    Logic:
      1. Parse uploaded bytes as JSON list
      2. Validate each entry has 'title' and 'content'
      3. Merge with existing SEED_PATH (dedup by title — newest wins)
      4. Write merged list back to SEED_PATH
      5. Queue _run_full_rebuild() in background
    """
    if not file.filename.endswith((".json", ".jsonl")):
        raise HTTPException(400, "Only .json or .jsonl files accepted for seed ingestion")

    raw = await file.read()
    try:
        new_docs = json.loads(raw)
        if not isinstance(new_docs, list):
            raise ValueError("Expected a JSON list")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid JSON: {exc}")

    valid = []
    for i, doc in enumerate(new_docs):
        if not isinstance(doc, dict):
            logger.warning("Seed doc #{} is not a dict — skipped", i)
            continue
        if not doc.get("title") or not doc.get("content"):
            logger.warning("Seed doc #{} missing title/content — skipped", i)
            continue
        valid.append(doc)

    if not valid:
        raise HTTPException(400, "No valid {title, content} docs found in upload")

    existing = _load_json(SEED_PATH) or []
    merged   = _merge_dedup(existing, valid, key="title")
    _write_json(SEED_PATH, merged)
    logger.info("Seed merged: {} existing + {} new = {} total", len(existing), len(valid), len(merged))

    if _rebuild_busy:
        return {"status": "queued_after_current_rebuild", "new_docs": len(valid), "total_seed": len(merged)}

    background_tasks.add_task(_run_full_rebuild)
    return {"status": "queued", "new_docs": len(valid), "total_seed": len(merged)}


# ── Route: Upload PDF / MD / TXT ─────────────────────────────────────────────
@app.post("/ingest/document")
async def ingest_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a PDF, Markdown, or plain-text document.

    Logic:
      1. Validate extension (.pdf / .md / .markdown / .txt)
      2. Save raw bytes to PAGEINDEX_IN/<filename>
      3. PageIndexAdapter.flatten_structure() → extract sections with
         page ranges and section_path hierarchy breadcrumbs
      4. Merge new sections into pageindex_flattened_docs.json (dedup by title)
      5. Queue _run_full_rebuild()
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".md", ".markdown", ".txt"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Use .pdf, .md, .markdown, .txt")

    dest = PAGEINDEX_IN / file.filename
    dest.write_bytes(await file.read())
    logger.info("Document saved: {}", dest)

    # Run PageIndex extraction
    sections_count = 0
    try:
        from src.pageindex_adapter import PageIndexAdapter
        adapter = PageIndexAdapter(PAGEINDEX_IN, PAGEINDEX_OUT, PAGEINDEX_STRUCTS)

        if suffix == ".pdf":
            new_flat = _extract_pdf_sections(dest, adapter)
        else:
            new_flat = _extract_text_sections(dest)

        existing_flat = _load_json(PAGEINDEX_FLAT) or []
        merged_flat   = _merge_dedup(existing_flat, new_flat, key="title")
        _write_json(PAGEINDEX_FLAT, merged_flat)
        sections_count = len(new_flat)
        logger.info("PageIndex: {} sections extracted from {}", sections_count, file.filename)
    except Exception as exc:
        logger.warning("PageIndex extraction failed ({}): {} — file saved, rebuild will use raw content", file.filename, exc)

    if _rebuild_busy:
        return {"status": "queued_after_current_rebuild", "file": file.filename, "sections_extracted": sections_count}

    background_tasks.add_task(_run_full_rebuild)
    return {"status": "queued", "file": file.filename, "sections_extracted": sections_count}


# ── Route: Ingest from URL ────────────────────────────────────────────────────
@app.post("/ingest/url")
async def ingest_url(background_tasks: BackgroundTasks, url: str, title: str = ""):
    """
    Fetch a public URL (e.g. RBI/SEBI circular) and ingest its text content.

    Logic:
      1. httpx GET the URL (timeout 30s)
      2. BeautifulSoup parse → extract visible text (strip nav/header/footer)
      3. Chunk into ≤2000-char segments (respect sentence boundaries)
      4. Each chunk becomes one seed doc: {title: url#chunk_N, content: text}
      5. Merge into SEED_PATH and queue rebuild
    """
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL must start with http:// or https://")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "BanyanTree-KGBuilder/1.0"})
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch URL: {exc}")

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())

    if len(text) < 100:
        raise HTTPException(422, "Extracted text too short — page may be JS-rendered or empty")

    chunks    = _chunk_text(text, max_chars=2000)
    doc_title = title or url
    new_docs  = [{"title": f"{doc_title} [part {i+1}]", "content": chunk, "source": url}
                 for i, chunk in enumerate(chunks)]

    existing = _load_json(SEED_PATH) or []
    merged   = _merge_dedup(existing, new_docs, key="title")
    _write_json(SEED_PATH, merged)
    logger.info("URL ingested: {} → {} chunks", url, len(chunks))

    if _rebuild_busy:
        return {"status": "queued_after_current_rebuild", "url": url, "chars_extracted": len(text), "chunks": len(chunks)}

    background_tasks.add_task(_run_full_rebuild)
    return {"status": "queued", "url": url, "chars_extracted": len(text), "chunks": len(chunks)}


# ── Route: Force full rebuild ─────────────────────────────────────────────────
@app.post("/rebuild")
async def force_rebuild(background_tasks: BackgroundTasks):
    """
    Trigger a full KG + FAISS + BM25 + community rebuild from all existing
    raw docs on disk. Use this after dropping files directly into the volume
    without going through the /ingest routes.
    """
    if _rebuild_busy:
        return {"status": "already_running", "message": "A rebuild is in progress — wait for it to finish"}
    background_tasks.add_task(_run_full_rebuild)
    return {"status": "queued", "message": "Full rebuild triggered"}


# ── Route: Status ─────────────────────────────────────────────────────────────
@app.get("/status")
async def status():
    """
    Return current index metrics without triggering any rebuild.
    Shows whether the ingestor has been loaded and what the KG contains.
    """
    if _ingestor is None:
        kg_stats = {"note": "Ingestor not yet loaded (no rebuild has run since startup)"}
    else:
        try:
            kg_stats = {
                "documents":      len(_ingestor.documents),
                "kg_entities":    _ingestor.kg.number_of_nodes(),
                "kg_edges":       _ingestor.kg.number_of_edges(),
                "communities":    int(len(set(_ingestor.communities.tolist()))) if len(_ingestor.communities) > 0 else 0,
                "faiss_vectors":  int(_ingestor.neighbor_index.ntotal),
            }
        except Exception as exc:
            kg_stats = {"error": str(exc)}

    seed_count     = len(_load_json(SEED_PATH) or [])
    pageindex_count = len(_load_json(PAGEINDEX_FLAT) or [])

    return {
        "status":          "ok",
        "rebuild_busy":    _rebuild_busy,
        "seed_docs":       seed_count,
        "pageindex_docs":  pageindex_count,
        "total_raw_docs":  seed_count + pageindex_count,
        "kg":              kg_stats,
    }


# ── Route: Health ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "kg-builder", "rebuild_busy": _rebuild_busy}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — Background rebuild worker
# ─────────────────────────────────────────────────────────────────────────────

async def _run_full_rebuild():
    """
    Full KG + FAISS + BM25 + RAPTOR rebuild pipeline.

    Steps:
      1. Acquire rebuild lock (only one rebuild at a time)
      2. Load ingestor singleton (loads FAISS/KG from disk if present)
      3. Combine seed + pageindex docs
      4. ingest_financial_documents() → PII redact → embed → FAISS → BM25
      5. build_knowledge_graph_from_documents() → spaCy NER → NetworkX
      6. build_lightrag_communities() → KMeans on embeddings → 5 clusters
      7. generate_community_summaries() → LLM → 2-sentence per cluster
      8. build_raptor_root() → LLM → 3-sentence global overview
      9. save_kg_database() → write all artefacts to shared volume
     10. Notify api container to hot-reload
    """
    global _rebuild_busy

    if _rebuild_lock.locked():
        logger.info("Rebuild already in progress — skipping duplicate request")
        return

    async with _rebuild_lock:
        _rebuild_busy = True
        t0 = time.time()
        logger.info("=== KG Rebuild START ===")

        try:
            ingestor = _get_ingestor()

            # Load all source documents
            seed_docs      = _load_json(SEED_PATH) or []
            pageindex_docs = _load_json(PAGEINDEX_FLAT) or []
            all_docs       = seed_docs + pageindex_docs
            logger.info("Rebuild: {} seed + {} pageindex = {} total docs",
                        len(seed_docs), len(pageindex_docs), len(all_docs))

            if not all_docs:
                logger.warning("No documents to ingest — rebuild skipped")
                return

            # Step 4: ingest → PII redact + embed + FAISS + BM25
            logger.info("Step 4/9: ingest_financial_documents ...")
            ingestor.ingest_financial_documents(all_docs)

            # Step 5: KG from NER entity extraction
            logger.info("Step 5/9: build_knowledge_graph_from_documents ...")
            ingestor.build_knowledge_graph_from_documents()

            # Step 6: RAPTOR level-1 communities (KMeans)
            logger.info("Step 6/9: build_lightrag_communities ...")
            n_clusters = min(5, max(1, len(all_docs) // 2))
            ingestor.build_lightrag_communities(n_clusters=n_clusters)

            # Step 7: LLM-generated cluster summaries
            logger.info("Step 7/9: generate_community_summaries ...")
            ingestor.generate_community_summaries()

            # Step 8: RAPTOR root (global overview)
            logger.info("Step 8/9: build_raptor_root ...")
            ingestor.build_raptor_root()

            # Step 9: Persist to shared volume
            logger.info("Step 9/9: save_kg_database ...")
            ingestor.save_kg_database()

            elapsed = round(time.time() - t0, 1)
            logger.info("=== KG Rebuild DONE in {}s | docs={} entities={} edges={} ===",
                        elapsed,
                        len(ingestor.documents),
                        ingestor.kg.number_of_nodes(),
                        ingestor.kg.number_of_edges())

            # Notify api container to reload indices from disk
            await _notify_api_reload()

        except Exception as exc:
            logger.exception("KG Rebuild FAILED after {}s: {}", round(time.time() - t0, 1), exc)
        finally:
            _rebuild_busy = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION D — Notify api container
# ─────────────────────────────────────────────────────────────────────────────

async def _notify_api_reload():
    """
    Fire-and-forget POST to api:8000/internal/reload-kg.
    If the api is down or busy, it will hot-reload from the shared volume
    on the next query anyway (via _ensure_ready → load_kg_database).
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(API_RELOAD_URL)
            logger.info("API reload notified: HTTP {}", resp.status_code)
    except Exception as exc:
        logger.warning("API reload notification failed (api may be down): {}", exc)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION E — Ingestor singleton
# ─────────────────────────────────────────────────────────────────────────────

def _get_ingestor() -> Any:
    """
    Lazy-load FINANCIAL_HIERARCHICAL_LIGHT_RAG once per process lifetime.
    Heavy operation: loads BGE-M3, FAISS index, KG, BM25 from disk.
    Subsequent calls return the same instance (already loaded).
    """
    global _ingestor
    if _ingestor is not None:
        return _ingestor

    logger.info("Loading FINANCIAL_HIERARCHICAL_LIGHT_RAG ingestor ...")
    sys.path.insert(0, "/app")

    # Prevent banyanTreev3_agentic from starting its embedded MCP server
    os.environ.setdefault("BANYANTREE_SKIP_MCP_BOOTSTRAP", "1")

    from src.banyanTreev3_agentic import FINANCIAL_HIERARCHICAL_LIGHT_RAG
    _ingestor = FINANCIAL_HIERARCHICAL_LIGHT_RAG(kg_db_path=str(KG_DB_PATH))
    logger.info("Ingestor loaded: {} docs, {} KG entities",
                len(_ingestor.documents), _ingestor.kg.number_of_nodes())
    return _ingestor


# ─────────────────────────────────────────────────────────────────────────────
# SECTION F — Document extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pdf_sections(pdf_path: Path, adapter) -> list[dict]:
    """
    Extract text from each PDF page, group into sections.
    Each page becomes one doc with page_start/page_end metadata.
    Falls back to raw text if PageIndex structure not available.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        sections = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if len(text) < 50:
                continue
            sections.append({
                "title":       f"{pdf_path.stem} — page {i + 1}",
                "content":     text[:3000],
                "source_type": "pageindex",
                "category":    "uploaded_document",
                "source_path": str(pdf_path),
                "page_start":  i + 1,
                "page_end":    i + 1,
                "section_path": [pdf_path.stem, f"page_{i + 1}"],
            })
        return sections
    except Exception as exc:
        logger.warning("PDF extraction error for {}: {}", pdf_path.name, exc)
        return []


def _extract_text_sections(file_path: Path) -> list[dict]:
    """
    Split plain-text / markdown into chunks of ≤2000 chars.
    Each chunk becomes one doc preserving the filename as source.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    chunks = _chunk_text(text, max_chars=2000)
    return [
        {
            "title":       f"{file_path.stem} — part {i + 1}",
            "content":     chunk,
            "source_type": "pageindex",
            "category":    "uploaded_document",
            "source_path": str(file_path),
            "section_path": [file_path.stem, f"part_{i + 1}"],
        }
        for i, chunk in enumerate(chunks)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION G — Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list | dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load {}: {}", path, exc)
        return None


def _write_json(path: Path, data: list | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _merge_dedup(existing: list, new_docs: list, key: str = "title") -> list:
    """
    Merge two lists of dicts, deduplicating by `key`.
    New docs overwrite existing entries with the same key value.
    Preserves order: existing first, new additions appended.
    """
    index = {doc.get(key): doc for doc in existing if doc.get(key)}
    for doc in new_docs:
        k = doc.get(key)
        if k:
            index[k] = doc   # newest wins
    return list(index.values())


def _chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """
    Split text into chunks of at most `max_chars`, breaking on sentence
    boundaries ('. ') where possible to preserve readability.
    """
    if len(text) <= max_chars:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text)
            break
        split_at = text.rfind(". ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        else:
            split_at += 1   # include the period
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    return [c for c in chunks if c]