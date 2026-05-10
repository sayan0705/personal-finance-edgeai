#!/usr/bin/env python3
"""Batch PDF ingestion CLI — populates the ChromaDB vector store offline.

Usage examples
--------------
# Ingest all PDFs from a local folder
python scripts/ingest.py --folder data/raw/public/sebi

# Ingest a single PDF from a URL
python scripts/ingest.py --url https://example.com/report.pdf

# Ingest multiple URLs listed in a text file
python scripts/ingest.py --url-file urls.txt

# Preview what would be ingested without writing to ChromaDB
python scripts/ingest.py --folder data/raw --dry-run

# Clear the collection then reindex from scratch
python scripts/ingest.py --folder data/raw --clear

# Override ChromaDB connection (default: localhost:8001)
python scripts/ingest.py --folder data/raw --chroma-host localhost --chroma-port 8001

# Inside Docker (connect to chromadb service on the same network)
python scripts/ingest.py --folder /data --chroma-host chromadb --chroma-port 8000
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table

# Ensure repo root is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingestion.chunker import SemanticChunker
from rag.ingestion.loader import DocumentLoader
from rag.models import Document

# ChromaDB-dependent imports are deferred to main() so the script can run
# (e.g. --dry-run, --help) without chromadb installed in the local env.
# from rag.ingestion.embedder import EmbeddingIndexer
# from rag.vector_store.chroma_store import ChromaVectorStore

console = Console()


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class SourceResult:
    """Per-source ingestion outcome."""

    label: str
    pages: int = 0
    chunks: int = 0
    indexed: int = 0
    elapsed_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class IngestSummary:
    """Aggregated result across all sources."""

    results: list[SourceResult] = field(default_factory=list)
    dry_run: bool = False

    @property
    def total_pages(self) -> int:
        return sum(r.pages for r in self.results)

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks for r in self.results)

    @property
    def total_indexed(self) -> int:
        return sum(r.indexed for r in self.results)

    @property
    def failed(self) -> list[SourceResult]:
        return [r for r in self.results if not r.ok]


# ---------------------------------------------------------------------------
# Core ingestion logic
# ---------------------------------------------------------------------------

def ingest_source(
    source: str,
    loader: DocumentLoader,
    chunker: SemanticChunker,
    indexer: EmbeddingIndexer | None,
    dry_run: bool,
) -> SourceResult:
    """Load, chunk, and optionally index a single source (file, folder, or URL).

    Args:
        source: Local path or HTTP(S) URL.
        loader: Initialised DocumentLoader.
        chunker: Initialised SemanticChunker.
        indexer: Initialised EmbeddingIndexer, or None in dry-run mode.
        dry_run: When True, skip the upsert step.

    Returns:
        SourceResult with counts and timing.
    """
    label = source if len(source) <= 60 else "…" + source[-57:]
    result = SourceResult(label=label)
    t0 = time.perf_counter()

    try:
        docs: list[Document] = loader.load(source)
        result.pages = len(docs)

        if not docs:
            logger.warning(f"No content extracted from: {source}")
            result.error = "no content"
            return result

        chunks = chunker.chunk(docs)
        result.chunks = len(chunks)

        if not dry_run and indexer is not None:
            result.indexed = indexer.index(chunks)
        else:
            result.indexed = len(chunks)  # would-be count

    except Exception as exc:
        logger.error(f"Failed processing {source}: {exc}")
        result.error = str(exc)[:120]

    result.elapsed_s = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def resolve_sources(
    folders: list[str],
    urls: list[str],
    url_files: list[str],
) -> list[str]:
    """Build the flat list of sources to ingest.

    Args:
        folders: Directory paths containing PDF files.
        urls: Explicit PDF URLs.
        url_files: Paths to text files listing one URL per line.

    Returns:
        Deduplicated list of source strings (paths and URLs).
    """
    seen: set[str] = set()
    sources: list[str] = []

    def _add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            sources.append(s)

    for folder in folders:
        _add(folder)

    for url in urls:
        _add(url)

    for uf in url_files:
        uf_path = Path(uf)
        if not uf_path.exists():
            logger.error(f"URL file not found: {uf}")
            continue
        for line in uf_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                _add(line)

    return sources


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------

def print_summary(summary: IngestSummary, chroma_total: int | None) -> None:
    """Print a formatted results table to the terminal.

    Args:
        summary: Aggregated ingestion results.
        chroma_total: Current ChromaDB document count (None if unavailable).
    """
    table = Table(
        title="Ingestion Summary" + (" (DRY RUN — nothing was written)" if summary.dry_run else ""),
        show_footer=True,
        footer_style="bold",
    )
    table.add_column("Source", style="cyan", footer="TOTAL")
    table.add_column("Pages", justify="right", footer=str(summary.total_pages))
    table.add_column("Chunks", justify="right", footer=str(summary.total_chunks))
    table.add_column("Indexed", justify="right", footer=str(summary.total_indexed))
    table.add_column("Time (s)", justify="right", footer="")
    table.add_column("Status", footer="")

    for r in summary.results:
        status = "[green]✓[/green]" if r.ok else f"[red]✗ {r.error[:40]}[/red]"
        table.add_row(
            r.label,
            str(r.pages),
            str(r.chunks),
            str(r.indexed),
            f"{r.elapsed_s:.1f}",
            status,
        )

    console.print(table)

    if chroma_total is not None and not summary.dry_run:
        console.print(
            f"\n[bold green]ChromaDB collection now contains {chroma_total:,} chunks.[/bold green]"
        )

    if summary.failed:
        console.print(
            f"\n[bold red]{len(summary.failed)} source(s) failed — check logs above.[/bold red]"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    p = argparse.ArgumentParser(
        prog="ingest",
        description=(
            "Batch-ingest PDFs into the FinEdge ChromaDB vector store.\n"
            "At least one of --folder, --url, or --url-file is required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_argument_group("Sources (at least one required)")
    src.add_argument(
        "--folder",
        metavar="DIR",
        action="append",
        default=[],
        help="Directory to scan recursively for PDF files (repeatable).",
    )
    src.add_argument(
        "--url",
        metavar="URL",
        action="append",
        default=[],
        help="PDF URL to download and ingest (repeatable).",
    )
    src.add_argument(
        "--url-file",
        metavar="FILE",
        action="append",
        default=[],
        help="Text file with one PDF URL per line; lines starting with # are ignored (repeatable).",
    )

    db = p.add_argument_group("ChromaDB connection")
    db.add_argument("--chroma-host", default="localhost", help="ChromaDB host (default: localhost)")
    db.add_argument("--chroma-port", type=int, default=8001, help="ChromaDB port (default: 8001)")
    db.add_argument(
        "--collection",
        default="finedge_finance_docs",
        help="ChromaDB collection name (default: finedge_finance_docs)",
    )

    proc = p.add_argument_group("Processing")
    proc.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model for embeddings.",
    )
    proc.add_argument("--chunk-size", type=int, default=512, help="Characters per chunk (default: 512)")
    proc.add_argument("--chunk-overlap", type=int, default=64, help="Overlap between chunks (default: 64)")
    proc.add_argument("--batch-size", type=int, default=128, help="ChromaDB upsert batch size (default: 128)")

    mode = p.add_argument_group("Mode")
    mode.add_argument(
        "--clear",
        action="store_true",
        help="Drop and recreate the ChromaDB collection before indexing.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk sources but do NOT write to ChromaDB.",
    )

    return p


def main() -> int:
    """Entry point. Returns exit code (0 = success, 1 = partial failure, 2 = fatal)."""
    parser = build_parser()
    args = parser.parse_args()

    sources = resolve_sources(args.folder, args.url, args.url_file)
    if not sources:
        parser.error("At least one of --folder, --url, or --url-file is required.")

    # ------------------------------------------------------------------
    # Set up pipeline components
    # ------------------------------------------------------------------
    loader = DocumentLoader()
    chunker = SemanticChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    store = None
    indexer = None

    if not args.dry_run:
        # Deferred import — chromadb only required when actually writing
        from rag.ingestion.embedder import EmbeddingIndexer          # noqa: PLC0415
        from rag.vector_store.chroma_store import ChromaVectorStore  # noqa: PLC0415

        console.print(
            f"[bold]Connecting to ChromaDB[/bold] at "
            f"[cyan]{args.chroma_host}:{args.chroma_port}[/cyan] "
            f"collection=[cyan]{args.collection}[/cyan]"
        )
        try:
            store = ChromaVectorStore(
                host=args.chroma_host,
                port=args.chroma_port,
                collection_name=args.collection,
                embedding_model=args.embedding_model,
            )
            if args.clear:
                console.print("[yellow]--clear: dropping existing collection…[/yellow]")
                store.delete_collection()
                console.print("[yellow]Collection cleared.[/yellow]")

            before_count = store.count()
            console.print(f"Collection currently has [bold]{before_count:,}[/bold] chunks.\n")
            indexer = EmbeddingIndexer(vector_store=store, batch_size=args.batch_size)
        except Exception as exc:
            console.print(f"[bold red]Cannot connect to ChromaDB: {exc}[/bold red]")
            logger.error(f"ChromaDB connection failed: {exc}")
            return 2
    else:
        console.print("[yellow]DRY RUN — ChromaDB will not be modified.[/yellow]\n")

    # ------------------------------------------------------------------
    # Process each source
    # ------------------------------------------------------------------
    summary = IngestSummary(dry_run=args.dry_run)
    console.print(f"Processing [bold]{len(sources)}[/bold] source(s)…\n")

    for i, source in enumerate(sources, start=1):
        console.print(f"[{i}/{len(sources)}] {source}")
        result = ingest_source(source, loader, chunker, indexer, args.dry_run)
        summary.results.append(result)

        if result.ok:
            action = "would index" if args.dry_run else "indexed"
            console.print(
                f"  → [green]{result.pages} pages, {result.chunks} chunks, {result.indexed} {action}[/green] "
                f"({result.elapsed_s:.1f}s)"
            )
        else:
            console.print(f"  → [red]FAILED: {result.error}[/red]")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    console.print()
    after_count: int | None = None
    if store is not None and not args.dry_run:
        try:
            after_count = store.count()
        except Exception:
            pass

    print_summary(summary, after_count)

    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
