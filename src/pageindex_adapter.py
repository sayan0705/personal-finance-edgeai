from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class PageIndexAdapter:
    """Adapter boundary for PageIndex-produced document structures.

    This module intentionally does not import PageIndex yet. It defines the
    stable local contract BanyanTree will use once PageIndex indexing is wired:
    raw documents in, structured section JSON out, LightRAG-ready docs back.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}

    def __init__(self, input_dir: Path | str, output_dir: Path | str, structures_dir: Path | str):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.structures_dir = Path(structures_dir).resolve()
        for path in (self.input_dir, self.output_dir, self.structures_dir):
            path.mkdir(parents=True, exist_ok=True)

    def discover_documents(self) -> list[Path]:
        if not self.input_dir.exists():
            return []
        docs = [
            path
            for path in self.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        return sorted(docs)

    def structure_path_for(self, source_path: Path | str) -> Path:
        source = Path(source_path).resolve()
        digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
        safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source.stem)
        return self.structures_dir / f"{safe_stem}_{digest}.json"

    def load_structure(self, structure_path: Path | str) -> dict[str, Any]:
        with open(structure_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"PageIndex structure must be a JSON object: {structure_path}")
        return data

    def flatten_structure(self, structure: dict[str, Any], source_path: Path | str = "") -> list[dict[str, Any]]:
        source = str(source_path or structure.get("source_path") or "")
        doc_id = str(structure.get("doc_id") or Path(source).stem or "pageindex_document")
        nodes = self._extract_nodes(structure)
        flattened = []
        for i, node in enumerate(nodes):
            title = self._node_title(node, fallback=f"{doc_id} section {i + 1}")
            content = self._node_content(node)
            if not content:
                continue
            flattened.append(
                {
                    "title": title,
                    "content": content,
                    "source_type": "pageindex",
                    "doc_id": doc_id,
                    "source_path": source,
                    "node_id": str(node.get("id") or node.get("node_id") or i),
                    "section_path": self._section_path(node),
                    "page_start": node.get("page_start") or node.get("start_page"),
                    "page_end": node.get("page_end") or node.get("end_page"),
                }
            )
        return flattened

    def flatten_structure_file(self, structure_path: Path | str, source_path: Path | str = "") -> list[dict[str, Any]]:
        return self.flatten_structure(self.load_structure(structure_path), source_path=source_path)

    def _extract_nodes(self, structure: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = []
        for key in ("nodes", "sections", "toc", "children"):
            value = structure.get(key)
            if isinstance(value, list):
                candidates.extend(value)
        if not candidates:
            candidates = [structure]

        out = []

        def walk(node: Any, parents: list[str] | None = None):
            if not isinstance(node, dict):
                return
            parents = parents or []
            title = self._node_title(node, fallback="")
            enriched = dict(node)
            if parents and not enriched.get("section_path"):
                enriched["section_path"] = parents + ([title] if title else [])
            out.append(enriched)
            children = node.get("children") or node.get("subsections") or []
            if isinstance(children, list):
                next_parents = parents + ([title] if title else [])
                for child in children:
                    walk(child, next_parents)

        for candidate in candidates:
            walk(candidate)
        return out

    @staticmethod
    def _node_title(node: dict[str, Any], fallback: str) -> str:
        return str(
            node.get("title")
            or node.get("heading")
            or node.get("name")
            or node.get("summary")
            or fallback
        ).strip()

    @staticmethod
    def _node_content(node: dict[str, Any]) -> str:
        parts = []
        for key in ("text", "content", "summary", "node_text"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        return "\n\n".join(dict.fromkeys(parts))

    @staticmethod
    def _section_path(node: dict[str, Any]) -> list[str]:
        value = node.get("section_path") or node.get("path") or []
        if isinstance(value, str):
            return [part.strip() for part in value.split("/") if part.strip()]
        if isinstance(value, list):
            return [str(part).strip() for part in value if str(part).strip()]
        return []
