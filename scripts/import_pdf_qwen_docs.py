from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("BANYANTREE_DATA_DIR", PROJECT_ROOT / "data")).resolve()
FINANCIAL_KG_ROOT = Path(os.environ.get("BANYANTREE_FINANCIAL_KG_ROOT", DATA_DIR / "financial_kg")).resolve()
PAGEINDEX_INPUT_DIR = Path(os.environ.get("BANYANTREE_PAGEINDEX_INPUT_DIR", FINANCIAL_KG_ROOT / "pageindex" / "inputs")).resolve()
RAW_DOCS_DIR = Path(os.environ.get("BANYANTREE_RAW_DOCS_DIR", FINANCIAL_KG_ROOT / "raw_docs")).resolve()
OUTPUT_PATH = Path(
    os.environ.get("BANYANTREE_PAGEINDEX_FLATTENED_DOCS_PATH", RAW_DOCS_DIR / "pageindex" / "pageindex_flattened_docs.json")
).resolve()
QWEN_MODEL_ID = os.environ.get("BANYANTREE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    import fitz

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i, "text": normalize_text(text)})
    return pages


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[dict[str, Any]], pages_per_chunk: int) -> list[dict[str, Any]]:
    chunks = []
    for start in range(0, len(pages), pages_per_chunk):
        group = pages[start : start + pages_per_chunk]
        if not group:
            continue
        chunks.append(
            {
                "page_start": group[0]["page"],
                "page_end": group[-1]["page"],
                "text": "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in group),
            }
        )
    return chunks


def build_prompt(doc_id: str, chunk: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You extract financial knowledge sections from PDF text for a retrieval knowledge base. "
        "Return only valid JSON. Do not include markdown."
    )
    user = f"""
Document ID: {doc_id}
Pages: {chunk['page_start']}-{chunk['page_end']}

PDF text:
{chunk['text'][:9000]}

Extract 1 to 5 useful financial knowledge sections from this text.
Return a JSON array. Each item must have:
- title: concise section title
- content: self-contained summary with important facts and numbers
- category: short snake_case category
- section_path: list of hierarchy labels, broad to specific
- page_start: integer
- page_end: integer

Use the given page range if exact pages are unclear.
Return only the JSON array.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_qwen():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the Qwen PDF importer.")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def qwen_json_sections(tokenizer, model, doc_id: str, chunk: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    messages = build_prompt(doc_id, chunk)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=12000, truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=900,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return parse_json_array(raw)


def parse_json_array(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        print(f"Could not find JSON array in Qwen output:\n{raw[:500]}")
        return []
    try:
        data = json.loads(match.group())
    except Exception as exc:
        print(f"Could not parse Qwen JSON output: {exc}\n{raw[:500]}")
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def normalize_section(section: dict[str, Any], pdf_path: Path, doc_id: str, chunk: dict[str, Any], index: int) -> dict[str, Any] | None:
    content = str(section.get("content") or "").strip()
    if not content:
        return None
    page_start = int(section.get("page_start") or chunk["page_start"])
    page_end = int(section.get("page_end") or chunk["page_end"])
    title = str(section.get("title") or f"{doc_id} pages {page_start}-{page_end}").strip()
    section_path = section.get("section_path") or [title]
    if isinstance(section_path, str):
        section_path = [part.strip() for part in section_path.split("/") if part.strip()]
    if not isinstance(section_path, list):
        section_path = [title]
    return {
        "title": title,
        "content": content,
        "source_type": "qwen_pdf",
        "category": str(section.get("category") or "financial_pdf").strip(),
        "doc_id": doc_id,
        "source_path": str(pdf_path),
        "node_id": f"{doc_id}_p{page_start}_{page_end}_{index}",
        "section_path": [str(part).strip() for part in section_path if str(part).strip()],
        "page_start": page_start,
        "page_end": page_end,
    }


def load_existing_output(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def main() -> int:
    parser = argparse.ArgumentParser(description="Import PDFs into flattened financial docs using local Qwen.")
    parser.add_argument("--input-dir", type=Path, default=PAGEINDEX_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--pages-per-chunk", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pdfs = sorted(args.input_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.input_dir}")
        return 0

    existing = [] if args.overwrite else load_existing_output(args.output)
    existing_keys = {(doc.get("doc_id"), doc.get("node_id")) for doc in existing}

    print(f"Loading local Qwen model: {QWEN_MODEL_ID}")
    tokenizer, model = load_qwen()

    imported = []
    for pdf_path in pdfs:
        doc_id = pdf_path.stem
        print(f"\nProcessing PDF: {pdf_path}")
        pages = extract_pdf_pages(pdf_path)
        chunks = chunk_pages(pages, args.pages_per_chunk)
        print(f"Extracted {len(pages)} pages into {len(chunks)} chunk(s)")
        for chunk_index, chunk in enumerate(chunks):
            print(f"  Qwen extracting pages {chunk['page_start']}-{chunk['page_end']}...")
            sections = qwen_json_sections(tokenizer, model, doc_id, chunk)
            for i, section in enumerate(sections):
                doc = normalize_section(section, pdf_path, doc_id, chunk, i)
                if not doc:
                    continue
                key = (doc["doc_id"], doc["node_id"])
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                imported.append(doc)

    output_docs = existing + imported
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_docs, f, indent=2, ensure_ascii=False)

    print(f"\nImported {len(imported)} new Qwen PDF section docs")
    print(f"Total flattened docs: {len(output_docs)}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
