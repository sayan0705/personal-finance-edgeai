"""Private uploaded-document storage, redaction, and parsers.

Raw files and full parsed rows stay under data/user_docs. MCP tools return only
redacted document profiles or aggregate observations for ReAct/LLM use.
"""
from __future__ import annotations

import csv, json, os, re, uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _data_dir() -> Path:
    return Path(os.environ.get("BANYANTREE_DATA_DIR", Path.cwd() / "data")).resolve()


def user_docs_root() -> Path:
    root = Path(os.environ.get("BANYANTREE_USER_DOCS_DIR", _data_dir() / "user_docs")).resolve()
    for name in ("uploads", "parsed", "redacted"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _doc_dir(doc_id: str, kind: str) -> Path:
    p = user_docs_root() / kind / doc_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(name or "document").name).strip(" .")
    return out or "document"


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_uploaded_document(filename: str, content: bytes, session_id: str | None = None) -> dict[str, Any]:
    safe = _safe_name(filename)
    suffix = Path(safe).suffix.lower() or ".bin"
    doc_id = f"doc_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    target = _doc_dir(doc_id, "uploads") / f"original{suffix}"
    target.write_bytes(content)
    meta = {
        "document_id": doc_id,
        "filename": safe,
        "stored_path": str(target),
        "session_id": session_id or "default",
        "size_bytes": len(content),
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "status": "uploaded",
    }
    _write_json(_doc_dir(doc_id, "uploads") / "metadata.json", meta)
    return {"document_id": doc_id, "filename": safe, "size_bytes": len(content), "status": "uploaded", "message": f"Use document_id {doc_id} in your query."}


def load_document_metadata(document_id: str) -> dict[str, Any]:
    meta = _read_json(_doc_dir(document_id, "uploads") / "metadata.json")
    if not meta:
        raise FileNotFoundError(f"Unknown document_id: {document_id}")
    return meta


class DocumentRedactor:
    def __init__(self) -> None:
        self.enabled = False
        try:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
            nlp = NlpEngineProvider(nlp_configuration={"nlp_engine_name":"spacy","models":[{"lang_code":"en","model_name":"en_core_web_sm"}]}).create_engine()
            self.analyzer = AnalyzerEngine(nlp_engine=nlp)
            self.anonymizer = AnonymizerEngine()
            self.operator_config = OperatorConfig
            self.enabled = True
            for entity, pattern, score in _custom_patterns():
                self.analyzer.registry.add_recognizer(PatternRecognizer(supported_entity=entity, patterns=[Pattern(name=entity.lower(), regex=pattern, score=score)]))
        except Exception:
            self.enabled = False

    def redact(self, text: str) -> str:
        text = str(text or "")
        if self.enabled:
            entities = ["PERSON","EMAIL_ADDRESS","PHONE_NUMBER","AADHAAR_NUMBER","PAN_NUMBER","INDIAN_PHONE","IFSC_CODE","BANK_ACCOUNT_NUMBER","CARD_NUMBER","UPI_ID","TRANSACTION_ID","POLICY_NUMBER","CUSTOMER_ID","GSTIN","EMPLOYEE_ID"]
            results = self.analyzer.analyze(text=text, entities=entities, language="en")
            ops = {e: self.operator_config("replace", {"new_value": f"<{e}>"}) for e in entities}
            return self.anonymizer.anonymize(text=text, analyzer_results=results, operators=ops).text
        return _regex_redact(text)


def _custom_patterns() -> list[tuple[str, str, float]]:
    return [
        ("AADHAAR_NUMBER", r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", .95),
        ("PAN_NUMBER", r"\b[A-Z]{5}\d{4}[A-Z]\b", .95),
        ("INDIAN_PHONE", r"\b[6-9]\d{9}\b", .85),
        ("IFSC_CODE", r"\b[A-Z]{4}0[A-Z0-9]{6}\b", .90),
        ("BANK_ACCOUNT_NUMBER", r"\b\d{9,18}\b", .55),
        ("CARD_NUMBER", r"\b(?:\d[ -]*?){13,19}\b", .65),
        ("UPI_ID", r"\b[\w.-]{2,}@[A-Za-z]{2,}\b", .85),
        ("TRANSACTION_ID", r"\b(?:UTR|REF|TXN|IMPS|NEFT|UPI)[\s:/-]*[A-Z0-9]{6,}\b", .70),
        ("POLICY_NUMBER", r"\b(?:policy\s*(?:no|number)\s*[:/-]?\s*)[A-Z0-9/-]{5,}\b", .80),
        ("CUSTOMER_ID", r"\b(?:customer\s*id|cust\s*id|consumer\s*no|ca\s*number)\s*[:/-]?\s*[A-Z0-9/-]{4,}\b", .75),
        ("GSTIN", r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b", .90),
        ("EMPLOYEE_ID", r"\b(?:employee\s*id|emp\s*id)\s*[:/-]?\s*[A-Z0-9/-]{3,}\b", .70),
    ]

_REDACTOR: DocumentRedactor | None = None

def redactor() -> DocumentRedactor:
    global _REDACTOR
    if _REDACTOR is None:
        _REDACTOR = DocumentRedactor()
    return _REDACTOR


def _regex_redact(text: str) -> str:
    for pattern, repl in [
        (r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", "<AADHAAR_NUMBER>"),
        (r"\b[A-Z]{5}\d{4}[A-Z]\b", "<PAN_NUMBER>"),
        (r"\b[6-9]\d{9}\b", "<INDIAN_PHONE>"),
        (r"\b[A-Z]{4}0[A-Z0-9]{6}\b", "<IFSC_CODE>"),
        (r"\b[\w.-]+@[\w.-]+\.\w+\b", "<EMAIL_ADDRESS>"),
        (r"\b[\w.-]{2,}@[A-Za-z]{2,}\b", "<UPI_ID>"),
        (r"\b(?:\d[ -]*?){13,19}\b", "<CARD_NUMBER>"),
        (r"\b(?:UTR|REF|TXN|IMPS|NEFT|UPI)[\s:/-]*[A-Z0-9]{6,}\b", "<TRANSACTION_ID>"),
        (r"\b(?:policy\s*(?:no|number)\s*[:/-]?\s*)[A-Z0-9/-]{5,}\b", "<POLICY_NUMBER>"),
        (r"\b(?:customer\s*id|cust\s*id|consumer\s*no|ca\s*number)\s*[:/-]?\s*[A-Z0-9/-]{4,}\b", "<CUSTOMER_ID>"),
        (r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b", "<GSTIN>"),
    ]:
        text = re.sub(pattern, repl, text, flags=re.I)
    return text


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Redact text fields while preserving numeric aggregates for analysis."""
    return _redact_value(payload)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, str):
        return redactor().redact(value)
    return value

def extract_document_private(document_id: str) -> dict[str, Any]:
    cache = _doc_dir(document_id, "parsed") / "private_parse.json"
    loaded = _read_json(cache)
    if loaded:
        return loaded
    meta = load_document_metadata(document_id)
    path = Path(meta["stored_path"])
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        parsed = _extract_pdf(path)
    elif suffix == ".csv":
        parsed = _extract_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        parsed = _extract_excel(path)
    else:
        parsed = _extract_text(path)
    parsed.update({"document_id": document_id, "filename": meta.get("filename", path.name), "format": suffix.lstrip(".") or "unknown", "extracted_at": datetime.utcnow().isoformat() + "Z"})
    _write_json(cache, parsed)
    return parsed


def _extract_pdf(path: Path) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("PDF parsing requires pypdf") from exc
    reader = PdfReader(str(path))
    pages = [{"page": i, "text": (page.extract_text() or "").strip()} for i, page in enumerate(reader.pages, start=1)]
    return {"pages": pages, "tables": []}


def _extract_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        rows = [{str(k or "").strip(): str(v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]
    headers = list(rows[0].keys()) if rows else []
    text = "\n".join(", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows[:250])
    return {"pages": [{"page": 1, "text": text}], "tables": [{"page": 1, "headers": headers, "rows": rows}]}


def _extract_excel(path: Path) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Excel parsing requires pandas/openpyxl") from exc
    pages, tables = [], []
    for i, (sheet, frame) in enumerate(pd.read_excel(path, sheet_name=None, dtype=str).items(), start=1):
        frame = frame.fillna("")
        rows = frame.to_dict(orient="records")
        headers = [str(c) for c in frame.columns]
        text = "\n".join(", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows[:250])
        pages.append({"page": i, "text": f"Sheet: {sheet}\n{text}"})
        tables.append({"page": i, "sheet": sheet, "headers": headers, "rows": rows})
    return {"pages": pages, "tables": tables}


def _extract_text(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = [text[i:i+6000] for i in range(0, len(text), 6000)] or [""]
    return {"pages": [{"page": i + 1, "text": chunk.strip()} for i, chunk in enumerate(chunks)], "tables": []}


def build_redacted_profile(document_id: str, user_query: str = "") -> dict[str, Any]:
    private = extract_document_private(document_id)
    pages = private.get("pages", [])
    samples, seen = [], set()
    for page in pages[:2] + (pages[-1:] if len(pages) > 2 else []):
        num = page.get("page")
        if num in seen:
            continue
        seen.add(num)
        text = str(page.get("text", ""))
        lines = [line.strip() for line in text.splitlines() if line.strip()][:10]
        samples.append({"page": num, "top_lines": redactor().redact("\n".join(lines)), "dates_found": _date_strings(text)[:6], "amount_count": len(_money_strings(text))})
    headers = [{"page": t.get("page"), "sheet": t.get("sheet"), "headers": [redactor().redact(str(h)) for h in t.get("headers", [])]} for t in private.get("tables", [])[:12]]
    profile = {"document_id": document_id, "filename": redactor().redact(str(private.get("filename", ""))), "format": private.get("format"), "page_count": len(pages), "sampled_pages": samples, "table_headers": headers, "type_hints": _type_hints(user_query, samples, headers), "available_extractors": ["bank_statement_analyzer", "credit_card_statement_analyzer", "bill_parser", "salary_slip_parser", "document_rag_search"]}
    result = {"result": f"Redacted document profile for {document_id}. Pages: {len(pages)}.", "redacted_profile": profile}
    _write_json(_doc_dir(document_id, "redacted") / "profile.json", result)
    return result


def _type_hints(user_query: str, samples: list[dict[str, Any]], headers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    haystack = " ".join([user_query or ""] + [str(s.get("top_lines", "")) for s in samples] + [" ".join(h.get("headers", [])) for h in headers]).lower()
    rules = {"bank_statement": ["statement", "narration", "withdrawal", "deposit", "balance", "debit", "credit"], "credit_card_statement": ["credit card", "minimum due", "payment due", "credit limit", "total due"], "insurance_bill": ["policy", "premium", "sum assured", "renewal", "life assured"], "utility_bill": ["consumer", "units", "bill amount", "electricity", "due date"], "salary_slip": ["salary", "basic", "hra", "provident fund", "net pay", "tds"]}
    out = []
    for doc_type, terms in rules.items():
        matched = [t for t in terms if t in haystack]
        if matched:
            out.append({"document_type": doc_type, "score": len(matched), "matched_terms": matched[:5]})
    return sorted(out, key=lambda x: x["score"], reverse=True)[:3]


def document_search(document_id: str, query: str, top_k: int = 6) -> dict[str, Any]:
    private = extract_document_private(document_id)
    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 2]
    scored = []
    for page in private.get("pages", []):
        text = str(page.get("text", ""))
        score = sum(text.lower().count(token) for token in tokens)
        if score:
            scored.append((score, page))
    scored.sort(key=lambda item: item[0], reverse=True)
    snippets = []
    for score, page in scored[:top_k]:
        text = str(page.get("text", ""))
        lower = text.lower()
        hits = [lower.find(t) for t in tokens if lower.find(t) >= 0]
        start = max(0, min(hits) - 120) if hits else 0
        snippets.append({"page": page.get("page"), "score": score, "snippet": redactor().redact(text[start:start+420].replace("\n", " ").strip())})
    return {"result": f"Found {len(snippets)} redacted snippets for {document_id}.", "document_id": document_id, "snippets": snippets}

def _all_text(private: dict[str, Any]) -> str:
    return "\n".join(str(p.get("text", "")) for p in private.get("pages", []))


def _date_strings(text: str) -> list[str]:
    out = []
    for pat in [
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        r"\b\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b",
    ]:
        out.extend(re.findall(pat, text or "", flags=re.I))
    return out


def _money_strings(text: str) -> list[str]:
    return re.findall(r"(?:Rs\.?|INR|\u20b9)?\s*\d[\d,]*(?:\.\d+)?\s*(?:lakh|lac|crore|cr|k)?", text or "", flags=re.I)


def _parse_money(value: Any) -> float | None:
    raw = str(value or "").strip()
    m = re.search(r"(-?\d[\d,]*(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|thousand)?", raw, flags=re.I)
    if not m:
        return None
    amt = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit in {"crore", "cr"}: amt *= 10_000_000
    elif unit in {"lakh", "lac"}: amt *= 100_000
    elif unit in {"k", "thousand"}: amt *= 1_000
    return abs(amt)


def _parse_date(value: Any) -> str | None:
    raw = re.sub(r"\s+", " ", str(value or "").strip())
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d/%b/%Y", "%d/%B/%Y", "%d %b %Y", "%d %B %Y", "%d %b %y", "%d %B %y", "%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y"]:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except Exception:
            pass
    found = _date_strings(raw)
    return _parse_date(found[0]) if found and found[0] != raw else None


def _norm_header(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {"date": {"date", "txn_date", "transaction_date", "value_date", "posting_date", "tran_date"}, "description": {"description", "narration", "particulars", "details", "remarks", "transaction_remarks"}, "debit": {"debit", "withdrawal", "withdrawals", "paid_out", "dr", "debit_amount"}, "credit": {"credit", "deposit", "deposits", "paid_in", "cr", "credit_amount"}, "amount": {"amount", "transaction_amount", "txn_amount"}, "balance": {"balance", "closing_balance", "available_balance"}}
    for canonical, names in aliases.items():
        if text in names:
            return canonical
    return text


def _rows(private: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for table in private.get("tables", []):
        for row in table.get("rows", []):
            clean = {_norm_header(k): v for k, v in row.items()}
            clean["_page"] = table.get("page")
            out.append(clean)
    return out


def _category(desc: str) -> str:
    d = desc.lower()
    rules = {"salary": ["salary", "payroll", "sal credit"], "rent": ["rent"], "emi": ["emi", "loan", "bajaj finance"], "insurance": ["insurance", "premium", "lic"], "investment": ["sip", "mutual fund", "zerodha", "groww", "cams", "kfin"], "food": ["swiggy", "zomato", "restaurant"], "shopping": ["amazon", "flipkart", "myntra"], "utilities": ["electricity", "water", "gas", "broadband", "airtel", "jio"], "cash": ["atm", "cash withdrawal"], "bank_charges": ["charge", "fee", "penalty", "bounce", "mandate fail"], "tax": ["income tax", "tds", "gst"]}
    for cat, terms in rules.items():
        if any(t in d for t in terms):
            return cat
    return "uncategorized"


def _bank_transactions(private: dict[str, Any]) -> list[dict[str, Any]]:
    txns = []
    for row in _rows(private):
        date = _parse_date(row.get("date"))
        desc = str(row.get("description") or row.get("particular") or "").strip()
        debit, credit, amount = _parse_money(row.get("debit")), _parse_money(row.get("credit")), _parse_money(row.get("amount"))
        if amount is not None and debit is None and credit is None:
            credit = amount if any(w in desc.lower() for w in ["salary", "credit", "deposit", "refund"]) else None
            debit = None if credit is not None else amount
        if date and desc and (debit is not None or credit is not None):
            txns.append({"date": date, "description": desc[:160], "debit": debit, "credit": credit, "amount": credit or debit or 0, "type": "credit" if credit is not None else "debit", "balance": _parse_money(row.get("balance")), "category": _category(desc), "page": row.get("_page")})
    return txns or _bank_transactions_from_text(_all_text(private))


def _bank_transactions_from_text(text: str) -> list[dict[str, Any]]:
    txns = []
    for line in (text or "").splitlines():
        dm = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4})\b", line)
        if not dm:
            continue
        amounts = [_parse_money(x) for x in _money_strings(line)]
        amounts = [x for x in amounts if x is not None]
        if not amounts:
            continue
        desc = line[dm.end():].strip()[:160]
        credit = amounts[0] if any(w in line.lower() for w in [" cr", " credit", "salary", "deposit", "refund"]) else None
        debit = None if credit is not None else amounts[0]
        txns.append({"date": _parse_date(dm.group(1)), "description": desc, "debit": debit, "credit": credit, "amount": credit or debit or 0, "type": "credit" if credit is not None else "debit", "balance": amounts[-1] if len(amounts) > 1 else None, "category": _category(desc), "page": None})
    return [t for t in txns if t.get("date")]


def analyze_bank_statement(document_id: str) -> dict[str, Any]:
    private = extract_document_private(document_id)
    txns = _bank_transactions(private)
    dates = sorted([t["date"] for t in txns if t.get("date")])
    total_cr = sum(t.get("credit") or 0 for t in txns)
    total_dr = sum(t.get("debit") or 0 for t in txns)
    months = max(1, len({d[:7] for d in dates}))
    cats: defaultdict[str, float] = defaultdict(float)
    for t in txns:
        if t.get("type") == "debit":
            cats[t.get("category", "uncategorized")] += t.get("amount") or 0
    flags = []
    if total_cr and total_dr / total_cr > .9: flags.append("High spend ratio versus credits")
    if cats.get("bank_charges", 0): flags.append("Bank fees or penalty transactions detected")
    payload = {"result": f"Bank statement analysis for {document_id}: {len(txns)} transactions, avg monthly credit Rs {total_cr / months:,.0f}, avg monthly debit Rs {total_dr / months:,.0f}.", "document_id": document_id, "document_type": "bank_statement", "statement_start_date": dates[0] if dates else None, "statement_end_date": dates[-1] if dates else None, "transaction_count": len(txns), "average_monthly_credit": round(total_cr / months), "average_monthly_debit": round(total_dr / months), "estimated_monthly_surplus": round((total_cr - total_dr) / months), "savings_rate_pct": round((total_cr - total_dr) / total_cr * 100, 1) if total_cr else None, "top_expense_categories": sorted([{"category": k, "amount": round(v)} for k, v in cats.items()], key=lambda x: x["amount"], reverse=True)[:8], "emi_payments_count": sum(1 for t in txns if t.get("category") == "emi"), "fee_or_penalty_count": sum(1 for t in txns if t.get("category") == "bank_charges"), "risk_flags": flags}
    return _safe_payload(payload)

def _date_pat() -> str:
    return r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"


def _field_money(text: str, labels: list[str]) -> float | None:
    for label in labels:
        e = re.escape(label)
        for pat in [rf"{e}\s*[:=-]?\s*(?:Rs\.?|INR|\u20b9)?\s*([\d,]+(?:\.\d+)?)", rf"{e}[^\n]{{0,80}}?(?:Rs\.?|INR|\u20b9)\s*([\d,]+(?:\.\d+)?)"]:
            m = re.search(pat, text or "", flags=re.I)
            if m:
                return _parse_money(m.group(1))
    return None


def _field_date(text: str, labels: list[str]) -> str | None:
    for label in labels:
        m = re.search(rf"{re.escape(label)}[^\n]{{0,80}}?({_date_pat()})", text or "", flags=re.I)
        if m:
            return _parse_date(m.group(1))
    return None


def _enum(text: str, values: list[str]) -> str | None:
    lower = text.lower()
    return next((v for v in values if v in lower), None)


def _annualized(amount: float | None, freq: str | None) -> float | None:
    if amount is None:
        return None
    f = (freq or "annual").lower()
    mult = 12 if "month" in f else 4 if "quarter" in f else 2 if "half" in f else 1
    return round(amount * mult)


def parse_bill(document_id: str) -> dict[str, Any]:
    text = _all_text(extract_document_private(document_id))
    lower = text.lower()
    doc_type = "insurance_bill" if any(t in lower for t in ["policy", "premium", "sum assured", "life assured"]) else "utility_bill" if any(t in lower for t in ["electricity", "consumer", "units consumed", "kwh"]) else "bill"
    premium = _field_money(text, ["premium amount", "renewal premium", "gross premium"])
    frequency = _enum(text, ["monthly", "quarterly", "half yearly", "annual", "yearly", "single premium"])
    cover = _field_money(text, ["sum assured", "life cover", "cover amount", "coverage amount"])
    flags = []
    late_fee = _field_money(text, ["late fee", "surcharge", "delayed payment charge"])
    if late_fee: flags.append("Late fee or surcharge detected")
    annual = _annualized(premium, frequency)
    if annual and cover and annual / cover > .05: flags.append("Premium appears high relative to cover; review policy type and benefits")
    return _safe_payload({"result": f"Parsed {doc_type} document {document_id}.", "document_id": document_id, "document_type": doc_type, "bill_date": _field_date(text, ["bill date", "invoice date", "statement date"]), "due_date": _field_date(text, ["due date", "payment due date", "renewal due date", "pay by"]), "amount_due": _field_money(text, ["amount due", "total amount", "net payable", "bill amount", "amount payable"]), "late_fee": late_fee, "premium_amount": premium, "premium_frequency": frequency, "annualized_premium": annual, "sum_assured": cover, "policy_start_date": _field_date(text, ["policy start date", "commencement date", "risk commencement date"]), "maturity_date": _field_date(text, ["maturity date", "policy end date", "expiry date"]), "flags": flags})


def analyze_credit_card_statement(document_id: str) -> dict[str, Any]:
    text = _all_text(extract_document_private(document_id))
    amount_due = _field_money(text, ["total amount due", "amount due", "total due"])
    min_due = _field_money(text, ["minimum amount due", "minimum due"])
    limit = _field_money(text, ["credit limit", "total credit limit"])
    flags = []
    util = round(amount_due / limit * 100, 1) if amount_due and limit else None
    if util and util > 40: flags.append("Credit utilization appears high")
    if min_due and amount_due and min_due < amount_due: flags.append("Paying only minimum due can create high interest cost")
    if any(t in text.lower() for t in ["late fee", "finance charge", "interest", "cash advance"]): flags.append("Fees or interest terms detected")
    return _safe_payload({"result": f"Credit card statement analysis for {document_id}.", "document_id": document_id, "document_type": "credit_card_statement", "statement_date": _field_date(text, ["statement date", "bill date"]), "payment_due_date": _field_date(text, ["payment due date", "due date"]), "total_amount_due": amount_due, "minimum_amount_due": min_due, "credit_limit": limit, "utilization_pct": util, "risk_flags": flags})


def parse_salary_slip(document_id: str) -> dict[str, Any]:
    text = _all_text(extract_document_private(document_id))
    gross = _field_money(text, ["gross pay", "gross salary", "gross earnings"])
    net = _field_money(text, ["net pay", "take home", "salary credited", "net salary"])
    flags = []
    if gross and net and net / gross < .65: flags.append("Net pay is materially lower than gross; review deductions")
    return _safe_payload({"result": f"Salary slip parse for {document_id}.", "document_id": document_id, "document_type": "salary_slip", "pay_period": _field_date(text, ["pay period", "salary month", "month"]), "gross_pay": gross, "net_pay": net, "basic_salary": _field_money(text, ["basic", "basic salary"]), "hra": _field_money(text, ["hra", "house rent allowance"]), "provident_fund": _field_money(text, ["provident fund", "pf", "employee pf"]), "tds": _field_money(text, ["tds", "income tax"]), "annualized_gross_income": round(gross * 12) if gross else None, "annualized_net_income": round(net * 12) if net else None, "flags": flags})
