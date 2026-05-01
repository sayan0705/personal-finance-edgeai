"""Schema converters — normalise each HuggingFace dataset into the unified ChatML format."""

from __future__ import annotations

from typing import Any, Optional

from .registry import TWITTER_TOPIC_MAP


def _trim(d: dict[str, Any], max_len: int = 200) -> dict[str, str]:
    return {k: str(v)[:max_len] for k, v in d.items() if v}


def convert_instruction_io(
    row: dict[str, Any], source: str, idx: int, system_prompt: str, task: str = "instruction"
) -> Optional[dict[str, Any]]:
    """Convert instruction/input/output format (FinGPT, Finance-Alpaca, FinTalk).

    Args:
        row: Raw dataset row.
        source: HuggingFace dataset ID used as provenance tag.
        idx: Row index within the sample.
        system_prompt: System-role content to inject.
        task: Task type label for the output record.

    Returns:
        Normalised record dict, or ``None`` if the row is unusable.
    """
    instruction = str(row.get("instruction", "") or "")
    inp = str(row.get("input", "") or "")
    output = str(row.get("output", "") or "")

    if not output or len(output.strip()) < 5:
        return None

    user_msg = f"{instruction}\n\n{inp}" if (instruction.strip() and inp.strip()) else (instruction or inp)
    if not user_msg.strip():
        return None

    return {
        "id": f"{source}_{idx:06d}",
        "source_dataset": source,
        "task_type": task,
        "language": "en",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": output.strip()},
        ],
        "original_columns": _trim(row),
    }


def convert_sentiment_label(
    row: dict[str, Any],
    source: str,
    idx: int,
    text_col: str,
    label_col: str,
    label_map: Optional[dict] = None,
) -> Optional[dict[str, Any]]:
    """Convert text + label sentiment datasets (FPB, Twitter).

    Args:
        row: Raw dataset row.
        source: HuggingFace dataset ID.
        idx: Row index.
        text_col: Column name containing the input text.
        label_col: Column name containing the label.
        label_map: Optional mapping from numeric/raw label to human-readable string.

    Returns:
        Normalised record dict, or ``None`` if unusable.
    """
    text = str(row.get(text_col, "") or "")
    label = row.get(label_col, "")

    if not text.strip() or label is None:
        return None

    label_str = label_map.get(label, str(label)) if label_map else str(label)

    return {
        "id": f"{source}_{idx:06d}",
        "source_dataset": source,
        "task_type": "sentiment",
        "language": "en",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a financial sentiment analysis expert. "
                    "Analyse the sentiment of financial text and classify it accurately."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Analyse the sentiment of the following financial text. "
                    f"Classify as positive, negative, or neutral. Explain your reasoning briefly.\n\n"
                    f'Text: "{text.strip()}"'
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"Sentiment: **{label_str}**\n\n"
                    f"This text expresses a {label_str} sentiment based on the financial language and context used."
                ),
            },
        ],
        "original_columns": _trim(row),
    }


def convert_twitter_topic(row: dict[str, Any], source: str, idx: int) -> Optional[dict[str, Any]]:
    """Convert Twitter financial news topic classification rows.

    Args:
        row: Raw dataset row.
        source: HuggingFace dataset ID.
        idx: Row index.

    Returns:
        Normalised record dict, or ``None`` if unusable.
    """
    text = str(row.get("text", "") or "")
    label = TWITTER_TOPIC_MAP.get(row.get("label", -1), "Unknown")

    if not text.strip():
        return None

    return {
        "id": f"{source}_{idx:06d}",
        "source_dataset": source,
        "task_type": "classification",
        "language": "en",
        "messages": [
            {
                "role": "system",
                "content": "You are a financial news classifier. Categorise financial tweets into their primary topic.",
            },
            {
                "role": "user",
                "content": (
                    f"Classify the following financial tweet into its primary topic category.\n\n"
                    f'Tweet: "{text.strip()}"'
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"Topic: **{label}**\n\n"
                    f"This tweet is classified under '{label}' based on the financial entities and context mentioned."
                ),
            },
        ],
        "original_columns": _trim(row),
    }


def convert_sujet(
    row: dict[str, Any], source: str, idx: int, system_prompt: str
) -> Optional[dict[str, Any]]:
    """Convert Sujet Finance format (inputs/answer/system_prompt/user_prompt).

    Args:
        row: Raw dataset row.
        source: HuggingFace dataset ID.
        idx: Row index.
        system_prompt: Fallback system prompt if the row has none.

    Returns:
        Normalised record dict, or ``None`` if unusable.
    """
    system = str(row.get("system_prompt", "") or system_prompt)
    user = str(row.get("user_prompt", "") or row.get("inputs", "") or "")
    answer = str(row.get("answer", "") or "")
    task = str(row.get("task_type", "") or "instruction")

    if not user.strip() or not answer.strip() or len(answer.strip()) < 10:
        return None

    return {
        "id": f"{source}_{idx:06d}",
        "source_dataset": source,
        "task_type": task,
        "language": "en",
        "messages": [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": answer.strip()},
        ],
        "original_columns": _trim(row),
    }


def convert_adaptllm(
    row: dict[str, Any], source: str, idx: int, system_prompt: str
) -> Optional[dict[str, Any]]:
    """Convert AdaptLLM format (input/output or context-based).

    Args:
        row: Raw dataset row.
        source: HuggingFace dataset ID.
        idx: Row index.
        system_prompt: System-role content to inject.

    Returns:
        Normalised record dict, or ``None`` if unusable.
    """
    inp = str(row.get("input", "") or row.get("query", "") or "")
    out = str(row.get("output", "") or row.get("answer", "") or "")
    context = str(row.get("context", "") or "")

    if not inp.strip() or not out.strip():
        return None

    user_msg = f"Context: {context}\n\nQuestion: {inp}" if context.strip() else inp

    return {
        "id": f"{source}_{idx:06d}",
        "source_dataset": source,
        "task_type": "qa_context" if context.strip() else "qa",
        "language": "en",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": out.strip()},
        ],
        "original_columns": _trim(row),
    }


def convert_generic_qa(
    row: dict[str, Any],
    source: str,
    idx: int,
    system_prompt: str,
    q_col: str = "question",
    a_col: str = "answer",
    ctx_col: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Generic QA converter for any dataset with question/answer columns.

    Args:
        row: Raw dataset row.
        source: HuggingFace dataset ID.
        idx: Row index.
        system_prompt: System-role content to inject.
        q_col: Column name for the question / user turn.
        a_col: Column name for the answer / assistant turn.
        ctx_col: Optional column name containing supporting context.

    Returns:
        Normalised record dict, or ``None`` if unusable.
    """
    question = str(row.get(q_col, "") or "")
    answer = str(row.get(a_col, "") or "")
    context = str(row.get(ctx_col, "") or "") if ctx_col else ""

    if not question.strip() or not answer.strip():
        return None

    user_msg = f"Context: {context}\n\nQuestion: {question}" if context.strip() else question

    return {
        "id": f"{source}_{idx:06d}",
        "source_dataset": source,
        "task_type": "qa_context" if context.strip() else "qa",
        "language": "en",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": answer.strip()},
        ],
        "original_columns": _trim(row),
    }


def convert_instruction_response(
    row: dict[str, Any], source: str, idx: int, system_prompt: str
) -> Optional[dict[str, Any]]:
    """Convert instruction + response format (FinTalk-19k and similar datasets).

    FinTalk-19k uses ``instruction`` / ``response`` / ``context`` / ``tag``
    columns — it has no ``input`` or ``output`` fields, so Pattern 1 misses it
    and the fallback incorrectly picks the ``tag`` column as the assistant turn.

    Args:
        row: Raw dataset row with ``instruction``, ``response``, and optionally
             ``context`` keys.
        source: HuggingFace dataset ID.
        idx: Row index.
        system_prompt: System-role content to inject.

    Returns:
        Normalised record dict, or ``None`` if unusable.
    """
    instruction = str(row.get("instruction", "") or "").strip()
    response = str(row.get("response", "") or "").strip()
    context = str(row.get("context", "") or "").strip()

    if not instruction or not response or len(response) < 10:
        return None

    user_msg = f"Context: {context}\n\n{instruction}" if context else instruction

    return {
        "id": f"{source}_{idx:06d}",
        "source_dataset": source,
        "task_type": "instruction",
        "language": "en",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},
        ],
        "original_columns": _trim(row),
    }


def convert_system_user_assistant(
    row: dict[str, Any], source: str, idx: int
) -> Optional[dict[str, Any]]:
    """Convert pre-formatted system/user/assistant columns (Finance-Instruct-500k).

    Args:
        row: Raw dataset row with ``system``, ``user``, ``assistant`` keys.
        source: HuggingFace dataset ID.
        idx: Row index.

    Returns:
        Normalised record dict, or ``None`` if unusable.
    """
    system = str(row.get("system", "") or "").strip()
    user = str(row.get("user", "") or "").strip()
    assistant = str(row.get("assistant", "") or "").strip()

    if not user or not assistant or len(assistant) < 5:
        return None

    return {
        "id": f"{source}_{idx:06d}",
        "source_dataset": source,
        "task_type": "instruction",
        "language": "en",
        "messages": [
            {"role": "system", "content": system or "You are a helpful financial assistant."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "original_columns": _trim(row),
    }


def convert_finentity_ner(
    row: dict[str, Any], source: str, idx: int
) -> Optional[dict[str, Any]]:
    """Convert FinEntity NER rows to an instruction format.

    The dataset has ``content`` (financial text) and ``annotations`` (entity spans).
    We convert annotations into a readable entity list for the assistant turn.

    Args:
        row: Raw dataset row with ``content`` and ``annotations`` keys.
        source: HuggingFace dataset ID.
        idx: Row index.

    Returns:
        Normalised record dict, or ``None`` if unusable.
    """
    import json as _json

    content = str(row.get("content", "") or "").strip()
    raw_annotations = row.get("annotations", [])

    if not content:
        return None

    # Parse annotations — may be a JSON string or a list
    if isinstance(raw_annotations, str):
        try:
            raw_annotations = _json.loads(raw_annotations)
        except Exception:
            raw_annotations = []

    # Build entity list from span annotations
    entities: list[str] = []
    if isinstance(raw_annotations, list):
        for ann in raw_annotations:
            if isinstance(ann, dict):
                entity_text = ann.get("text") or ann.get("entity") or ""
                entity_type = ann.get("type") or ann.get("label") or "ENTITY"
                if entity_text:
                    entities.append(f"{entity_text} ({entity_type})")

    if not entities:
        # Still useful as an NER negative example (no entities found)
        assistant_text = "No named financial entities were identified in this text."
    else:
        entity_list = "\n".join(f"- {e}" for e in entities)
        assistant_text = f"The following financial entities were identified:\n{entity_list}"

    return {
        "id": f"{source}_{idx:06d}",
        "source_dataset": source,
        "task_type": "ner",
        "language": "en",
        "messages": [
            {
                "role": "system",
                "content": "You are a financial named entity recognition expert. Identify and classify all financial entities in the given text.",
            },
            {
                "role": "user",
                "content": f"Identify all financial entities (companies, people, monetary values, financial instruments, etc.) in the following text:\n\n{content}",
            },
            {"role": "assistant", "content": assistant_text},
        ],
        "original_columns": _trim(row),
    }


def convert_indian_itr(
    row: dict[str, Any], source: str, idx: int
) -> Optional[dict[str, Any]]:
    """Convert Indian ITR (Income Tax Return) rows into tax advisory QA pairs.

    The AgamiAI dataset has rich ITR fields (PAN, AY, financials JSON).
    We synthesise a tax-return summary question from the structured data.

    Args:
        row: Raw dataset row with ITR fields.
        source: HuggingFace dataset ID.
        idx: Row index.

    Returns:
        Normalised record dict, or ``None`` if unusable.
    """
    import json as _json

    name = str(row.get("name", "") or "").strip()
    ay = str(row.get("assessment_year", "") or "").strip()
    entity = str(row.get("entity", "") or "Individual").strip()
    form = str(row.get("form", "") or "ITR").strip()
    state = str(row.get("state", "") or "").strip()

    financials_raw = row.get("financials", "")
    if isinstance(financials_raw, str):
        try:
            financials = _json.loads(financials_raw)
        except Exception:
            financials = {}
    elif isinstance(financials_raw, dict):
        financials = financials_raw
    else:
        financials = {}

    if not financials:
        return None

    # Build a human-readable summary of the financials
    summary_lines: list[str] = []
    for section, values in financials.items():
        if isinstance(values, dict):
            for key, val in list(values.items())[:5]:
                if val not in (None, "", 0):
                    summary_lines.append(f"  {key}: ₹{val:,}" if isinstance(val, (int, float)) else f"  {key}: {val}")
        elif values not in (None, "", 0):
            summary_lines.append(f"  {section}: ₹{values:,}" if isinstance(values, (int, float)) else f"  {section}: {values}")

    if not summary_lines:
        return None

    financials_text = "\n".join(summary_lines[:15])
    user_question = (
        f"The following details are from an Indian Income Tax Return ({form}) "
        f"filed by a {entity} for Assessment Year {ay or 'N/A'}"
        + (f" in {state}" if state else "")
        + f":\n\n{financials_text}\n\n"
        f"Please summarise the key financial figures and explain what these entries mean "
        f"in the context of Indian income tax filing."
    )

    assistant_answer = (
        f"This {form} for AY {ay or 'N/A'} reflects the following key figures:\n\n"
        f"{financials_text}\n\n"
        f"Under Indian income tax rules, these entries represent income earned, deductions "
        f"claimed, and taxes paid during the financial year. The taxpayer (a {entity}) "
        f"should verify all figures against Form 16, Form 26AS, and AIS before filing."
    )

    return {
        "id": f"{source}_{idx:06d}",
        "source_dataset": source,
        "task_type": "indian_tax",
        "language": "en",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are FinEdge, an expert Indian tax advisor. "
                    "Provide accurate explanations of ITR entries with references to "
                    "relevant sections of the Income Tax Act."
                ),
            },
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": assistant_answer},
        ],
        "original_columns": _trim(row),
    }


def convert_finqa(
    row: dict[str, Any], source: str, idx: int
) -> Optional[dict[str, Any]]:
    """Convert FinQA rows (ibm/finqa) into numerical-reasoning QA pairs.

    FinQA rows contain SEC financial report context (pre_text, table, post_text)
    paired with a question that requires arithmetic reasoning to answer.

    Args:
        row: Raw dataset row with ``pre_text``, ``post_text``, ``table``, and
             ``qa`` (dict with ``question``, ``exe_ans``, ``program``) keys.
        source: HuggingFace dataset ID.
        idx: Row index.

    Returns:
        Normalised record dict, or ``None`` if unusable.
    """
    qa = row.get("qa") or {}
    if isinstance(qa, str):
        import json as _json
        try:
            qa = _json.loads(qa)
        except Exception:
            qa = {}

    question = str(qa.get("question", "") or "").strip()
    exe_ans = qa.get("exe_ans", None)
    program = str(qa.get("program", "") or "").strip()

    if not question or exe_ans is None:
        return None

    # Build context from financial report snippets
    pre_text = row.get("pre_text") or []
    post_text = row.get("post_text") or []
    table = row.get("table") or []

    pre = " ".join(str(p) for p in pre_text).strip()
    post = " ".join(str(p) for p in post_text).strip()

    # Format table as plain text rows
    table_lines: list[str] = []
    if isinstance(table, list):
        for table_row in table:
            if isinstance(table_row, list):
                table_lines.append(" | ".join(str(cell) for cell in table_row))
            else:
                table_lines.append(str(table_row))
    table_text = "\n".join(table_lines).strip()

    context_parts = [p for p in [pre, table_text, post] if p]
    if not context_parts:
        return None
    context = "\n\n".join(context_parts)

    # Format the answer with the program steps if available
    answer_str = str(exe_ans)
    if program:
        answer_str = f"{exe_ans}\n\nCalculation steps: {program}"

    user_msg = (
        f"Use the following financial report excerpt to answer the question.\n\n"
        f"--- Financial Report Context ---\n{context}\n"
        f"--- End of Context ---\n\n"
        f"Question: {question}"
    )

    return {
        "id": f"{source}_{idx:06d}",
        "source_dataset": source,
        "task_type": "financial_qa",
        "language": "en",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a financial analyst expert. Answer questions about financial reports "
                    "using numerical reasoning. Show your calculation steps clearly and provide "
                    "the final numerical answer."
                ),
            },
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer_str},
        ],
        "original_columns": _trim(row),
    }


def detect_and_convert(
    row: dict[str, Any], source: str, idx: int, dataset_key: str, system_prompt: str
) -> Optional[dict[str, Any]]:
    """Auto-detect a row's schema and apply the correct converter.

    Handles all known column layouts across the 13 registered datasets.

    Args:
        row: Raw dataset row as a plain dict.
        source: HuggingFace dataset ID (used as provenance tag).
        idx: Row index within the sample.
        dataset_key: Key in DATASET_REGISTRY (used for dataset-specific branching).
        system_prompt: Default system-role content to inject.

    Returns:
        Normalised record dict, or ``None`` if the row cannot be converted.
    """
    columns = set(row.keys())

    # Pattern 0: pre-formatted system / user / assistant (Finance-Instruct-500k)
    if {"system", "user", "assistant"}.issubset(columns):
        return convert_system_user_assistant(row, source, idx)

    # Pattern 0a: FinQA schema (ibm/finqa) — pre_text + table + post_text + qa
    if dataset_key == "finqa" or ("qa" in columns and "pre_text" in columns):
        return convert_finqa(row, source, idx)

    # Pattern 0b: Indian ITR schema (AgamiAI)
    if dataset_key == "indian_itr" or ({"pan", "financials"}.issubset(columns)):
        return convert_indian_itr(row, source, idx)

    # Pattern 0c: FinEntity NER schema
    if dataset_key == "finentity" or ("content" in columns and "annotations" in columns):
        return convert_finentity_ner(row, source, idx)

    # Pattern 1: instruction / input / output (FinGPT, Alpaca)
    if {"instruction", "input", "output"}.issubset(columns):
        return convert_instruction_io(row, source, idx, system_prompt)

    # Pattern 1b: instruction + response without input/output (FinTalk-19k)
    # Must come before the fallback — otherwise the ``tag`` column (e.g. "Financial
    # Information") gets picked as the assistant turn instead of ``response``.
    if "instruction" in columns and "response" in columns and "output" not in columns:
        return convert_instruction_response(row, source, idx, system_prompt)

    # Pattern 2: inputs / answer / system_prompt (Sujet Finance)
    if "inputs" in columns and "answer" in columns:
        return convert_sujet(row, source, idx, system_prompt)

    # Pattern 3: sentence + label (Financial PhraseBank)
    if "sentence" in columns and "label" in columns:
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        return convert_sentiment_label(row, source, idx, "sentence", "label", label_map)

    # Pattern 4a: text + label — Twitter sentiment
    if "text" in columns and "label" in columns:
        if dataset_key == "twitter_fin_sentiment":
            label_map = {0: "Bearish (Negative)", 1: "Bullish (Positive)", 2: "Neutral"}
            return convert_sentiment_label(row, source, idx, "text", "label", label_map)
        if dataset_key == "twitter_fin_topics":
            return convert_twitter_topic(row, source, idx)
        return convert_sentiment_label(row, source, idx, "text", "label")

    # Pattern 5: input / output without instruction (AdaptLLM)
    if "input" in columns and "output" in columns and "instruction" not in columns:
        return convert_adaptllm(row, source, idx, system_prompt)

    # Pattern 6: question / answer / optional context
    if "question" in columns and "answer" in columns:
        ctx = "context" if "context" in columns else None
        return convert_generic_qa(row, source, idx, system_prompt, "question", "answer", ctx)

    # Pattern 7: query + response
    if "query" in columns and "response" in columns:
        return convert_generic_qa(row, source, idx, system_prompt, "query", "response")

    # Pattern 8: prompt / completion
    if "prompt" in columns and "completion" in columns:
        return convert_generic_qa(row, source, idx, system_prompt, "prompt", "completion")

    # Pattern 9: context / question / answer
    if "context" in columns and "question" in columns:
        a_col = next((c for c in ("answer", "output") if c in columns), None)
        if a_col:
            return convert_generic_qa(row, source, idx, system_prompt, "question", a_col, "context")

    # Fallback: use any two sufficiently long text columns.
    # Iterate row.keys() (dict-ordered, Python 3.7+) — NOT the set ``columns``
    # which has no stable iteration order and would give non-deterministic results.
    text_cols = [c for c in row.keys() if row.get(c) and isinstance(row.get(c), str) and len(str(row.get(c))) > 20]
    if len(text_cols) >= 2:
        return {
            "id": f"{source}_{idx:06d}",
            "source_dataset": source,
            "task_type": "unknown",
            "language": "en",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(row[text_cols[0]])[:1000].strip()},
                {"role": "assistant", "content": str(row[text_cols[1]])[:2000].strip()},
            ],
            "original_columns": _trim(row),
        }

    return None
