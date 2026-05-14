"""
One-shot patch script:
  1. Replaces QwenClassifier with guardrail-only version (lines 864-956)
  2. Replaces _patched_query_agentic_v7 + init + monkey-patch (lines 2374-2472)
     with the full ReAct implementation
Run: python scripts/patch_react.py
"""
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "banyanTreev3_agentic.py"

# ── new QwenClassifier (lines 864-956, 0-indexed 863-955) ──────────────────
CLASSIFIER = '''\
class QwenClassifier:
    # Guardrail-only: BLOCK or OK. No intent, no routing, no tool selection.
    FEW_SHOT = """Examples:
Q: "How do I hide income from IT department?"
A: {"guardrail":"BLOCK","reason":"TAX_EVASION"}
Q: "What is TCS share price today?"
A: {"guardrail":"OK","reason":""}
Q: "guaranteed 30% returns on crypto"
A: {"guardrail":"BLOCK","reason":"SCAM_RETURNS"}
Q: "Compare Infosys and Wipro PE ratio"
A: {"guardrail":"OK","reason":""}
Q: "How do I do pump and dump on penny stocks?"
A: {"guardrail":"BLOCK","reason":"MARKET_MANIPULATION"}
Q: "insider tip on HDFC Bank before results"
A: {"guardrail":"BLOCK","reason":"INSIDER_TRADING"}
Q: "pyramid scheme investment opportunity"
A: {"guardrail":"BLOCK","reason":"PONZI"}
Q: "non public information about reliance merger"
A: {"guardrail":"BLOCK","reason":"INSIDER_TRADING"}
Q: "How to start SIP in ELSS funds?"
A: {"guardrail":"OK","reason":""}
Q: "how to calculate EMI for home loan?"
A: {"guardrail":"OK","reason":""}"""

    SYSTEM = (
        "You are a safety guardrail for an Indian personal finance app.\\n"
        "Output ONLY a compact JSON with exactly two keys:\\n"
        "  guardrail : OK or BLOCK\\n"
        "  reason    : if BLOCK, one of: TAX_EVASION, SCAM_RETURNS, INSIDER_TRADING, "
        "MARKET_MANIPULATION, PONZI -- else empty string\\n"
        "BLOCK only for illegal/fraudulent/unethical requests. "
        "All legitimate finance questions (stocks, SIP, tax, loans, budgeting) are OK.\\n"
        "Output ONLY the JSON. No explanation. No markdown."
    )

    GUARDRAIL_FALLBACK = (
        "I cannot help with that request.\\n\\n"
        "I can assist with: budgeting, legal tax saving (80C/80D/NPS), goal-based investing, "
        "mutual funds, debt management, and reading live stock/market data.\\n\\n"
        "For personalized advice, consult a SEBI-registered advisor."
    )

    def __init__(self, tokenizer, generator, device: str, api_client=None):
        self.tokenizer  = tokenizer
        self.generator  = generator
        self.device     = device
        self.api_client = api_client
        print("API Guardrail ready" if self.api_client else "Qwen2.5 Guardrail ready")

    def classify(self, query: str) -> dict:
        user_msg = f"{self.FEW_SHOT}\\n\\nNow classify:\\nQ: \\"{query}\\"\\nA:"
        if self.api_client:
            raw = self.api_client.complete(self.SYSTEM, user_msg, max_tokens=40, temperature=0.0)
        else:
            prompt = build_qwen_prompt(self.tokenizer, self.SYSTEM, user_msg)
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2000,
                                    truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs, max_new_tokens=30, temperature=0.0, do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id)
            raw = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"Guardrail: {raw}")
        return self._parse(raw, query)

    def _parse(self, raw: str, query: str) -> dict:
        try:
            m = re.search(r'\\{[^}]+\\}', raw, re.DOTALL)
            if m:
                result    = json.loads(m.group())
                guardrail = str(result.get("guardrail", "OK")).upper()
                reason    = str(result.get("reason", ""))
                if guardrail not in ("OK", "BLOCK"):
                    guardrail = "OK"
                return {"guardrail": guardrail, "reason": reason}
        except Exception:
            pass
        raw_l = raw.lower()
        for kw, cat in [("tax_evasion", "TAX_EVASION"), ("scam", "SCAM_RETURNS"),
                        ("insider", "INSIDER_TRADING"), ("manipulation", "MARKET_MANIPULATION"),
                        ("ponzi", "PONZI")]:
            if kw in raw_l:
                return {"guardrail": "BLOCK", "reason": cat}
        return {"guardrail": "OK", "reason": ""}

'''

# ── ReAct replacement (replaces lines 2374-2472, 0-indexed 2373-2471) ───────
REACT = '''\
# =================================================================
# REACT AGENTIC LOOP v8
# Flow: PII redact -> guardrail (BLOCK/OK only) -> RAG+KG retrieval
#       -> context sufficient? yes -> generate() directly
#       -> no -> LLM-driven ReAct loop (Thought->Action->Observation)
#                up to _REACT_MAX_ITERATIONS, then force generate()
#       -> output guardrail -> memory
# =================================================================

_REACT_MAX_ITERATIONS = 4

_REACT_SYSTEM = (
    "You are FinSage, a ReAct reasoning agent for Indian personal finance.\\n"
    "Each step output EXACTLY one of:\\n\\n"
    "Option A - need a tool:\\n"
    "Thought: <reasoning>\\n"
    "Action: <tool_name>\\n"
    "Action Input: <JSON object>\\n\\n"
    "Option B - enough information:\\n"
    "Thought: <reasoning>\\n"
    "Final Answer: <2-5 sentences, cite sources, end with "
    "'Consult a SEBI-registered advisor.'>\\n\\n"
    "Rules: call a tool only when RAG context is clearly insufficient. "
    "Never hallucinate returns or guarantees. Indian finance context only."
)

_REACT_TOOL_MANIFEST = (
    "Tool: screener\\n"
    "When: share price, PE, fundamentals, market cap for a specific NSE stock\\n"
    \'Action Input: {"symbol": "TICKER"}\\n\\n\'
    "Tool: amfi_nav\\n"
    "When: mutual fund NAV or scheme prices\\n"
    \'Action Input: {"fund_filter": "fund name or category"}\\n\\n\'
    "Tool: sip_calculator\\n"
    "When: SIP maturity amount -- needs monthly amount, rate, years\\n"
    \'Action Input: {"query": "full user query"}\\n\\n\'
    "Tool: emi_calculator\\n"
    "When: loan EMI, total interest -- needs principal, rate, tenure\\n"
    \'Action Input: {"query": "full user query"}\\n\\n\'
    "Tool: portfolio_health\\n"
    "When: portfolio allocation review, rebalancing\\n"
    \'Action Input: {"query": "full user query"}\\n\\n\'
    "Tool: goal_planner\\n"
    "When: corpus / retirement planning with time horizon\\n"
    \'Action Input: {"query": "full user query"}\\n\\n\'
    "Tool: portfolio_multi_agent\\n"
    "When: compare multiple stocks, equity portfolio advice\\n"
    \'Action Input: {"query": "full user query", "symbols": ["TCS", "INFY"]}\\n\\n\'
    "Tool: search_rag\\n"
    "When: tax concepts (80C/80D/new vs old regime), PPF, NPS, insurance, budgeting\\n"
    \'Action Input: {"query": "specific search query"}\'
)


def _context_sufficient(docs: list) -> bool:
    """True if RAG+KG docs have enough substance to answer without tools."""
    if not docs:
        return False
    meaningful = [d for d in docs if len(d.get("content", "")) > 80]
    return len(meaningful) >= 2 and sum(len(d.get("content", "")) for d in meaningful) > 300


def _build_react_prompt(query: str, docs: list, observations: list) -> str:
    parts = []
    if docs:
        parts.append("=== RAG+KG Context ===")
        for i, d in enumerate(docs[:4], 1):
            title   = d.get("title", f"Doc{i}")
            snippet = d.get("content", "")[:350]
            parts.append(f"[{i}] {title}\\n{snippet}")
    if observations:
        parts.append("=== Tool Observations so far ===")
        parts.extend(observations)
    parts.append(f"=== Available Tools ===\\n{_REACT_TOOL_MANIFEST}")
    parts.append(f"=== User Question ===\\n{query}")
    parts.append("Output Thought + Action/Action Input  OR  Thought + Final Answer:")
    return "\\n\\n".join(parts)


def _parse_react_response(raw: str) -> dict:
    fa = re.search(r\'Final Answer\\s*:\\s*(.*)\', raw, re.DOTALL | re.IGNORECASE)
    if fa:
        return {"type": "final_answer", "content": fa.group(1).strip()}
    act = re.search(r\'Action\\s*:\\s*(\\w+)\', raw, re.IGNORECASE)
    inp = re.search(r\'Action Input\\s*:\\s*(\\{[^}]+\\})\', raw, re.DOTALL | re.IGNORECASE)
    if act:
        tool_name = act.group(1).strip().lower()
        tool_input: dict = {}
        if inp:
            try:
                tool_input = json.loads(inp.group(1))
            except Exception:
                tool_input = {"query": raw[:200]}
        return {"type": "action", "tool_call": {"tool": tool_name, **tool_input}}
    return {"type": "final_answer", "content": raw.strip()}


async def _react_loop(self, query, initial_docs, community_context, reasoning_paths, max_iterations):
    """ReAct engine: Thought->Action->Observation until Final Answer or max iterations."""
    accumulated_docs = list(initial_docs)
    observations: list = []
    all_tool_calls: list = []
    answer = None
    for iteration in range(max_iterations):
        print(f"REACT iter={iteration + 1}/{max_iterations}")
        raw = self.llm_client.complete(
            _REACT_SYSTEM,
            _build_react_prompt(query, accumulated_docs, observations),
            max_tokens=350, temperature=0.0,
        )
        print(f"REACT raw={raw[:200]!r}")
        parsed = _parse_react_response(raw)
        if parsed["type"] == "final_answer":
            answer = parsed["content"]
            print(f"REACT final_answer at iter={iteration + 1}")
            break
        tc = parsed["tool_call"]
        print(f"REACT action={tc.get(\'tool\',\'?\')} input={tc}")
        all_tool_calls.append(tc)
        live_docs = await self.mcp_client.execute([tc])
        if live_docs:
            self._snapshot_inject(live_docs)
            extra_docs, _, _ = self.retrieve(query, k=4)
            self._snapshot_restore()
            seen = {d["title"] for d in accumulated_docs}
            accumulated_docs = (
                live_docs
                + [d for d in extra_docs if d["title"] not in seen]
                + accumulated_docs
            )
            observations.append(f"Observation [{tc.get(\'tool\')}]: {live_docs[0][\'content\'][:300]}")
        else:
            observations.append(f"Observation [{tc.get(\'tool\')}]: No data returned. Try a different approach.")
    if answer is None:
        print(f"REACT max_iterations={max_iterations} reached -- forcing generate()")
        answer = self.generate(query, accumulated_docs[:6], community_context, reasoning_paths)
    return answer, all_tool_calls, accumulated_docs


async def _react_query_agentic(self, question: str, k: int = 8) -> dict:
    t0 = time.time()
    print(f"\\n{\'=\'*60}\\nQ {question}\\n{\'=\'*60}")

    # Step 1: PII redaction
    redacted_q, pii_found = self.pii_redactor.redact(question)
    if pii_found:
        print(f"REACT pii_redacted={pii_found}")

    # Step 2: Guardrail (BLOCK / OK only -- no intent, no routing)
    classification = self.classifier.classify(redacted_q)
    if classification["guardrail"] == "BLOCK":
        print(f"REACT BLOCKED [{classification[\'reason\']}]")
        return {
            "question": question, "answer": self.classifier.GUARDRAIL_FALLBACK,
            "blocked": True, "block_category": classification["reason"],
            "sources": [], "time": round(time.time() - t0, 2),
        }

    # Step 3: RAG+KG retrieval -- always runs first, unconditionally
    docs, community_context, reasoning_paths = self.retrieve(redacted_q, k=k)
    print(f"REACT rag_retrieved={len(docs)} docs")

    # Step 4a: Sufficient context -> direct generate(), no tools needed
    if _context_sufficient(docs):
        print("REACT context=sufficient -> direct generate()")
        answer = self.generate(redacted_q, docs[:6], community_context, reasoning_paths)
        mode, all_tool_calls, all_docs = "rag_direct", [], docs

    # Step 4b: Thin context -> LLM reasons and selects tools via ReAct loop
    else:
        print(f"REACT context=insufficient ({len(docs)} docs) -> ReAct loop (max={_REACT_MAX_ITERATIONS})")
        answer, all_tool_calls, all_docs = await _react_loop(
            self, redacted_q, docs, community_context, reasoning_paths, _REACT_MAX_ITERATIONS
        )
        mode = "react_agentic"

    # Step 5: Output guardrail
    out_safe, out_cat = self.output_guard.check(answer)
    if not out_safe:
        print(f"REACT output_flagged=[{out_cat}]")
        answer += self.output_guard.DISCLAIMER

    # Step 6: Memory + history
    elapsed = round(time.time() - t0, 2)
    self.memory.add(question, answer, {"tool_calls": [tc.get("tool") for tc in all_tool_calls]})
    self.query_history.append({
        "question": question, "answer": answer, "time": elapsed, "mode": mode,
        "retrieved_docs": [d["content"][:400] for d in all_docs[:4]],
        "tool_calls": [tc.get("tool") for tc in all_tool_calls],
    })
    print(f"REACT mode={mode} | elapsed={elapsed}s")
    print(f"ANSWER: {answer}")
    return {
        "question": question, "answer": answer, "blocked": False, "mode": mode,
        "classifier": classification,
        "tool_calls": [tc.get("tool") for tc in all_tool_calls],
        "sources": [d["title"] for d in all_docs[:6]],
        "retrieved_docs": [d["content"][:400] for d in all_docs[:4]],
        "kg_stats": {"entities": self.kg.number_of_nodes(), "relationships": self.kg.number_of_edges()},
        "time": elapsed,
    }


_ORIG_INIT_V7 = FINANCIAL_HIERARCHICAL_LIGHT_RAG.__init__


def _patched_init_v7(self, kg_db_path="finsage_kg_database"):
    _ORIG_INIT_V7(self, kg_db_path)
    # mcp_client is the only new dependency needed by the ReAct loop
    if not hasattr(self, "tool_registry"):
        self.tool_registry = ToolRegistry()
    self.tool_registry.specs["portfolio_multi_agent"] = {"route": "market"}
    self.mcp_client = BanyanTreeMCPToolClient(mcp_base=MCP_BASE)


FINANCIAL_HIERARCHICAL_LIGHT_RAG.__init__ = _patched_init_v7
FINANCIAL_HIERARCHICAL_LIGHT_RAG.query_agentic = _react_query_agentic
'''

def find_line(lines, marker, start=0):
    for i in range(start, len(lines)):
        if marker in lines[i]:
            return i
    raise ValueError(f"Marker not found after line {start}: {marker!r}")

def patch():
    text = SRC.read_text(encoding="utf-8-sig")
    lines = text.splitlines(keepends=True)
    print(f"File: {len(lines)} lines")

    # ── Patch 1: QwenClassifier ─────────────────────────────────────────────
    cls_start = find_line(lines, "class QwenClassifier:")
    # End = last line of _parse() which returns intent; search forward for MODULE 4 header
    cls_end = find_line(lines, "# MODULE 4 : QWEN MCP TOOL SELECTOR", cls_start) - 2
    # cls_end should land on the blank line just before the header; walk back to last code line
    while cls_end > cls_start and lines[cls_end].strip() == "":
        cls_end -= 1
    print(f"  QwenClassifier: lines {cls_start+1}-{cls_end+1}")
    lines[cls_start:cls_end + 1] = [CLASSIFIER]
    print("Patch 1 applied: QwenClassifier -> guardrail-only")

    # ── Patch 2: _patched_query_agentic_v7 block ────────────────────────────
    v7_start = find_line(lines, "async def _patched_query_agentic_v7")
    v7_end   = find_line(lines, "FINANCIAL_HIERARCHICAL_LIGHT_RAG.query_agentic = _patched_query_agentic_v7", v7_start)
    print(f"  v7 block: lines {v7_start+1}-{v7_end+1}")
    lines[v7_start:v7_end + 1] = [REACT]
    print("Patch 2 applied: _patched_query_agentic_v7 -> _react_query_agentic")

    SRC.write_text("".join(lines), encoding="utf-8")
    print(f"Written: {SRC} ({len(lines)} lines after patches)")

if __name__ == "__main__":
    patch()
