# Eval Metrics Summary — Personal Finance Edge AI Agent

> Synthesized from research across Claude, ChatGPT, and Gemini eval guides.
> Scoped to: Indian personal finance domain · on-device SLM (GGUF/llama-cpp) · ReAct agentic workflow

---

## Core Evaluation Philosophy

Single-axis accuracy is insufficient. All three research sources converge on the same conclusion: evaluation must span three traditions simultaneously.

| Lens | What It Measures | Primary Failure It Catches |
|---|---|---|
| **Agentic** | Did the agent reason, plan, select tools, complete the task? | Wrong tool called, redundant loops, stuck reasoning |
| **Finance Domain** | Are the numbers right? Is the advice safe and regulation-compliant? | Calculation errors, hallucinated returns, unregulated advice |
| **Edge / Systems** | Latency, memory, energy, thermal behavior | Throttling, OOM kills, battery drain, GIL contention |

**Key rule:** Agents evaluated only on final-output quality pass 20–40% more test cases than full trajectory evaluation reveals. Trajectory-level eval is non-negotiable.

---

## CLEAR Framework — Master Scoring Rubric

The CLEAR framework (2025) predicts production success at ρ=0.83 vs ρ=0.41 for accuracy-only evaluation. Instantiated for this project:

| Dimension | What to Measure | How to Measure | Target |
|---|---|---|---|
| **Cost** | Inference cost per query (device compute + power) | tokens × energy-per-token; ₹/month device cost | Cost-Normalized Accuracy (CNA) = accuracy / cost-per-task |
| **Latency** | TTFT, inter-token latency, end-to-end task time | p50, p95, p99 — cold boot and sustained (hot) | < 2s TTFT; 40–50 tokens/sec generation |
| **Efficiency** | RAM peak, disk size, energy per inference | tegrastats / powermetrics / RAPL | < 4 GB RAM; < 2 GB model; < 5 J/query |
| **Assurance** | Refusal rate on regulated advice, PII safety, policy adherence | Promptfoo red-team + custom rubric | Policy Adherence Score (PAS) > 95% |
| **Reliability** | Consistency across repeated runs | pass@1 and pass@5 on same prompts | pass@5 variance < 10% |

---

## Metric Catalog by Category

### 1. Task / Agent Metrics

| Metric | Definition | Tool |
|---|---|---|
| **Task Completion Rate** | % of finance queries resolved end-to-end correctly | Custom benchmark + LLM-as-judge |
| **Tool-Call Accuracy** | Correct tool selected + correct JSON input (AST check) | BFCL-style evaluation |
| **Redundant Call Rate** | # unnecessary tool calls per task | Langfuse trace analysis |
| **Recovery-from-Error Rate** | % tasks completed after a failed tool call | Trajectory replay |
| **Multi-turn Consistency** | Does the agent contradict itself across turns? | τ-bench style harness |
| **Relevance Detection** | Does the agent know when NOT to call a tool? | BFCL v4 relevance detection category |
| **pass@1 / pass@k** | Single-run success / success in k attempts | Run each case 5×, compute both |
| **Plan Depth** | Number of reasoning steps executed correctly | AgentBench-style step counting |

### 2. Finance Domain Metrics (India-specific)

| Metric | Definition | Test Case Type |
|---|---|---|
| **Tax Calculation Accuracy** | Correct tax payable under old/new regime (FY 2024-25 slabs) | Unit tests + custom benchmark |
| **SIP/EMI Calculation Accuracy** | Correct maturity amount, XIRR, EMI, amortization schedule | Deterministic math check |
| **Transaction Categorization F1** | Precision/Recall/F1 on expense categories (Food, Rent, Utilities…) | Kaggle-style labeled dataset |
| **Budget Recommendation Quality** | LLM-as-judge rubric on actionability and correctness | Human + judge scoring |
| **RAG Faithfulness** | Does the answer stay grounded in retrieved SEBI/RBI docs? | RAGAS faithfulness metric |
| **RAG Answer Relevancy** | Is the retrieved context relevant to the query? | RAGAS answer relevancy |
| **RAG Context Precision/Recall** | Is the right document chunk retrieved? | RAGAS context metrics |
| **Hallucination Rate** | % responses containing fabricated financial figures | DeepEval hallucination metric |
| **Forecast Error (MAPE)** | Mean Absolute Percentage Error on spending forecasts | Time-series eval on synthetic data |
| **Debt Strategy Accuracy** | Correct snowball/avalanche math and recommendation | Custom test cases |
| **Goal Planning Accuracy** | Correct emergency fund, retirement corpus calculations | Custom test cases |
| **Source Citation Rate** | % RAG responses that cite SEBI/RBI source + date | Metadata check in pipeline |

### 3. Edge / Systems Metrics

| Metric | Definition | Tool |
|---|---|---|
| **TTFT (Time-to-First-Token)** | Milliseconds from prompt submission to first token | llama.cpp timing logs |
| **Inter-Token Latency** | Average ms between tokens during generation | llama.cpp / tegrastats |
| **Throughput (tokens/sec)** | Sustained generation speed under normal load | llama-bench |
| **Thermal Throughput** | tokens/sec after 10–15 min continuous use (throttled state) | Sustained load test loop |
| **Peak RAM** | Maximum resident memory during inference | OS memory profiler |
| **Model Disk Size** | GGUF file size post-quantization | File system stat |
| **Energy per Query (J or mWh)** | Power consumed per end-to-end response | RAPL / powermetrics / physical meter |
| **Tokens per Joule** | Efficiency index = throughput / power draw | Derived metric |
| **Cold Boot Latency** | Latency when model loads from disk (first query) | Timed cold run |
| **Quantization BPW Curve** | Accuracy vs effective bits-per-weight across Q4/Q5/Q8 | Sweep: Q4_K_M, Q5_K_M, Q8_0 |
| **Blocking Ratio (β)** | GIL contention vs genuine I/O wait in Python layer | Lightweight profiler |
| **Memory Bandwidth Utilization** | GB/s consumed during inference | Performance counters |

> **Key finding from research:** Heavily quantized large models (e.g. Qwen2-7B at Q4) consistently outperform smaller high-precision models. Accuracy threshold appears around ~3.5 effective bits-per-weight. Always plot the BPW curve.

### 4. Safety & Compliance Metrics

| Metric | Definition | Tool |
|---|---|---|
| **Policy Adherence Score (PAS)** | % queries where agent correctly refuses out-of-scope financial advice | Promptfoo + manual rubric |
| **PII Leakage Rate** | Rate of unintended personal data exposure in responses | Membership inference test / Giskard |
| **Prompt Injection Resistance** | Does the agent ignore hidden instructions in uploaded PDFs/CSVs? | Promptfoo red-team |
| **Jailbreak Resistance** | Does the agent resist attempts to bypass regulated-advice refusals? | Promptfoo + Agent-SafetyBench |
| **Tool-Call Abuse Rate** | Does the agent call transactional tools (budget_save, etc.) when not instructed? | Agent-SafetyBench subset |
| **Safety Violation Rate** | # of responses that cross regulatory / ethical lines | Agent-SafetyBench (8 categories) |
| **Cross-Turn Constraint Adherence** | Does the agent remember and honor user-specified risk tolerance across turns? | CURATe-style multi-turn test |
| **Fairness** | Demographic parity in advice (if agent handles user profiles) | Disparate impact check |

---

## Eval Layer Pyramid

Four layers ordered from cheapest/fastest to most expensive. Run all four.

```
Layer 4 ─ Safety / Red-team      (manual, gated, ~once per milestone)
Layer 3 ─ Edge / Systems         (hardware profiling, ~per model variant)
Layer 2 ─ Trajectory / Agent     (nightly, full multi-turn benchmark)
Layer 1 ─ Unit-level             (every commit, <5 min)
```

### Layer 1 — Unit-level (every commit)
- DeepEval assertions: format correctness, refusal triggers, JSON schema validity
- BFCL-style AST checks on every tool call (tax_calculator, sip_calculator, loan_advisor)
- RAGAS metrics on individual RAG retrievals: faithfulness, relevancy, precision
- Deterministic math checks: SIP/EMI/tax outputs against known-good values

### Layer 2 — Trajectory / Agent (nightly)
- **Custom India Personal Finance Benchmark** — 50 hand-crafted cases, 4 categories:
  - 15 × Tax & investment Q&A (FY 2024-25 slabs, 80C/80D deductions, SIP returns)
  - 10 × Debt strategy (EMI calculation, prepayment, snowball vs avalanche)
  - 10 × Goal planning (emergency fund, retirement corpus, insurance coverage)
  - 15 × Refusal / scope-limiting (stock tips, guaranteed returns, regulated advice)
- LLM-as-judge scoring with rubrics; validate judge against human grading on 10 cases first (target ≥ 80% agreement)
- τ-bench style 5–10 multi-turn scenarios using a simulated user (bigger LLM)
- Trajectory metrics: tool-call accuracy, redundant-call rate, recovery rate

### Layer 3 — Edge / Systems
- Latency p50/p95/p99 on target hardware — cold and warm
- Sustained-load test: run multi-turn scenarios in tight loop for 10–15 min, measure throttling
- Memory and energy per query
- Quantization sweep: Q4_K_M → Q5_K_M → Q8_0 accuracy vs BPW curve
- Blocking ratio (β) profile on the Python ReAct orchestration layer

### Layer 4 — Safety / Red-team
- Promptfoo config covering 6 attack classes:
  1. Prompt injection via user message
  2. Indirect injection via uploaded bank statement / PDF
  3. Jailbreaks targeting regulated advice refusals
  4. PII leakage / cross-turn data contamination
  5. Tool-call abuse (unasked budget writes, transfers)
  6. Data poisoning via user interaction feedback loop
- Agent-SafetyBench subset (finance-relevant categories)
- Document failure-mode taxonomy

---

## Recommended Benchmarks

| Benchmark | What It Tests | Fit for This Project | Priority |
|---|---|---|---|
| **BFCL v4** | Tool/function-calling accuracy via AST checks | HIGH — every tool call must be evaluated | Must-have |
| **Custom India Finance Benchmark** | Personal finance accuracy (our 50 cases) | HIGH — most project-specific signal | Must-have |
| **τ-bench harness (adapted)** | Multi-turn policy adherence, simulated user | HIGH — ReAct agent must respect policy | Must-have |
| **RAGAS** | RAG: faithfulness, relevancy, context precision | HIGH — SEBI/RBI document grounding | Must-have |
| **FinAgentBench** | RAG + multi-step financial document retrieval | MEDIUM — good for RAG pipeline validation | Recommended |
| **Agent-SafetyBench (subset)** | Safety: 8 risk categories, 10 failure modes | HIGH — finance is regulated domain | Must-have |
| **GAIA Level 2** | General multi-step reasoning with tools | MEDIUM — literature anchor | Optional |
| **Finance Agent Benchmark (methodology)** | Expert rubric structure to borrow | HIGH (methodology) / LOW (as-is) | Borrow methodology |

---

## Tooling Stack

| Tool | Role | Why This Project |
|---|---|---|
| **Langfuse** (self-hosted Docker) | Trace every agent step, dataset versioning, prompt management | Free, MIT, framework-agnostic — works with llama-cpp |
| **DeepEval** | Metric assertions in CI (pytest-style), 50+ research-backed metrics | Apache-2.0, integrates with Langfuse |
| **RAGAS** | RAG-specific scoring (faithfulness, relevancy, context recall) | Purpose-built; academically defensible |
| **Promptfoo** | Red-team and security eval via declarative YAML | MIT; finance-specific attack configs |
| **Giskard** | RAG compliance, EU AI Act / NIST RMF alignment checks | Strong on regulatory framing |
| **llama-bench** | Latency and throughput profiling for GGUF models | Native to llama.cpp stack |
| **tegrastats / RAPL / powermetrics** | Energy and memory profiling on edge hardware | Platform-native power measurement |

---

## India-specific Eval Dimensions

These are NOT covered by any published benchmark and must be built custom:

| Scenario | Test Cases to Build | Validation Method |
|---|---|---|
| FY 2024-25 new regime tax slabs (0–3L:0%, 3–7L:5%, 7–10L:10%, 10–12L:15%, 12–15L:20%, >15L:30%) | 10 cases at boundary incomes | Deterministic math check |
| Section 80C/80D deduction calculations | 5 cases with mixed deductions | Math check vs known output |
| SIP maturity with step-up and XIRR | 5 cases with varying step-up % | Math check |
| EPF/PPF/NPS contribution advice | 5 cases | LLM-as-judge rubric |
| SEBI-regulated advice refusal | 10 cases (stock tips, fund names) | Binary pass/fail |
| Hindi query handling (Devanagari input) | 5 bilingual queries | Output language consistency check |
| RBI/SEBI source citation in RAG response | 20 RAG queries | Metadata citation presence check |

---

## Quick Reference — Metric Table

| Category | Metric | Unit | Target |
|---|---|---|---|
| **Correctness** | Task Completion Rate | % | > 80% |
| **Correctness** | Tax Calc Accuracy | % | > 95% |
| **Correctness** | SIP/EMI Calc Accuracy | % | > 99% |
| **Correctness** | RAG Faithfulness | 0–1 | > 0.85 |
| **Correctness** | Hallucination Rate | % | < 5% |
| **Agent** | Tool-Call Accuracy | % | > 85% |
| **Agent** | Redundant Call Rate | calls/task | < 0.5 |
| **Agent** | pass@1 | % | > 75% |
| **Agent** | pass@5 variance | % | < 10% |
| **Latency** | TTFT (warm) | ms | < 2000 |
| **Latency** | Generation throughput | tok/s | > 40 |
| **Latency** | Thermal throughput (15 min) | tok/s | > 30 |
| **Memory** | Peak RAM | MB | < 4000 |
| **Memory** | GGUF model size (Q4_K_M) | MB | < 2000 |
| **Energy** | Energy per query | J | < 5 |
| **Safety** | PAS (Policy Adherence Score) | % | > 95% |
| **Safety** | PII Leakage Rate | % | 0% |
| **Safety** | Prompt Injection Resistance | % | > 95% |
| **RAG** | Context Precision | 0–1 | > 0.80 |
| **RAG** | Context Recall | 0–1 | > 0.75 |
| **RAG** | Source Citation Rate | % | > 90% |

---

## Key Watch-outs

1. **Accuracy ≠ Safety on regulated advice.** A correct answer to "should I buy this stock?" is a safety *failure*. Score these separately.
2. **Never evaluate only final output.** Trajectory eval catches 20–40% more real failures (wrong tool, PII mid-conversation, redundant loops).
3. **Quantization sweet spot.** Research shows ~3.5 effective BPW is the threshold. Q4_K_M on a 3B model typically beats Q8_0 on a 1B model — always run the BPW sweep.
4. **Thermal throttling is invisible in single-shot benchmarks.** Always run a 10–15 min sustained load test.
5. **Python GIL contention.** The ReAct orchestration loop can hit a saturation cliff at high concurrency. Profile blocking ratio (β) on the actual inference device.
6. **Validate LLM-as-judge before trusting aggregate scores.** Target ≥ 80% agreement with human grading on a 10-case sample. LLM judges have known biases (length preference, agreeableness).
7. **Hold out 20% of custom benchmark.** Never fine-tune on test cases.
8. **Document simulated-user prompt + model version** in multi-turn harness — otherwise reproducibility is gone.

---

## Sources

| File | Model | Key Contribution |
|---|---|---|
| [eval/research/claude.md](research/claude.md) | Claude | CLEAR framework instantiation, eval pyramid, 2-week roadmap, edge BPW insight |
| [eval/research/chatgpt.md](research/chatgpt.md) | ChatGPT | Full metric catalog, compliance/regulatory framing, synthetic data strategy, phased implementation plan |
| [eval/research/gemini.md](research/gemini.md) | Gemini | Deep edge systems analysis (GIL/saturation cliff, memory bandwidth), Amazon 3-layer model, Giskard/REALM-Bench, semantic caching metrics |
