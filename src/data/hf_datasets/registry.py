"""Dataset registry — 13 open-source financial HuggingFace datasets."""

from __future__ import annotations

from typing import Any

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "fingpt_sentiment": {
        "hf_id": "FinGPT/fingpt-sentiment-train",
        "task": "sentiment",
        "split": "train",
        "config": None,
        "description": "76K financial sentiment (instruction/input/output format)",
    },
    "financial_phrasebank": {
        # Uses a legacy loading script that HF no longer supports; kept for reference.
        # Will fail gracefully — pipeline continues without it.
        "hf_id": "takala/financial_phrasebank",
        "task": "sentiment",
        "split": "train",
        "config": "sentences_allagree",
        "description": "5K financial sentence sentiment classification",
    },
    "finance_alpaca": {
        "hf_id": "gbharti/finance-alpaca",
        "task": "instruction",
        "split": "train",
        "config": None,
        "description": "68K finance instruction-following pairs",
    },
    "sujet_finance": {
        "hf_id": "sujet-ai/Sujet-Finance-Instruct-177k",
        "task": "multi_task",
        "split": "train",
        "config": None,
        "description": "177K multi-task finance (sentiment, QA, NER, summarisation)",
    },
    "twitter_fin_sentiment": {
        "hf_id": "zeroshot/twitter-financial-news-sentiment",
        "task": "sentiment",
        "split": "train",
        "config": None,
        "description": "12K financial tweet sentiment (Bearish/Bullish/Neutral)",
    },
    "twitter_fin_topics": {
        "hf_id": "zeroshot/twitter-financial-news-topic",
        "task": "classification",
        "split": "train",
        "config": None,
        "description": "21K financial tweet topic classification",
    },
    "fintalk_19k": {
        "hf_id": "ceadar-ie/FinTalk-19k",
        "task": "conversation",
        "split": "train",
        "config": None,
        "description": "19K financial instruction conversations",
    },
    "finance_instruct_500k": {
        "hf_id": "Josephgflowers/Finance-Instruct-500k",
        "task": "instruction",
        "split": "train",
        "config": None,
        "description": "500K mega finance instruction set",
    },
    "indian_itr": {
        "hf_id": "AgamiAI/Indian-Income-Tax-Returns",
        "task": "indian_tax",
        "split": "train",
        "config": None,
        "description": "Synthetic Indian ITR forms — India-specific",
    },
    "adaptllm_fpb": {
        # Removed from HF Hub — fails gracefully.
        "hf_id": "AdaptLLM/finance-tasks_FPB",
        "task": "sentiment",
        "split": "test",
        "config": None,
        "description": "Financial PhraseBank adapted for LLM evaluation",
    },
    "adaptllm_fiqa": {
        # Removed from HF Hub — fails gracefully.
        "hf_id": "AdaptLLM/finance-tasks_FiQA_SA",
        "task": "sentiment",
        "split": "test",
        "config": None,
        "description": "FiQA Sentiment adapted for LLM evaluation",
    },
    "adaptllm_convfinqa": {
        # Removed from HF Hub — fails gracefully.
        "hf_id": "AdaptLLM/finance-tasks_ConvFinQA",
        "task": "qa_context",
        "split": "test",
        "config": None,
        "description": "Conversational Financial QA with context",
    },
    "finentity": {
        "hf_id": "yixuantt/FinEntity",
        "task": "ner",
        "split": "train",
        "config": None,
        "description": "Financial named entity recognition",
    },
    "finqa": {
        # Downloaded via run_finqa_extraction.py directly from github.com/czyssrs/FinQA
        # ibm/finqa on HF Hub requires a deprecated loading script — use the GitHub source instead.
        "hf_id": "finqasite/finqa",
        "task": "financial_qa",
        "split": "train",
        "config": None,
        "description": "FinQA — 6.2K financial report QA pairs requiring numerical reasoning (SEC filings)",
    },
}

# Layer-assignment keyword lists
LAYER_KEYWORDS: dict[str, list[str]] = {
    "L2_indian_regulatory": [
        "sebi", "rbi", "income tax", "itr", "gst", "section 80", "hra",
        "nps", "ppf", "epf", "hra exemption", "indian", "india", "rupee",
        "nifty", "sensex", "bse", "nse", "mutual fund india", "sip",
        "lakh", "crore", "assessment year", "financial year", "pan card",
        "aadhaar", "pfrda", "irdai", "nbfc", "agamai",
    ],
    "L3_personal_finance": [
        "budget", "savings", "emergency fund", "retirement", "mortgage",
        "home loan", "car loan", "insurance", "term life", "health insurance",
        "credit card", "debt", "emi", "sip", "etf", "index fund",
        "portfolio", "asset allocation", "rebalancing", "tax planning",
        "401k", "ira", "pension", "social security", "estate planning",
        "will", "trust", "loan", "refinance", "down payment",
        "credit score", "fico", "expense", "income", "salary",
        "financial planning", "financial advisor", "robo-advisor",
        "personal finance", "money management", "cash flow",
    ],
    "L4_community_conversational": [
        "should i", "recommend", "what would you suggest", "help me",
        "is it a good idea", "opinion", "advice", "thoughts on",
        "which is better", "how should i", "can you help",
        "beginner", "newbie", "first time", "just started",
    ],
}

TWITTER_TOPIC_MAP: dict[int, str] = {
    0: "Analyst Update", 1: "Fed | Central Banks", 2: "Company | Product News",
    3: "Treasuries | Corporate Debt", 4: "Dividend", 5: "Earnings",
    6: "Energy | Oil", 7: "Financials", 8: "Currencies",
    9: "General News | Opinion", 10: "Gold | Metals | Materials",
    11: "IPO", 12: "Legal | Regulation", 13: "M&A | Investments",
    14: "Macro", 15: "Markets", 16: "Politics",
    17: "Personnel Change", 18: "Stock Commentary", 19: "Stock Movement",
}

UNSAFE_PATTERNS: list[str] = [
    r"guaranteed\s+returns?",
    r"risk[- ]?free\s+investment",
    r"insider\s+tip",
    r"ponzi",
    r"pyramid\s+scheme",
    r"get\s+rich\s+quick",
]
