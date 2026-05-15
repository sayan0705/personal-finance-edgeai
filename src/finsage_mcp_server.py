"""BanyanTree MCP-style tools server for Docker.

This is the faithful container split of the original generated MCP server:

- API service runs on port 8000.
- MCP tools service runs on port 8010.
- The original BanyanTree RAG calls this service through MCPToolClient.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from agents.tools.banyantree_tools import (
    AMFINavTool,
    EMIQueryCalculator,
    GoalPlannerTool,
    PortfolioHealthTool,
    PortfolioMultiAgentTool,
    SIPQueryCalculator,
    ScreenerTool,
    SearchRAGTool,
)
from agents.tools.document_tools import (
    BankStatementAnalyzerTool,
    BillParserTool,
    CreditCardStatementAnalyzerTool,
    DocumentProfileTool,
    DocumentRAGSearchTool,
    SalarySlipParserTool,
)
from agents.tools.registry import ToolRegistry


if hasattr(asyncio, "DefaultEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


app = FastAPI(title="FinSage MCP Tool Server", version="8.0.0-docker")


class AMFIRequest(BaseModel):
    fund_filter: Optional[str] = ""


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 4


class QueryRequest(BaseModel):
    query: str


class SymbolRequest(BaseModel):
    symbol: str


class PortfolioWorkflowRequest(BaseModel):
    query: str
    max_stocks: int = 4
    symbols: Optional[list[str]] = None


class DocumentRequest(BaseModel):
    document_id: str
    query: Optional[str] = ""
    top_k: int = 6


class ToolResponse(BaseModel):
    tool: str
    success: bool
    data: dict
    error: Optional[str] = None


TOOL_NAMES = [
    "amfi_nav",
    "search_rag",
    "screener",
    "sip_calculator",
    "emi_calculator",
    "portfolio_health",
    "goal_planner",
    "portfolio_multi_agent",
    "document_profile",
    "document_rag_search",
    "bank_statement_analyzer",
    "bill_parser",
    "credit_card_statement_analyzer",
    "salary_slip_parser",
]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        AMFINavTool(),
        SearchRAGTool(),
        ScreenerTool(),
        SIPQueryCalculator(),
        EMIQueryCalculator(),
        PortfolioHealthTool(),
        GoalPlannerTool(),
        PortfolioMultiAgentTool(),
        DocumentProfileTool(),
        DocumentRAGSearchTool(),
        BankStatementAnalyzerTool(),
        BillParserTool(),
        CreditCardStatementAnalyzerTool(),
        SalarySlipParserTool(),
    ):
        registry.register(tool)
    return registry


REGISTRY = _registry()


def _execute(tool: str, args: dict) -> ToolResponse:
    raw = REGISTRY.execute(tool, args)
    try:
        import json

        payload = json.loads(raw)
    except Exception:
        payload = {"result": raw}
    summary = payload.get("result") or payload.get("summary") or raw
    return ToolResponse(tool=tool, success=True, data={**payload, "summary": summary})


@app.post("/tools/amfi_nav", response_model=ToolResponse)
async def amfi_nav(req: AMFIRequest):
    return _execute("amfi_nav", req.model_dump())


@app.post("/tools/search_rag", response_model=ToolResponse)
async def search_rag(req: RAGSearchRequest):
    return _execute("search_rag", req.model_dump())


@app.post("/tools/screener", response_model=ToolResponse)
async def screener(req: SymbolRequest):
    return _execute("screener", req.model_dump())


@app.post("/tools/sip_calculator", response_model=ToolResponse)
async def sip_calculator(req: QueryRequest):
    return _execute("sip_calculator", req.model_dump())


@app.post("/tools/emi_calculator", response_model=ToolResponse)
async def emi_calculator(req: QueryRequest):
    return _execute("emi_calculator", req.model_dump())


@app.post("/tools/portfolio_health", response_model=ToolResponse)
async def portfolio_health(req: QueryRequest):
    return _execute("portfolio_health", req.model_dump())


@app.post("/tools/goal_planner", response_model=ToolResponse)
async def goal_planner(req: QueryRequest):
    return _execute("goal_planner", req.model_dump())


@app.post("/tools/portfolio_multi_agent", response_model=ToolResponse)
async def portfolio_multi_agent(req: PortfolioWorkflowRequest):
    return _execute("portfolio_multi_agent", req.model_dump())



@app.post("/tools/document_profile", response_model=ToolResponse)
async def document_profile(req: DocumentRequest):
    return _execute("document_profile", req.model_dump())


@app.post("/tools/document_rag_search", response_model=ToolResponse)
async def document_rag_search(req: DocumentRequest):
    return _execute("document_rag_search", req.model_dump())


@app.post("/tools/bank_statement_analyzer", response_model=ToolResponse)
async def bank_statement_analyzer(req: DocumentRequest):
    return _execute("bank_statement_analyzer", req.model_dump())


@app.post("/tools/bill_parser", response_model=ToolResponse)
async def bill_parser(req: DocumentRequest):
    return _execute("bill_parser", req.model_dump())


@app.post("/tools/credit_card_statement_analyzer", response_model=ToolResponse)
async def credit_card_statement_analyzer(req: DocumentRequest):
    return _execute("credit_card_statement_analyzer", req.model_dump())


@app.post("/tools/salary_slip_parser", response_model=ToolResponse)
async def salary_slip_parser(req: DocumentRequest):
    return _execute("salary_slip_parser", req.model_dump())

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_loaded": False,
        "tools": TOOL_NAMES,
        "server": "canonical_mcp_tools_server",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/tools")
async def list_tools():
    return {
        "server": "FinSage MCP Tool Server",
        "architecture": "Dedicated Docker MCP tools server for BanyanTree API.",
        "tools": [
            {"name": name, "endpoint": f"/tools/{name}", "method": "POST"}
            for name in TOOL_NAMES
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010, reload=False, workers=1)
