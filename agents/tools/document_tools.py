"""MCP tools for private uploaded-document analysis."""
from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool
from agents.tools.document_store import analyze_bank_statement, analyze_credit_card_statement, build_redacted_profile, document_search, parse_bill, parse_salary_slip


class DocumentProfileTool(BaseTool):
    @property
    def name(self) -> str: return "document_profile"
    @property
    def description(self) -> str: return "Return a redacted profile of an uploaded document so ReAct can choose the correct extractor."
    @property
    def input_schema(self) -> dict[str, Any]: return {"type":"object","properties":{"document_id":{"type":"string"},"query":{"type":"string"}},"required":["document_id"]}
    def execute(self, args: dict[str, Any]) -> dict[str, Any]: return build_redacted_profile(str(args.get("document_id") or ""), str(args.get("query") or ""))


class DocumentRAGSearchTool(BaseTool):
    @property
    def name(self) -> str: return "document_rag_search"
    @property
    def description(self) -> str: return "Search redacted snippets inside an uploaded document by document_id and query."
    @property
    def input_schema(self) -> dict[str, Any]: return {"type":"object","properties":{"document_id":{"type":"string"},"query":{"type":"string"},"top_k":{"type":"integer","default":6}},"required":["document_id","query"]}
    def execute(self, args: dict[str, Any]) -> dict[str, Any]: return document_search(str(args.get("document_id") or ""), str(args.get("query") or ""), int(args.get("top_k", 6)))


class BankStatementAnalyzerTool(BaseTool):
    @property
    def name(self) -> str: return "bank_statement_analyzer"
    @property
    def description(self) -> str: return "Analyse uploaded bank statements for income, expenses, savings rate, EMI, recurring payments, and risks."
    @property
    def input_schema(self) -> dict[str, Any]: return {"type":"object","properties":{"document_id":{"type":"string"}},"required":["document_id"]}
    def execute(self, args: dict[str, Any]) -> dict[str, Any]: return analyze_bank_statement(str(args.get("document_id") or ""))


class BillParserTool(BaseTool):
    @property
    def name(self) -> str: return "bill_parser"
    @property
    def description(self) -> str: return "Parse uploaded insurance or utility bills for amount due, due date, premium, cover, fees, and flags."
    @property
    def input_schema(self) -> dict[str, Any]: return {"type":"object","properties":{"document_id":{"type":"string"}},"required":["document_id"]}
    def execute(self, args: dict[str, Any]) -> dict[str, Any]: return parse_bill(str(args.get("document_id") or ""))


class CreditCardStatementAnalyzerTool(BaseTool):
    @property
    def name(self) -> str: return "credit_card_statement_analyzer"
    @property
    def description(self) -> str: return "Analyse uploaded credit-card statements for dues, utilization, fees, interest, and repayment risks."
    @property
    def input_schema(self) -> dict[str, Any]: return {"type":"object","properties":{"document_id":{"type":"string"}},"required":["document_id"]}
    def execute(self, args: dict[str, Any]) -> dict[str, Any]: return analyze_credit_card_statement(str(args.get("document_id") or ""))


class SalarySlipParserTool(BaseTool):
    @property
    def name(self) -> str: return "salary_slip_parser"
    @property
    def description(self) -> str: return "Parse uploaded salary slips for gross pay, net pay, HRA, PF, TDS, and annualized income."
    @property
    def input_schema(self) -> dict[str, Any]: return {"type":"object","properties":{"document_id":{"type":"string"}},"required":["document_id"]}
    def execute(self, args: dict[str, Any]) -> dict[str, Any]: return parse_salary_slip(str(args.get("document_id") or ""))