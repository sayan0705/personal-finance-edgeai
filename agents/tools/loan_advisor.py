"""Loan EMI calculator and amortization tool."""

from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool


class LoanAdvisor(BaseTool):
    """Computes EMI, total interest, and amortization schedule for loans.

    Uses the standard EMI formula:
        EMI = P × r × (1+r)^n / ((1+r)^n - 1)
    where P = principal, r = monthly rate, n = tenure in months.
    """

    @property
    def name(self) -> str:
        return "loan_advisor"

    @property
    def description(self) -> str:
        return (
            "Calculate loan EMI, total interest payable, and amortization schedule. "
            "Supports home loans, car loans, personal loans."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Loan amount in INR"},
                "annual_rate_pct": {"type": "number", "description": "Annual interest rate as percentage (e.g. 8.5 for 8.5%)"},
                "tenure_years": {"type": "number", "description": "Loan tenure in years"},
                "include_schedule": {"type": "boolean", "description": "Whether to include year-wise amortization schedule", "default": False},
            },
            "required": ["principal", "annual_rate_pct", "tenure_years"],
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Calculate EMI and loan repayment details.

        Args:
            args: Must include principal, annual_rate_pct, tenure_years.

        Returns:
            Dict with emi, total_payment, total_interest, and optionally amortization_schedule.
        """
        principal = float(args["principal"])
        annual_rate = float(args["annual_rate_pct"]) / 100
        tenure_months = round(float(args["tenure_years"]) * 12)
        include_schedule = bool(args.get("include_schedule", False))

        emi = self._compute_emi(principal, annual_rate, tenure_months)
        total_payment = round(emi * tenure_months)
        total_interest = total_payment - round(principal)

        result = {
            "emi": emi,
            "total_payment": total_payment,
            "total_interest": total_interest,
            "tenure_months": tenure_months,
            "result": (
                f"For a loan of ₹{principal:,.0f} at {annual_rate*100:.2f}% p.a. over "
                f"{args['tenure_years']} years: EMI = ₹{emi:,.0f}/month, "
                f"total interest = ₹{total_interest:,.0f}, "
                f"total payment = ₹{total_payment:,.0f}."
            ),
        }

        if include_schedule:
            result["amortization_schedule"] = self._yearly_schedule(principal, annual_rate, tenure_months, emi)

        return result

    @staticmethod
    def _compute_emi(principal: float, annual_rate: float, tenure_months: int) -> int:
        r = annual_rate / 12
        n = tenure_months
        if r == 0:
            return round(principal / n)
        return round(principal * r * (1 + r) ** n / ((1 + r) ** n - 1))

    @staticmethod
    def _yearly_schedule(
        principal: float, annual_rate: float, tenure_months: int, emi: int
    ) -> list[dict]:
        """Build a year-wise amortization schedule."""
        r = annual_rate / 12
        balance = principal
        schedule = []
        year = 1
        year_principal = 0.0
        year_interest = 0.0

        for month in range(1, tenure_months + 1):
            interest = balance * r
            principal_paid = emi - interest
            balance = max(0.0, balance - principal_paid)
            year_principal += principal_paid
            year_interest += interest

            if month % 12 == 0 or month == tenure_months:
                schedule.append({
                    "year": year,
                    "principal_paid": round(year_principal),
                    "interest_paid": round(year_interest),
                    "closing_balance": round(balance),
                })
                year += 1
                year_principal = 0.0
                year_interest = 0.0

        return schedule
