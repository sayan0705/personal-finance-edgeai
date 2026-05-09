"""SIP and mutual fund investment calculator tool."""

from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool


class SIPCalculator(BaseTool):
    """Calculates SIP maturity, lumpsum future value, and step-up SIP returns.

    Uses standard Indian SIP formula: FV = P × [(1+r)^n - 1] / r × (1+r)
    where r = monthly rate and n = total months.
    """

    @property
    def name(self) -> str:
        return "sip_calculator"

    @property
    def description(self) -> str:
        return (
            "Calculate SIP maturity amount, lumpsum future value, or step-up SIP returns. "
            "Specify calculation_type as 'sip', 'lumpsum', 'step_up_sip', or 'reverse_sip'."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "calculation_type": {
                    "type": "string",
                    "enum": ["sip", "lumpsum", "step_up_sip", "reverse_sip"],
                    "description": "Type of calculation",
                },
                "monthly_amount": {"type": "number", "description": "Monthly SIP instalment in INR (for sip / step_up_sip)"},
                "amount": {"type": "number", "description": "Lumpsum investment amount in INR (for lumpsum)"},
                "target_amount": {"type": "number", "description": "Desired corpus in INR (for reverse_sip)"},
                "annual_rate_pct": {"type": "number", "description": "Expected annual return rate as percentage (e.g. 12 for 12%)"},
                "years": {"type": "integer", "description": "Investment duration in years"},
                "step_up_rate_pct": {"type": "number", "description": "Annual step-up percentage for step_up_sip (e.g. 10 for 10%)", "default": 10},
            },
            "required": ["calculation_type", "annual_rate_pct", "years"],
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run the requested SIP calculation.

        Args:
            args: Parameters matching input_schema.

        Returns:
            Dict with calculation results and a human-readable ``result`` string.
        """
        calc_type = str(args["calculation_type"])
        rate = float(args["annual_rate_pct"]) / 100
        years = int(args["years"])

        if calc_type == "sip":
            monthly = float(args["monthly_amount"])
            maturity = self._sip_maturity(monthly, rate, years)
            total_invested = monthly * years * 12
            returns = maturity - total_invested
            return {
                "maturity_amount": maturity,
                "total_invested": round(total_invested),
                "total_returns": round(returns),
                "result": (
                    f"SIP of ₹{monthly:,.0f}/month for {years} years at {rate*100:.1f}% p.a. "
                    f"grows to ₹{maturity:,.0f}. "
                    f"Total invested: ₹{total_invested:,.0f}, returns: ₹{returns:,.0f}."
                ),
            }

        if calc_type == "lumpsum":
            amount = float(args["amount"])
            fv = round(amount * (1 + rate) ** years)
            gains = fv - amount
            return {
                "future_value": fv,
                "total_gains": gains,
                "result": (
                    f"Lumpsum of ₹{amount:,.0f} at {rate*100:.1f}% p.a. for {years} years "
                    f"grows to ₹{fv:,.0f} (gains: ₹{gains:,.0f})."
                ),
            }

        if calc_type == "step_up_sip":
            initial = float(args["monthly_amount"])
            step_up = float(args.get("step_up_rate_pct", 10)) / 100
            maturity = self._step_up_sip(initial, rate, years, step_up)
            return {
                "maturity_amount": maturity,
                "result": (
                    f"Step-up SIP starting at ₹{initial:,.0f}/month, increasing {step_up*100:.0f}% "
                    f"yearly, at {rate*100:.1f}% p.a. for {years} years grows to ₹{maturity:,.0f}."
                ),
            }

        if calc_type == "reverse_sip":
            target = float(args["target_amount"])
            monthly = self._reverse_sip(target, rate, years)
            return {
                "required_monthly_sip": monthly,
                "result": (
                    f"To accumulate ₹{target:,.0f} in {years} years at {rate*100:.1f}% p.a., "
                    f"you need a monthly SIP of ₹{monthly:,.0f}."
                ),
            }

        return {"result": f"Unknown calculation_type: {calc_type}"}

    @staticmethod
    def _sip_maturity(monthly: float, annual_rate: float, years: int) -> int:
        r = annual_rate / 12
        n = years * 12
        if r == 0:
            return round(monthly * n)
        return round(monthly * ((1 + r) ** n - 1) / r * (1 + r))

    @staticmethod
    def _step_up_sip(initial: float, annual_rate: float, years: int, step_up: float) -> int:
        r = annual_rate / 12
        total = 0.0
        for yr in range(years):
            monthly = initial * ((1 + step_up) ** yr)
            months_left = (years - yr) * 12
            if r == 0:
                total += monthly * months_left
            else:
                total += monthly * ((1 + r) ** months_left - 1) / r * (1 + r)
        return round(total)

    @staticmethod
    def _reverse_sip(target: float, annual_rate: float, years: int) -> int:
        r = annual_rate / 12
        n = years * 12
        if r == 0:
            return round(target / n)
        return round(target * r / ((1 + r) ** n - 1) / (1 + r))
