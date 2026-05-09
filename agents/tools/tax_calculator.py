"""Indian income tax calculator tool — FY 2024-25, new and old regimes."""

from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool

# ── Tax slab data (FY 2024-25) ────────────────────────────────────────────────

_NEW_SLABS = [
    (300_000, 0.00),
    (700_000, 0.05),
    (1_000_000, 0.10),
    (1_200_000, 0.15),
    (1_500_000, 0.20),
    (float("inf"), 0.30),
]

_OLD_SLABS = [
    (250_000, 0.00),
    (500_000, 0.05),
    (1_000_000, 0.20),
    (float("inf"), 0.30),
]

_CESS_RATE = 0.04


def _apply_slabs(taxable: float, slabs: list[tuple]) -> tuple[float, list[dict]]:
    tax = 0.0
    breakdown = []
    prev = 0.0
    for limit, rate in slabs:
        if taxable <= prev:
            break
        slab_income = min(taxable, limit) - prev
        slab_tax = slab_income * rate
        tax += slab_tax
        breakdown.append({"from": prev, "to": min(taxable, limit), "rate_pct": rate * 100, "tax": round(slab_tax)})
        prev = limit
    return tax, breakdown


class TaxCalculator(BaseTool):
    """Indian income tax calculator for FY 2024-25.

    Supports both new and old tax regimes with standard deductions,
    Section 87A rebate, and 4% Health & Education Cess.
    """

    @property
    def name(self) -> str:
        return "tax_calculator"

    @property
    def description(self) -> str:
        return (
            "Calculate Indian income tax for FY 2024-25 under new or old regime. "
            "Returns tax payable, effective rate, and slab-wise breakdown."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "annual_income": {"type": "number", "description": "Gross annual income in INR"},
                "regime": {"type": "string", "enum": ["new", "old"], "description": "Tax regime (new or old)"},
                "is_salaried": {"type": "boolean", "description": "True if salaried (applies standard deduction)", "default": True},
                "deductions_80c": {"type": "number", "description": "Section 80C investments in INR (old regime only, max 1,50,000)", "default": 0},
                "deductions_80d": {"type": "number", "description": "Section 80D health insurance premium (old regime only)", "default": 0},
                "hra_exemption": {"type": "number", "description": "HRA exemption under Section 10(13A) (old regime only)", "default": 0},
                "other_deductions": {"type": "number", "description": "Any other eligible deductions (old regime only)", "default": 0},
            },
            "required": ["annual_income", "regime"],
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Compute income tax for the given parameters.

        Args:
            args: Must include ``annual_income`` and ``regime``; all others optional.

        Returns:
            Dict with keys: tax_payable, effective_rate_pct, taxable_income,
            rebate_87a, cess, tax_before_cess, slab_breakdown, result.
        """
        income = float(args["annual_income"])
        regime = str(args.get("regime", "new")).lower()
        is_salaried = bool(args.get("is_salaried", True))

        if regime == "new":
            std_deduction = 75_000 if is_salaried else 0
            taxable = max(0.0, income - std_deduction)
            tax_raw, breakdown = _apply_slabs(taxable, _NEW_SLABS)
            rebate = min(tax_raw, 25_000) if taxable <= 700_000 else 0.0
        else:
            std_deduction = 50_000 if is_salaried else 0
            cap_80c = min(float(args.get("deductions_80c", 0)), 150_000)
            total_deductions = (
                std_deduction
                + cap_80c
                + float(args.get("deductions_80d", 0))
                + float(args.get("hra_exemption", 0))
                + float(args.get("other_deductions", 0))
            )
            taxable = max(0.0, income - total_deductions)
            tax_raw, breakdown = _apply_slabs(taxable, _OLD_SLABS)
            rebate = min(tax_raw, 12_500) if taxable <= 500_000 else 0.0

        tax_after_rebate = max(0.0, tax_raw - rebate)
        cess = round(tax_after_rebate * _CESS_RATE)
        tax_payable = round(tax_after_rebate + cess)
        effective_rate = round(tax_payable / income * 100, 2) if income > 0 else 0.0

        summary = (
            f"Under the {regime} regime, on an annual income of ₹{income:,.0f}, "
            f"taxable income is ₹{taxable:,.0f}. "
            f"Tax payable (after 87A rebate of ₹{rebate:,.0f} and 4% cess of ₹{cess:,.0f}) "
            f"is ₹{tax_payable:,.0f} ({effective_rate}% effective rate)."
        )

        return {
            "tax_payable": tax_payable,
            "effective_rate_pct": effective_rate,
            "taxable_income": round(taxable),
            "tax_before_cess": round(tax_raw),
            "rebate_87a": round(rebate),
            "cess": cess,
            "regime": regime,
            "slab_breakdown": breakdown,
            "result": summary,
        }
