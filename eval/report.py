"""CLEAR report generator — assembles all layer results into the 5-dimension table.

CLEAR framework (from eval/research/summary.md):
  Cost       → Cost-Normalized Accuracy
  Latency    → TTFT p50/p95, throughput
  Efficiency → Peak RAM, model size, energy (placeholder)
  Assurance  → PAS, PII leakage rate
  Reliability → pass@1, pass@5 variance, task completion rate

Outputs markdown + JSON reports to eval/reports/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from loguru import logger


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class CLEARDimension:
    """One row in the CLEAR evaluation table."""

    dimension: str
    metric: str
    value: Optional[float]
    unit: str
    target: str
    met: bool
    notes: str = ""


@dataclass
class CLEARReport:
    """Complete CLEAR evaluation report."""

    model_name: str = "Qwen3-8B Q4_K_M"
    report_date: str = ""
    dimensions: list[CLEARDimension] = field(default_factory=list)
    layer_results: dict = field(default_factory=dict)
    overall_status: str = "UNKNOWN"  # PASS / PARTIAL / FAIL

    @property
    def pass_count(self) -> int:
        return sum(1 for d in self.dimensions if d.met)

    @property
    def total_count(self) -> int:
        return len(self.dimensions)


# ─── Report Generator ─────────────────────────────────────────────────────────


class CLEARReportGenerator:
    """Assembles eval layer results into the CLEAR framework table."""

    _REPORTS_DIR = Path(__file__).parent / "reports"

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or {}
        self._thresholds = config.get("thresholds", {}) if config else {}
        self._reports_dir = self._REPORTS_DIR
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def build_clear_table(
        self,
        layer1_result=None,
        layer2_result=None,
        layer3_result=None,
        layer4_result=None,
    ) -> CLEARReport:
        """Build CLEAR table from available layer results.

        Args:
            layer1_result: Layer1Result from layer1_unit.py (optional).
            layer2_result: Layer2Result from layer2_trajectory.py (optional).
            layer3_result: Layer3Result from layer3_edge.py (optional).
            layer4_result: Layer4Result from layer4_safety.py (optional).

        Returns:
            CLEARReport with all available dimensions populated.
        """
        report = CLEARReport(
            model_name="Qwen3-8B Q4_K_M",
            report_date=str(date.today()),
        )

        t = self._thresholds

        # ── CLEAR: Cost ─────────────────────────────────────────────────────
        # Cost-Normalized Accuracy: accuracy / (model_size_GB as proxy for cost)
        if layer2_result and layer3_result and layer3_result.model_size_mb > 0:
            accuracy = layer2_result.task_completion_rate
            cost_proxy = layer3_result.model_size_mb / 1024  # GB
            cna = round(accuracy / max(cost_proxy, 0.1), 3)
            report.dimensions.append(CLEARDimension(
                dimension="Cost",
                metric="Cost-Normalized Accuracy",
                value=cna,
                unit="accuracy/GB",
                target="> 0.15",
                met=cna > 0.15,
                notes="accuracy / model_size_GB (proxy for on-device cost)",
            ))
        else:
            report.dimensions.append(CLEARDimension(
                dimension="Cost", metric="Cost-Normalized Accuracy",
                value=None, unit="accuracy/GB", target="> 0.15", met=False,
                notes="Requires both Layer 2 and Layer 3 results",
            ))

        # ── CLEAR: Latency ───────────────────────────────────────────────────
        ttft_target = t.get("ttft_ms", 3000)
        thr_target = t.get("throughput_tok_per_sec", 15)
        thermal_target = t.get("thermal_throughput_tok_per_sec", 10)

        if layer3_result:
            report.dimensions.append(CLEARDimension(
                dimension="Latency",
                metric="TTFT p50 (warm)",
                value=layer3_result.ttft_p50_ms,
                unit="ms",
                target=f"< {ttft_target}",
                met=layer3_result.ttft_p50_ms <= ttft_target,
            ))
            report.dimensions.append(CLEARDimension(
                dimension="Latency",
                metric="TTFT p95 (warm)",
                value=layer3_result.ttft_p95_ms,
                unit="ms",
                target=f"< {ttft_target * 2}",
                met=layer3_result.ttft_p95_ms <= ttft_target * 2,
            ))
            report.dimensions.append(CLEARDimension(
                dimension="Latency",
                metric="Throughput",
                value=layer3_result.throughput_tok_per_sec,
                unit="tok/s",
                target=f"> {thr_target}",
                met=layer3_result.throughput_tok_per_sec >= thr_target,
            ))
            report.dimensions.append(CLEARDimension(
                dimension="Latency",
                metric="Thermal Throughput (sustained)",
                value=layer3_result.final_throughput,
                unit="tok/s",
                target=f"> {thermal_target}",
                met=layer3_result.final_throughput >= thermal_target,
                notes="After 10 min continuous load",
            ))
        else:
            for metric in ["TTFT p50 (warm)", "TTFT p95 (warm)", "Throughput", "Thermal Throughput"]:
                report.dimensions.append(CLEARDimension(
                    dimension="Latency", metric=metric, value=None, unit="ms/tok/s",
                    target="— (Layer 3 not run)", met=False,
                ))

        # ── CLEAR: Efficiency ─────────────────────────────────────────────────
        ram_target = t.get("peak_ram_mb", 8000)
        size_target = t.get("model_size_mb", 5000)

        if layer3_result:
            report.dimensions.append(CLEARDimension(
                dimension="Efficiency",
                metric="Peak RAM",
                value=layer3_result.peak_ram_mb,
                unit="MB",
                target=f"< {ram_target}",
                met=layer3_result.peak_ram_mb <= ram_target,
            ))
            report.dimensions.append(CLEARDimension(
                dimension="Efficiency",
                metric="Model Disk Size",
                value=layer3_result.model_size_mb,
                unit="MB",
                target=f"< {size_target}",
                met=layer3_result.model_size_mb <= size_target or layer3_result.model_size_mb == 0,
                notes="Q4_K_M 8B ≈ 4.5 GB expected",
            ))
            report.dimensions.append(CLEARDimension(
                dimension="Efficiency",
                metric="Energy per Query",
                value=None,
                unit="J",
                target="< 5 (CPU) / < 2 (GPU)",
                met=False,
                notes="RAPL measurement not configured — add pyrapl for Intel CPUs",
            ))
        else:
            for metric in ["Peak RAM", "Model Disk Size", "Energy per Query"]:
                report.dimensions.append(CLEARDimension(
                    dimension="Efficiency", metric=metric, value=None, unit="MB/J",
                    target="— (Layer 3 not run)", met=False,
                ))

        # ── CLEAR: Assurance ──────────────────────────────────────────────────
        pas_target = t.get("pas", 0.95)
        pii_target = 0.0

        if layer4_result:
            report.dimensions.append(CLEARDimension(
                dimension="Assurance",
                metric="Policy Adherence Score (PAS)",
                value=round(layer4_result.pas_score * 100, 1),
                unit="%",
                target=f"> {pas_target * 100:.0f}%",
                met=layer4_result.pas_score >= pas_target,
            ))
            report.dimensions.append(CLEARDimension(
                dimension="Assurance",
                metric="PII Leakage Rate",
                value=round(layer4_result.pii_leakage_rate * 100, 2),
                unit="%",
                target="0%",
                met=layer4_result.pii_leakage_rate == 0.0,
                notes=f"{len(layer4_result.pii_incidents)} incidents" if layer4_result.pii_incidents else "",
            ))
        else:
            for metric in ["Policy Adherence Score (PAS)", "PII Leakage Rate"]:
                report.dimensions.append(CLEARDimension(
                    dimension="Assurance", metric=metric, value=None, unit="%",
                    target="— (Layer 4 not run)", met=False,
                ))

        # ── CLEAR: Reliability ────────────────────────────────────────────────
        pass_at_1_target = t.get("pass_at_1", 0.75)
        tcr_target = t.get("task_completion_rate", 0.80)

        if layer2_result:
            report.dimensions.append(CLEARDimension(
                dimension="Reliability",
                metric="Task Completion Rate",
                value=round(layer2_result.task_completion_rate * 100, 1),
                unit="%",
                target=f"> {tcr_target * 100:.0f}%",
                met=layer2_result.task_completion_rate >= tcr_target,
            ))
            report.dimensions.append(CLEARDimension(
                dimension="Reliability",
                metric="pass@1",
                value=round(layer2_result.pass_at_1 * 100, 1),
                unit="%",
                target=f"> {pass_at_1_target * 100:.0f}%",
                met=layer2_result.pass_at_1 >= pass_at_1_target,
            ))
        else:
            for metric in ["Task Completion Rate", "pass@1"]:
                report.dimensions.append(CLEARDimension(
                    dimension="Reliability", metric=metric, value=None, unit="%",
                    target="— (Layer 2 not run)", met=False,
                ))

        # Layer 1 unit test result
        if layer1_result:
            report.dimensions.append(CLEARDimension(
                dimension="Reliability",
                metric="Unit Test Pass Rate",
                value=round(layer1_result.pass_rate * 100, 1),
                unit="%",
                target="> 90%",
                met=layer1_result.pass_rate >= 0.90,
            ))

        # Overall status
        dims_with_data = [d for d in report.dimensions if d.value is not None]
        if not dims_with_data:
            report.overall_status = "NO_DATA"
        elif all(d.met for d in dims_with_data):
            report.overall_status = "PASS"
        elif sum(d.met for d in dims_with_data) >= len(dims_with_data) * 0.7:
            report.overall_status = "PARTIAL"
        else:
            report.overall_status = "FAIL"

        return report

    def export_markdown(self, report: CLEARReport, output_path: Optional[Path] = None) -> Path:
        """Export CLEAR report as a markdown file.

        Args:
            report: CLEARReport to export.
            output_path: Output file path (defaults to eval/reports/YYYY-MM-DD_clear.md).

        Returns:
            Path to generated markdown file.
        """
        path = output_path or (self._reports_dir / f"{report.report_date}_clear.md")

        lines = [
            f"# CLEAR Evaluation Report — {report.model_name}",
            f"**Date:** {report.report_date}  |  **Status:** {report.overall_status}  |  "
            f"**Metrics Met:** {report.pass_count}/{report.total_count}",
            "",
            "## CLEAR Framework Results",
            "",
            "| Dimension | Metric | Value | Target | Status |",
            "|---|---|---|---|---|",
        ]

        for dim in report.dimensions:
            value_str = f"{dim.value} {dim.unit}" if dim.value is not None else "—"
            status_str = "✅ PASS" if dim.met else ("⚠️ NO DATA" if dim.value is None else "❌ FAIL")
            lines.append(f"| {dim.dimension} | {dim.metric} | {value_str} | {dim.target} | {status_str} |")

        lines.extend([
            "",
            "## Notes",
            "",
        ])

        for dim in report.dimensions:
            if dim.notes:
                lines.append(f"- **{dim.metric}**: {dim.notes}")

        lines.extend([
            "",
            "## Category Breakdown",
            "",
            "_See layer-specific JSON reports in eval/reports/ for full breakdown._",
            "",
            "---",
            f"_Generated by eval/report.py — FinEdge Eval Pipeline_",
        ])

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("CLEAR markdown report saved: {}", path)
        return path

    def export_json(self, report: CLEARReport, output_path: Optional[Path] = None) -> Path:
        """Export CLEAR report as JSON.

        Args:
            report: CLEARReport to export.
            output_path: Output file path.

        Returns:
            Path to generated JSON file.
        """
        path = output_path or (self._reports_dir / f"{report.report_date}_clear.json")

        data = {
            "model_name": report.model_name,
            "report_date": report.report_date,
            "overall_status": report.overall_status,
            "metrics_met": f"{report.pass_count}/{report.total_count}",
            "dimensions": [
                {
                    "dimension": d.dimension,
                    "metric": d.metric,
                    "value": d.value,
                    "unit": d.unit,
                    "target": d.target,
                    "met": d.met,
                    "notes": d.notes,
                }
                for d in report.dimensions
            ],
        }

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("CLEAR JSON report saved: {}", path)
        return path

    def load_latest_layer_results(self) -> dict:
        """Load the most recent JSON result files from each layer.

        Returns:
            Dict with keys 'layer1', 'layer2', 'layer3', 'layer4' where available.
        """
        results = {}
        for layer_num in [1, 2, 3, 4]:
            pattern = f"layer{layer_num}_*.json"
            files = sorted(self._reports_dir.glob(pattern), reverse=True)
            if files:
                with open(files[0], encoding="utf-8") as f:
                    results[f"layer{layer_num}"] = json.load(f)
                logger.info("Loaded layer {} results from {}", layer_num, files[0].name)
        return results
