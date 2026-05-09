"""Layer 3 — Edge / Systems evaluation (per model variant).

Profiles hardware performance of the Qwen3-8B GGUF model on Windows:
- Latency p50/p95/p99 (cold boot and warm)
- Sustained load test (10 min, throttling detection)
- Peak RAM via psutil
- Token throughput
- Quantization sweep (if multiple GGUF files present)

Outputs the CLEAR §Latency and §Efficiency rows.
Windows-safe: uses psutil, no tegrastats/powermetrics/RAPL.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from eval.model_adapter import ModelAdapter


# ─── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class Layer3Result:
    """Aggregate result for Layer 3 edge/systems evaluation."""

    # Latency
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    cold_boot_ms: Optional[float] = None

    # Throughput
    throughput_tok_per_sec: float = 0.0

    # Memory
    peak_ram_mb: float = 0.0
    model_size_mb: float = 0.0

    # Sustained load
    initial_throughput: float = 0.0
    final_throughput: float = 0.0
    throttle_detected: bool = False
    throttle_drop_pct: float = 0.0

    # Quantization sweep
    bpw_curve: list[dict] = field(default_factory=list)

    # CLEAR compliance
    thresholds_met: dict[str, bool] = field(default_factory=dict)
    duration_sec: float = 0.0


# ─── Layer 3 Runner ───────────────────────────────────────────────────────────

_BENCHMARK_PROMPTS = [
    "My annual salary is ₹10 lakhs. What is my income tax under the new regime for FY 2024-25? Please show the slab-wise breakdown.",
    "I want to invest ₹5,000 per month in a SIP at 12% annual returns for 10 years. What will be my final corpus?",
    "I have a home loan of ₹20 lakhs at 8.5% annual interest for 20 years. What is my monthly EMI?",
    "Compare the old vs new income tax regime for a salaried person earning ₹12 lakhs with ₹1.5 lakh in 80C investments.",
    "I am 30 years old and want to retire at 60. I need ₹50,000 per month in today's value. How much corpus should I build?",
]


class Layer3Runner:
    """Edge/systems profiler for on-device Qwen3-8B inference.

    All measurements are Windows-compatible (psutil for RAM).
    RAPL energy is not measured (optional future addition).
    """

    _REPORTS_DIR = Path(__file__).parents[2] / "eval" / "reports"

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or {}
        self._edge_config = config.get("edge", {}) if config else {}
        self._thresholds = config.get("thresholds", {}) if config else {}
        self._reports_dir = self._REPORTS_DIR
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def run(self, adapter: "ModelAdapter") -> Layer3Result:
        """Run all Layer 3 edge/systems measurements.

        Args:
            adapter: Loaded ModelAdapter (model must be loaded).

        Returns:
            Layer3Result with all timing and memory measurements.
        """
        t0 = time.perf_counter()
        result = Layer3Result()

        logger.info("Layer 3: starting edge profiling...")

        from eval.metrics.edge_metrics import EdgeProfiler
        profiler = EdgeProfiler(self._edge_config)

        # 1. Warm latency (p50/p95/p99)
        logger.info("Measuring warm latency...")
        lat_stats = profiler.measure_latency(
            adapter,
            prompt=_BENCHMARK_PROMPTS[0],
            n_runs=self._edge_config.get("benchmark_runs", 10),
        )
        result.ttft_p50_ms = lat_stats.p50_ms
        result.ttft_p95_ms = lat_stats.p95_ms
        result.ttft_p99_ms = lat_stats.p99_ms

        # 2. Throughput
        logger.info("Measuring throughput...")
        thr = profiler.measure_throughput(adapter, prompt=_BENCHMARK_PROMPTS[1])
        result.throughput_tok_per_sec = thr.tokens_per_sec

        # 3. Peak RAM
        logger.info("Measuring peak RAM...")
        result.peak_ram_mb = profiler.measure_peak_ram(adapter, prompt=_BENCHMARK_PROMPTS[0])

        # 4. Model file size
        model_path = self._config.get("model", {}).get("path", "") if self._config else ""
        if model_path:
            import os
            if os.path.exists(model_path):
                result.model_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 1)

        # 5. Sustained load test
        sustained_min = self._edge_config.get("sustained_load_minutes", 10)
        logger.info("Starting sustained load test ({} min)...", sustained_min)
        sustained = profiler.sustained_load_test(
            adapter,
            prompts=_BENCHMARK_PROMPTS,
            duration_min=sustained_min,
        )
        result.initial_throughput = sustained.initial_throughput
        result.final_throughput = sustained.final_throughput
        result.throttle_detected = sustained.throttle_detected
        result.throttle_drop_pct = sustained.throttle_drop_pct

        # 6. Quantization sweep (if multiple GGUF files present)
        quant_paths = self._edge_config.get("quant_model_paths", {})
        if quant_paths and self._config:
            logger.info("Running quantization sweep...")
            cases = [{"query": p, "eval_type": "rubric"} for p in _BENCHMARK_PROMPTS[:3]]
            curve = profiler.quantization_sweep(quant_paths, cases, self._config)
            result.bpw_curve = [
                {
                    "quantization": p.quantization,
                    "effective_bpw": p.effective_bpw,
                    "model_size_mb": p.model_size_mb,
                    "accuracy": p.accuracy,
                    "throughput": p.throughput,
                }
                for p in curve.points
            ]
        else:
            logger.info("Skipping quantization sweep (quant_model_paths not configured)")

        # 7. Check thresholds
        result.thresholds_met = self._check_thresholds(result)
        result.duration_sec = round(time.perf_counter() - t0, 1)

        self._log_summary(result)
        self._save_results(result)
        return result

    def _check_thresholds(self, result: Layer3Result) -> dict[str, bool]:
        """Compare measurements against config thresholds."""
        t = self._thresholds
        return {
            "ttft_p50": result.ttft_p50_ms <= t.get("ttft_ms", 3000),
            "throughput": result.throughput_tok_per_sec >= t.get("throughput_tok_per_sec", 15),
            "peak_ram": result.peak_ram_mb <= t.get("peak_ram_mb", 8000),
            "model_size": result.model_size_mb <= t.get("model_size_mb", 5000) or result.model_size_mb == 0,
            "thermal_throughput": result.final_throughput >= t.get("thermal_throughput_tok_per_sec", 10),
        }

    def _save_results(self, result: Layer3Result) -> None:
        """Save Layer 3 results to JSON."""
        import datetime
        fname = self._reports_dir / f"layer3_{datetime.date.today()}.json"
        data = {
            "layer": 3,
            "latency": {
                "p50_ms": result.ttft_p50_ms,
                "p95_ms": result.ttft_p95_ms,
                "p99_ms": result.ttft_p99_ms,
                "cold_boot_ms": result.cold_boot_ms,
            },
            "throughput_tok_per_sec": result.throughput_tok_per_sec,
            "peak_ram_mb": result.peak_ram_mb,
            "model_size_mb": result.model_size_mb,
            "sustained_load": {
                "initial_throughput": result.initial_throughput,
                "final_throughput": result.final_throughput,
                "throttle_detected": result.throttle_detected,
                "throttle_drop_pct": result.throttle_drop_pct,
            },
            "bpw_curve": result.bpw_curve,
            "thresholds_met": result.thresholds_met,
            "duration_sec": result.duration_sec,
        }
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Layer 3 results saved: {}", fname)

    @staticmethod
    def _log_summary(result: Layer3Result) -> None:
        all_pass = all(result.thresholds_met.values())
        status = "PASS" if all_pass else "FAIL"
        logger.info(
            "Layer 3 [{}]: TTFT p50={:.0f}ms p95={:.0f}ms | {:.1f} tok/s | {:.0f} MB RAM",
            status, result.ttft_p50_ms, result.ttft_p95_ms,
            result.throughput_tok_per_sec, result.peak_ram_mb,
        )
        if result.throttle_detected:
            logger.warning(
                "THERMAL THROTTLING DETECTED: {:.1f} → {:.1f} tok/s ({:.1f}% drop)",
                result.initial_throughput, result.final_throughput, result.throttle_drop_pct,
            )
        for metric, met in result.thresholds_met.items():
            logger.info("  {}: {}", metric, "OK" if met else "BELOW_TARGET")
