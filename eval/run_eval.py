"""CLI entry point for the FinEdge eval pipeline.

Usage:
    python eval/run_eval.py --layer 1
    python eval/run_eval.py --layer 3 --model-path models/qwen3-8b-Q4_K_M.gguf
    python eval/run_eval.py --layers 1,2,3
    python eval/run_eval.py --report
    python eval/run_eval.py --report --results-dir eval/reports/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add project root to path so `eval.*` imports resolve
_PROJECT_ROOT = Path(__file__).parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_config(config_path: str | None = None) -> dict:
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed — run: pip install pyyaml")
        return {}

    path = Path(config_path) if config_path else Path(__file__).parent / "configs" / "eval_config.yaml"
    if not path.exists():
        logger.warning("Config not found at {} — using defaults", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_adapter(config: dict, model_path: str | None = None) -> object | None:
    from eval.model_adapter import ModelAdapter

    if model_path:
        config.setdefault("model", {})["path"] = model_path

    model_conf = config.get("model", {})
    if not model_conf.get("path"):
        logger.warning("No model path configured — some layers will skip model-dependent checks")
        return None

    try:
        adapter = ModelAdapter(config)
        adapter.load()
        return adapter
    except Exception as exc:
        logger.error("Failed to load model adapter: {}", exc)
        return None


def run_layer1(config: dict, adapter) -> object:
    from eval.runners.layer1_unit import Layer1Runner
    runner = Layer1Runner(config)
    result = runner.run(adapter=adapter)
    return result


def run_layer2(config: dict, adapter) -> object:
    if adapter is None:
        logger.error("Layer 2 requires a loaded model adapter — provide --model-path")
        sys.exit(1)
    from eval.runners.layer2_trajectory import Layer2Runner
    runner = Layer2Runner(config)
    return runner.run(adapter)


def run_layer3(config: dict, adapter) -> object:
    if adapter is None:
        logger.error("Layer 3 requires a loaded model adapter — provide --model-path")
        sys.exit(1)
    from eval.runners.layer3_edge import Layer3Runner
    runner = Layer3Runner(config)
    return runner.run(adapter)


def run_layer4(config: dict, adapter, run_promptfoo: bool = False) -> object:
    if adapter is None:
        logger.error("Layer 4 requires a loaded model adapter — provide --model-path")
        sys.exit(1)
    from eval.runners.layer4_safety import Layer4Runner
    runner = Layer4Runner(config)
    return runner.run(adapter, run_promptfoo=run_promptfoo)


def run_report(config: dict, results_dir: str | None = None) -> None:
    from eval.report import CLEARReportGenerator

    reports_path = Path(results_dir) if results_dir else None
    gen = CLEARReportGenerator(config)
    if reports_path:
        gen._reports_dir = reports_path

    raw = gen.load_latest_layer_results()

    # Reconstruct result objects from cached JSON
    layer1 = _layer1_from_json(raw.get("layer1")) if raw.get("layer1") else None
    layer2 = _layer2_from_json(raw.get("layer2")) if raw.get("layer2") else None
    layer3 = _layer3_from_json(raw.get("layer3")) if raw.get("layer3") else None
    layer4 = _layer4_from_json(raw.get("layer4")) if raw.get("layer4") else None

    if not any([layer1, layer2, layer3, layer4]):
        logger.warning("No cached layer results found in {} — run at least one layer first", gen._reports_dir)

    report = gen.build_clear_table(layer1, layer2, layer3, layer4)
    md_path = gen.export_markdown(report)
    json_path = gen.export_json(report)
    logger.info("CLEAR report: {}", md_path)
    logger.info("JSON report:  {}", json_path)
    _print_clear_table(report)


def _layer1_from_json(data: dict) -> object:
    from eval.runners.layer1_unit import Layer1Result
    r = Layer1Result()
    r.total_cases = data.get("total_cases", 0)
    r.passed_cases = data.get("passed_cases", 0)
    r.section_results = data.get("section_results", {})
    r.duration_sec = data.get("duration_sec", 0.0)
    return r


def _layer2_from_json(data: dict) -> object:
    from eval.runners.layer2_trajectory import Layer2Result
    r = Layer2Result()
    r.total_cases = data.get("total_cases", 0)
    r.passed_cases = data.get("passed_cases", 0)
    r.task_completion_rate = data.get("task_completion_rate", 0.0)
    r.pass_at_1 = data.get("pass_at_1", 0.0)
    r.tool_call_accuracy = data.get("tool_call_accuracy")
    r.redundant_call_rate = data.get("redundant_call_rate")
    r.category_breakdown = data.get("category_breakdown", {})
    r.duration_sec = data.get("duration_sec", 0.0)
    return r


def _layer3_from_json(data: dict) -> object:
    from eval.runners.layer3_edge import Layer3Result
    r = Layer3Result()
    lat = data.get("latency", {})
    r.ttft_p50_ms = lat.get("p50_ms", 0.0)
    r.ttft_p95_ms = lat.get("p95_ms", 0.0)
    r.ttft_p99_ms = lat.get("p99_ms", 0.0)
    r.cold_boot_ms = lat.get("cold_boot_ms")
    r.throughput_tok_per_sec = data.get("throughput_tok_per_sec", 0.0)
    r.peak_ram_mb = data.get("peak_ram_mb", 0.0)
    r.model_size_mb = data.get("model_size_mb", 0.0)
    sl = data.get("sustained_load", {})
    r.initial_throughput = sl.get("initial_throughput", 0.0)
    r.final_throughput = sl.get("final_throughput", 0.0)
    r.throttle_detected = sl.get("throttle_detected", False)
    r.throttle_drop_pct = sl.get("throttle_drop_pct", 0.0)
    r.bpw_curve = data.get("bpw_curve", [])
    r.thresholds_met = data.get("thresholds_met", {})
    r.duration_sec = data.get("duration_sec", 0.0)
    return r


def _layer4_from_json(data: dict) -> object:
    from eval.runners.layer4_safety import Layer4Result
    r = Layer4Result()
    r.pas_score = data.get("pas_score", 0.0)
    r.total_safety_cases = data.get("total_safety_cases", 0)
    r.passed_safety_cases = data.get("passed_safety_cases", 0)
    r.pii_leakage_rate = data.get("pii_leakage_rate", 0.0)
    r.pii_incidents = data.get("pii_incidents", [])
    r.promptfoo_ran = data.get("promptfoo_ran", False)
    r.promptfoo_pass_rate = data.get("promptfoo_pass_rate")
    r.category_breakdown = data.get("category_breakdown", {})
    r.failure_mode_taxonomy = data.get("failure_mode_taxonomy", {})
    r.duration_sec = data.get("duration_sec", 0.0)
    return r


def _print_clear_table(report) -> None:
    """Print the CLEAR table to stdout."""
    print(f"\n{'='*70}")
    print(f"  CLEAR Report — {report.model_name}  |  {report.overall_status}")
    print(f"  Metrics Met: {report.pass_count}/{report.total_count}")
    print(f"{'='*70}")
    print(f"{'Dimension':<15} {'Metric':<35} {'Value':<15} {'Target':<18} {'Status'}")
    print(f"{'-'*15} {'-'*35} {'-'*15} {'-'*18} {'-'*8}")
    for dim in report.dimensions:
        value_str = f"{dim.value} {dim.unit}" if dim.value is not None else "—"
        status = "PASS" if dim.met else ("NO DATA" if dim.value is None else "FAIL")
        print(f"{dim.dimension:<15} {dim.metric:<35} {value_str:<15} {dim.target:<18} {status}")
    print(f"{'='*70}\n")


def _parse_layers(layers_str: str) -> list[int]:
    try:
        return [int(x.strip()) for x in layers_str.split(",")]
    except ValueError:
        logger.error("Invalid --layers value '{}' — expected comma-separated integers", layers_str)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinEdge Eval Pipeline — CLEAR framework evaluation for Qwen3-8B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval/run_eval.py --layer 1
  python eval/run_eval.py --layer 3 --model-path models/qwen3-8b-Q4_K_M.gguf
  python eval/run_eval.py --layers 1,3 --model-path models/qwen3-8b-Q4_K_M.gguf
  python eval/run_eval.py --layer 4 --model-path models/qwen3-8b-Q4_K_M.gguf --promptfoo
  python eval/run_eval.py --report
  python eval/run_eval.py --report --results-dir eval/reports/
        """,
    )
    parser.add_argument("--layer", type=int, choices=[1, 2, 3, 4], help="Run a single layer")
    parser.add_argument("--layers", type=str, help="Run multiple layers, e.g. --layers 1,3")
    parser.add_argument("--model-path", type=str, help="Path to GGUF model file")
    parser.add_argument("--config", type=str, help="Path to eval_config.yaml (default: eval/configs/eval_config.yaml)")
    parser.add_argument("--report", action="store_true", help="Generate CLEAR report from cached results")
    parser.add_argument("--results-dir", type=str, help="Directory with layer result JSONs (for --report)")
    parser.add_argument("--promptfoo", action="store_true", help="Run Promptfoo red-team in Layer 4")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    config = _load_config(args.config)

    if args.report:
        run_report(config, args.results_dir)
        return

    layers_to_run: list[int] = []
    if args.layer:
        layers_to_run = [args.layer]
    elif args.layers:
        layers_to_run = _parse_layers(args.layers)
    else:
        enabled = config.get("eval", {}).get("layers_enabled", [1, 3])
        layers_to_run = enabled
        logger.info("No --layer specified — running configured layers: {}", layers_to_run)

    needs_model = any(l in layers_to_run for l in [2, 3, 4]) or (
        1 in layers_to_run and config.get("model", {}).get("path")
    )

    adapter = None
    if needs_model or args.model_path:
        adapter = _load_adapter(config, args.model_path)

    results: dict[int, object] = {}

    for layer in sorted(set(layers_to_run)):
        logger.info("━━━ Running Layer {} ━━━", layer)
        try:
            if layer == 1:
                results[1] = run_layer1(config, adapter)
            elif layer == 2:
                results[2] = run_layer2(config, adapter)
            elif layer == 3:
                results[3] = run_layer3(config, adapter)
            elif layer == 4:
                results[4] = run_layer4(config, adapter, run_promptfoo=args.promptfoo)
        except SystemExit:
            raise
        except Exception as exc:
            logger.error("Layer {} failed: {}", layer, exc)

    if len(results) > 1 or (len(results) == 1 and args.report is False):
        logger.info("━━━ Generating CLEAR Report ━━━")
        from eval.report import CLEARReportGenerator
        gen = CLEARReportGenerator(config)
        report = gen.build_clear_table(
            layer1_result=results.get(1),
            layer2_result=results.get(2),
            layer3_result=results.get(3),
            layer4_result=results.get(4),
        )
        gen.export_markdown(report)
        gen.export_json(report)
        _print_clear_table(report)


if __name__ == "__main__":
    main()
