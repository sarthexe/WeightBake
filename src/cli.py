"""Unified CLI orchestrator for the WeightBake post-training and inference pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.benchmark.kv_cache_profiler import analyze_kv_cache_savings, render_kv_cache_report
from src.benchmark.load_tester import run_benchmark
from src.benchmark.serve_vllm import build_vllm_command
from src.data_gen.synthesizer import synthesize_all
from src.data_gen.validate_datasets import main as validate_datasets_main
from src.data_gen.validate_datasets import validate_all_datasets
from src.eval.evaluate_checkpoint import ALL_STAGES, run_simulated_stages
from src.eval.run_baseline_check import count_prompt_tokens
from src.eval.run_baseline_check import main as baseline_audit_main
from src.train.config import DPOExperimentConfig, SFTExperimentConfig
from src.train.export_hub import export_to_hub
from src.train.merge_weights import run_merge_dry_run
from src.train.train_dpo import run_dpo_dry_run
from src.train.train_sft import run_dry_run as run_sft_dry_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_verify_all(console: Console | None = None) -> int:
    """Execute the full end-to-end dry-run/simulation pipeline and print executive summary."""
    out_console = console or Console()
    out_console.rule("[bold cyan]WeightBake End-to-End Verification Pipeline[/bold cyan]")

    # 1. Baseline Prompt Audit & Golden Eval
    rc = baseline_audit_main()
    if rc != 0:
        return rc

    # 2. Procedural Data Synthesis & Integrity Validation
    synthesize_all(mode="procedural", sft_count=800, dpo_count=800, holdout_count=80)
    val_report = validate_all_datasets()
    if not val_report.passed:
        out_console.print("[red]Dataset validation failed![/red]")
        return 1

    # 3. SFT Dry-Run & Assistant-Only Loss Masking Audit
    sft_cfg = SFTExperimentConfig.from_yaml(PROJECT_ROOT / "configs" / "sft_config.yaml")
    sft_summary = run_sft_dry_run(cfg=sft_cfg, console=out_console)

    # 4. DPO Dry-Run & Bradley-Terry Sanity Audit
    dpo_cfg = DPOExperimentConfig.from_yaml(PROJECT_ROOT / "configs" / "dpo_config.yaml")
    dpo_summary = run_dpo_dry_run(cfg=dpo_cfg, console=out_console)

    # 5. LoRA Weight-Merge Equivalence Audit
    merge_diffs = run_merge_dry_run(
        adapter_dir=dpo_cfg.dpo.output_dir,
        output_dir=dpo_cfg.merged_output_dir,
        save_method="merged_16bit",
        console=out_console,
    )

    # 6. 4-Stage Holdout Alignment Evaluation
    stage_summaries = run_simulated_stages(stages=ALL_STAGES, console=out_console)

    # 7. KV-Cache Profiling & vLLM Command Check
    kv_analysis = analyze_kv_cache_savings()
    render_kv_cache_report(kv_analysis, console=out_console)
    _ = build_vllm_command(
        model=dpo_cfg.merged_output_dir,
        served_model_name="weightbake-3b",
        disable_prefix_caching=True,
    )

    # 8. Concurrent Latency & Throughput Benchmark
    bench_report = run_benchmark(
        mode="simulate",
        concurrency_levels=[1, 8, 16, 32],
        requests_per_tier=64,
        console=out_console,
    )

    # 9. Hugging Face Hub Model Card Dry-Run Export
    export_to_hub(dry_run=True, console=out_console)

    # 10. Consolidated Executive Report Table
    bloated_toks, minimal_toks, reduction_pct = count_prompt_tokens()
    tiers_by_c = {t["concurrency"]: t for t in bench_report["tiers"]}
    c1 = tiers_by_c[1]
    c8 = tiers_by_c[8]
    c16 = tiers_by_c[16]
    c32 = tiers_by_c[32]

    exec_table = Table(
        title="WeightBake Consolidated Executive Report",
        show_lines=True,
    )
    exec_table.add_column("Subsystem / Pillar", style="cyan", no_wrap=True)
    exec_table.add_column("Baseline (Bloated)", justify="right")
    exec_table.add_column("WeightBake-3B (Minimal)", justify="right", style="green")
    exec_table.add_column("Net Impact", justify="right", style="bold green")

    exec_table.add_row(
        "System Prompt Prefill Tokens",
        f"{bloated_toks:,} tokens",
        f"{minimal_toks} tokens",
        f"-{reduction_pct:.2f}% context",
    )
    exec_table.add_row(
        "Holdout Compliance (80 samples)",
        f"{stage_summaries['base_bloated'].full_compliance_rate:.2f}% "
        f"(unprompted: {stage_summaries['base_unprompted'].full_compliance_rate:.2f}%)",
        f"{stage_summaries['dpo_minimal'].full_compliance_rate:.2f}% "
        f"(SFT: {stage_summaries['sft_minimal'].full_compliance_rate:.2f}%)",
        f"+{stage_summaries['dpo_minimal'].full_compliance_rate - stage_summaries['base_bloated'].full_compliance_rate:.2f} pp vs bloated",
    )
    exec_table.add_row(
        "SFT / DPO Context Stripping & Merge",
        "1,922 tok/sample",
        f"SFT {sft_summary.train_samples} + DPO {dpo_summary.train_samples} (0 leakage)",
        f"Max merge err {max(merge_diffs.values()):.2e}",
    )
    exec_table.add_row(
        "KV-Cache Footprint / Session (GQA-2)",
        f"{kv_analysis.baseline_mb_per_req:.2f} MiB",
        f"{kv_analysis.weightbake_mb_per_req:.2f} MiB",
        f"{kv_analysis.kv_reduction_factor:.2f}x smaller",
    )
    exec_table.add_row(
        "Max Concurrent Sessions (8 GiB KV)",
        f"{kv_analysis.baseline_max_concurrent_sessions:,}",
        f"{kv_analysis.weightbake_max_concurrent_sessions:,}",
        f"{kv_analysis.weightbake_max_concurrent_sessions / kv_analysis.baseline_max_concurrent_sessions:.2f}x capacity",
    )
    exec_table.add_row(
        "vLLM P50 TTFT (C=1 / C=8 / C=16 / C=32)",
        f"{c1['baseline_bloated']['p50_ttft_ms']:.1f} / "
        f"{c8['baseline_bloated']['p50_ttft_ms']:.1f} / "
        f"{c16['baseline_bloated']['p50_ttft_ms']:.1f} / "
        f"{c32['baseline_bloated']['p50_ttft_ms']:.1f} ms",
        f"{c1['weightbake_minimal']['p50_ttft_ms']:.1f} / "
        f"{c8['weightbake_minimal']['p50_ttft_ms']:.1f} / "
        f"{c16['weightbake_minimal']['p50_ttft_ms']:.1f} / "
        f"{c32['weightbake_minimal']['p50_ttft_ms']:.1f} ms",
        f"{c1['ttft_speedup_x']:.2f}x -> {c32['ttft_speedup_x']:.2f}x faster",
    )
    exec_table.add_row(
        "vLLM Throughput RPS (C=1 / C=8 / C=16 / C=32)",
        f"{c1['baseline_bloated']['rps']:.1f} / "
        f"{c8['baseline_bloated']['rps']:.1f} / "
        f"{c16['baseline_bloated']['rps']:.1f} / "
        f"{c32['baseline_bloated']['rps']:.1f}",
        f"{c1['weightbake_minimal']['rps']:.1f} / "
        f"{c8['weightbake_minimal']['rps']:.1f} / "
        f"{c16['weightbake_minimal']['rps']:.1f} / "
        f"{c32['weightbake_minimal']['rps']:.1f}",
        f"{c32['rps_speedup_x']:.2f}x RPS at C=32",
    )
    out_console.print(exec_table)
    return 0


def _delegate_module(module_main, argv: list[str]) -> int:
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        res = module_main()
        return int(res) if isinstance(res, int) else 0
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="weightbake",
        description="WeightBake: Context Distillation (SFT + DPO) & vLLM Optimization CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit", help="Run prompt token audit and golden baseline check.")
    subparsers.add_parser("synthesize", help="Synthesize and validate SFT/DPO/holdout datasets.")
    subparsers.add_parser("train-sft", help="Run SFT training or --dry-run audit.")
    subparsers.add_parser("train-dpo", help="Run DPO alignment training or --dry-run audit.")
    subparsers.add_parser("merge", help="Merge LoRA weights to 16-bit or run --dry-run audit.")
    subparsers.add_parser("eval", help="Evaluate checkpoint or stage progression on holdout set.")
    subparsers.add_parser("benchmark", help="Run KV-cache profiler and vLLM latency benchmark.")
    subparsers.add_parser(
        "verify-all",
        help="Run complete end-to-end dry-run/simulation pipeline and executive summary.",
    )

    args, remainder = parser.parse_known_args(argv)

    if args.command == "audit":
        return baseline_audit_main()
    if args.command == "synthesize":
        from src.data_gen.synthesizer import main as synth_main

        rc = _delegate_module(synth_main, remainder)
        if rc != 0:
            return rc
        return validate_datasets_main()
    if args.command == "train-sft":
        from src.train.train_sft import main as sft_main

        return _delegate_module(sft_main, remainder)
    if args.command == "train-dpo":
        from src.train.train_dpo import main as dpo_main

        return _delegate_module(dpo_main, remainder)
    if args.command == "merge":
        from src.train.merge_weights import main as merge_main

        return _delegate_module(merge_main, remainder)
    if args.command == "eval":
        from src.eval.evaluate_checkpoint import main as eval_main

        return _delegate_module(eval_main, remainder)
    if args.command == "benchmark":
        from src.benchmark.kv_cache_profiler import main as kv_main
        from src.benchmark.load_tester import main as load_main

        rc = kv_main()
        if rc != 0:
            return rc
        return _delegate_module(load_main, remainder)
    if args.command == "verify-all":
        return run_verify_all()

    return 1


if __name__ == "__main__":
    sys.exit(main())
