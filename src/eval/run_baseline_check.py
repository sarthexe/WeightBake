"""CLI audit and smoke-test runner for Phase 1 WeightBake evaluation harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tiktoken
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.eval.evaluator import ComplianceResult, evaluate_response

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOATED_PROMPT_PATH = PROJECT_ROOT / "prompts" / "bloated_system_prompt.txt"
MINIMAL_PROMPT_PATH = PROJECT_ROOT / "prompts" / "minimal_system_prompt.txt"
GOLDEN_EVAL_PATH = PROJECT_ROOT / "data" / "golden_eval.jsonl"

MIN_BLOATED_TOKENS = 1400


def count_prompt_tokens(
    bloated_path: Path = BLOATED_PROMPT_PATH,
    minimal_path: Path = MINIMAL_PROMPT_PATH,
    encoding_name: str = "cl100k_base",
) -> tuple[int, int, float]:
    """Return (bloated_tokens, minimal_tokens, reduction_pct)."""
    enc = tiktoken.get_encoding(encoding_name)
    bloated_text = bloated_path.read_text(encoding="utf-8")
    minimal_text = minimal_path.read_text(encoding="utf-8")

    bloated_tokens = len(enc.encode(bloated_text))
    minimal_tokens = len(enc.encode(minimal_text))
    reduction_pct = ((bloated_tokens - minimal_tokens) / bloated_tokens) * 100.0
    return bloated_tokens, minimal_tokens, reduction_pct


def load_golden_dataset(path: Path = GOLDEN_EVAL_PATH) -> list[dict]:
    """Load all rows from the golden evaluation JSONL file."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def evaluate_dataset(
    rows: list[dict],
) -> tuple[list[ComplianceResult], list[ComplianceResult]]:
    """Evaluate mock_unaligned_output and gold_aligned_output across all rows."""
    unaligned_results: list[ComplianceResult] = []
    gold_results: list[ComplianceResult] = []

    for row in rows:
        expected_intent = row["expected_intent"]
        expected_tool = row["expected_tool"]
        expected_requires_human = row["expected_requires_human"]

        unaligned_results.append(
            evaluate_response(
                row["mock_unaligned_output"],
                expected_intent=expected_intent,
                expected_tool=expected_tool,
                expected_requires_human=expected_requires_human,
            )
        )
        gold_results.append(
            evaluate_response(
                row["gold_aligned_output"],
                expected_intent=expected_intent,
                expected_tool=expected_tool,
                expected_requires_human=expected_requires_human,
            )
        )
    return unaligned_results, gold_results


def main() -> int:
    console = Console()

    bloated_tokens, minimal_tokens, reduction_pct = count_prompt_tokens()
    token_table = Table(title="Prefill System Prompt Token Audit (cl100k_base)")
    token_table.add_column("Prompt Artifact", style="cyan", no_wrap=True)
    token_table.add_column("Token Count", justify="right", style="bold")
    token_table.add_column("Status", justify="left")

    bloated_ok = bloated_tokens >= MIN_BLOATED_TOKENS
    token_table.add_row(
        "prompts/bloated_system_prompt.txt",
        f"{bloated_tokens:,}",
        "[green]PASS (>= 1,400 tokens)[/green]"
        if bloated_ok
        else f"[red]FAIL (< {MIN_BLOATED_TOKENS} tokens)[/red]",
    )
    token_table.add_row(
        "prompts/minimal_system_prompt.txt",
        f"{minimal_tokens:,}",
        "[green]Compact Runtime Trigger[/green]",
    )
    token_table.add_row(
        "Prefill Token Reduction",
        f"-{bloated_tokens - minimal_tokens:,}",
        f"[bold green]{reduction_pct:.2f}% reduction[/bold green]",
    )
    console.print(token_table)

    rows = load_golden_dataset()
    unaligned_results, gold_results = evaluate_dataset(rows)
    total = len(rows)

    check_names = list(gold_results[0].checks.keys())
    eval_table = Table(
        title=f"Golden Benchmark Deterministic Compliance ({total} Test Cases)"
    )
    eval_table.add_column("Check / Metric", style="cyan", no_wrap=True)
    eval_table.add_column("Mock Unaligned (Baseline)", justify="right")
    eval_table.add_column("Gold Aligned (Target)", justify="right")

    for check in check_names:
        unaligned_pass = sum(1 for r in unaligned_results if r.checks[check])
        gold_pass = sum(1 for r in gold_results if r.checks[check])
        eval_table.add_row(
            check,
            f"{unaligned_pass}/{total} ({unaligned_pass / total * 100:.1f}%)",
            f"{gold_pass}/{total} ({gold_pass / total * 100:.1f}%)",
        )

    eval_table.add_section()

    unaligned_avg_score = sum(r.score for r in unaligned_results) / total * 100.0
    gold_avg_score = sum(r.score for r in gold_results) / total * 100.0
    eval_table.add_row(
        "Mean Check Score",
        f"{unaligned_avg_score:.1f}%",
        f"{gold_avg_score:.1f}%",
    )

    unaligned_full = sum(1 for r in unaligned_results if r.passed_all)
    gold_full = sum(1 for r in gold_results if r.passed_all)
    eval_table.add_row(
        "[bold]Overall Full Compliance (passed_all)[/bold]",
        f"[red]{unaligned_full}/{total} ({unaligned_full / total * 100:.1f}%)[/red]",
        f"[green]{gold_full}/{total} ({gold_full / total * 100:.1f}%)[/green]",
    )
    console.print(eval_table)

    if gold_full != total:
        for row, res in zip(rows, gold_results):
            if not res.passed_all:
                console.print(
                    Panel(
                        "\n".join(res.errors),
                        title=f"Gold Failure: {row['id']}",
                        border_style="red",
                    )
                )

    if not bloated_ok or unaligned_full != 0 or gold_full != total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
