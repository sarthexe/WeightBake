"""Dataset validator and statistics CLI for Phase 2 WeightBake context-distilled data."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import tiktoken
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from src.data_gen.negative_miner import NEGATIVE_CATEGORIES
from src.data_gen.scenarios import ALLOWED_INTENTS
from src.eval.evaluator import evaluate_response

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MINIMAL_PROMPT_PATH = PROJECT_ROOT / "prompts" / "minimal_system_prompt.txt"
BLOATED_PROMPT_PATH = PROJECT_ROOT / "prompts" / "bloated_system_prompt.txt"


class ValidationReport(BaseModel):
    passed: bool
    sft_count: int
    dpo_count: int
    holdout_count: int
    sft_pass_count: int
    dpo_chosen_pass_count: int
    dpo_rejected_fail_count: int
    holdout_pass_count: int
    context_stripped_count: int
    total_context_rows: int
    query_leakage_count: int
    sft_intent_counts: dict[str, int]
    dpo_intent_counts: dict[str, int]
    holdout_intent_counts: dict[str, int]
    negative_category_counts: dict[str, int]
    bloated_prompt_tokens: int
    minimal_prompt_tokens: int
    dataset_tokens_saved: int
    errors: list[str]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def validate_all_datasets(data_dir: Path = DATA_DIR) -> ValidationReport:
    """Validate SFT, DPO, and Holdout datasets and return structured metrics."""
    minimal_prompt = MINIMAL_PROMPT_PATH.read_text(encoding="utf-8").strip()
    bloated_prompt = BLOATED_PROMPT_PATH.read_text(encoding="utf-8").strip()

    enc = tiktoken.get_encoding("cl100k_base")
    bloated_tokens = len(enc.encode(bloated_prompt))
    minimal_tokens = len(enc.encode(minimal_prompt))

    sft_rows = _load_jsonl(data_dir / "sft_train.jsonl")
    dpo_rows = _load_jsonl(data_dir / "dpo_train.jsonl")
    holdout_rows = _load_jsonl(data_dir / "test_holdout.jsonl")

    errors: list[str] = []
    context_stripped_count = 0
    total_context_rows = len(sft_rows) + len(dpo_rows)

    # Validate SFT
    sft_pass_count = 0
    sft_intents: Counter[str] = Counter()
    train_queries: set[str] = set()

    for idx, row in enumerate(sft_rows):
        msgs = row.get("messages", [])
        meta = row.get("metadata", {})
        if len(msgs) != 3:
            errors.append(f"sft[{idx}]: Expected 3 messages, got {len(msgs)}.")
            continue
        sys_msg, user_msg, asst_msg = msgs[0], msgs[1], msgs[2]
        if (
            sys_msg.get("role") == "system"
            and sys_msg.get("content", "").strip() == minimal_prompt
            and bloated_prompt[:80] not in sys_msg.get("content", "")
        ):
            context_stripped_count += 1
        else:
            errors.append(f"sft[{idx}]: System prompt is not minimal_system_prompt.txt.")

        user_q = user_msg.get("content", "").strip()
        train_queries.add(user_q)

        res = evaluate_response(
            asst_msg.get("content", ""),
            expected_intent=meta.get("expected_intent"),
            expected_tool=meta.get("expected_tool"),
            expected_requires_human=meta.get("expected_requires_human"),
        )
        if res.passed_all:
            sft_pass_count += 1
        else:
            errors.append(f"sft[{idx}]: Assistant output failed compliance: {res.errors}")

        parsed_intent = json.loads(asst_msg["content"])["intent"]
        sft_intents[parsed_intent] += 1

    # Validate DPO
    dpo_chosen_pass_count = 0
    dpo_rejected_fail_count = 0
    dpo_intents: Counter[str] = Counter()
    neg_cats: Counter[str] = Counter()

    for idx, row in enumerate(dpo_rows):
        prompt_msgs = row.get("prompt", [])
        chosen_msgs = row.get("chosen", [])
        rejected_msgs = row.get("rejected", [])
        meta = row.get("metadata", {})

        if len(prompt_msgs) != 2 or len(chosen_msgs) != 1 or len(rejected_msgs) != 1:
            errors.append(f"dpo[{idx}]: Invalid TRL DPO triplet structure.")
            continue

        sys_msg, user_msg = prompt_msgs[0], prompt_msgs[1]
        if (
            sys_msg.get("role") == "system"
            and sys_msg.get("content", "").strip() == minimal_prompt
            and bloated_prompt[:80] not in sys_msg.get("content", "")
        ):
            context_stripped_count += 1
        else:
            errors.append(f"dpo[{idx}]: System prompt is not minimal_system_prompt.txt.")

        user_q = user_msg.get("content", "").strip()
        train_queries.add(user_q)

        chosen_res = evaluate_response(
            chosen_msgs[0].get("content", ""),
            expected_intent=meta.get("expected_intent"),
            expected_tool=meta.get("expected_tool"),
            expected_requires_human=meta.get("expected_requires_human"),
        )
        if chosen_res.passed_all:
            dpo_chosen_pass_count += 1
        else:
            errors.append(f"dpo[{idx}]: Chosen failed compliance: {chosen_res.errors}")

        rejected_res = evaluate_response(
            rejected_msgs[0].get("content", ""),
            expected_intent=meta.get("expected_intent"),
            expected_tool=meta.get("expected_tool"),
            expected_requires_human=meta.get("expected_requires_human"),
        )
        if not rejected_res.passed_all:
            dpo_rejected_fail_count += 1
        else:
            errors.append(f"dpo[{idx}]: Rejected unexpectedly passed evaluate_response.")

        parsed_intent = json.loads(chosen_msgs[0]["content"])["intent"]
        dpo_intents[parsed_intent] += 1
        neg_cats[meta.get("negative_category", "unknown")] += 1

    # Validate Holdout & Check Zero Query Leakage
    holdout_pass_count = 0
    holdout_intents: Counter[str] = Counter()
    leakage_count = 0

    golden_queries: set[str] = set()
    golden_path = data_dir / "golden_eval.jsonl"
    if golden_path.exists():
        for row in _load_jsonl(golden_path):
            golden_queries.add(row["user_query"].strip())

    for idx, row in enumerate(holdout_rows):
        uq = row["user_query"].strip()
        if uq in train_queries or uq in golden_queries:
            leakage_count += 1
            errors.append(f"holdout[{idx}]: Query leakage detected for '{uq[:60]}'.")

        res = evaluate_response(
            row["gold_aligned_output"],
            expected_intent=row["expected_intent"],
            expected_tool=row["expected_tool"],
            expected_requires_human=row["expected_requires_human"],
        )
        if res.passed_all:
            holdout_pass_count += 1
        else:
            errors.append(f"holdout[{idx}]: Gold output failed compliance: {res.errors}")
        holdout_intents[row["expected_intent"]] += 1

    # Check class balance across all 8 intents & 6 negative failure modes
    for name, counter, total in (
        ("SFT", sft_intents, len(sft_rows)),
        ("DPO", dpo_intents, len(dpo_rows)),
        ("Holdout", holdout_intents, len(holdout_rows)),
    ):
        expected_per_intent = total // len(ALLOWED_INTENTS)
        for intent in ALLOWED_INTENTS:
            if counter[intent] != expected_per_intent:
                errors.append(
                    f"{name}: Intent '{intent}' count {counter[intent]} != expected {expected_per_intent}."
                )

    for neg_cat in NEGATIVE_CATEGORIES:
        if neg_cats[neg_cat] < (len(dpo_rows) // len(NEGATIVE_CATEGORIES)):
            errors.append(
                f"DPO: Negative category '{neg_cat}' underrepresented ({neg_cats[neg_cat]})."
            )

    dataset_tokens_saved = (bloated_tokens - minimal_tokens) * total_context_rows
    passed = (
        len(errors) == 0
        and sft_pass_count == len(sft_rows)
        and dpo_chosen_pass_count == len(dpo_rows)
        and dpo_rejected_fail_count == len(dpo_rows)
        and holdout_pass_count == len(holdout_rows)
        and context_stripped_count == total_context_rows
        and leakage_count == 0
    )

    return ValidationReport(
        passed=passed,
        sft_count=len(sft_rows),
        dpo_count=len(dpo_rows),
        holdout_count=len(holdout_rows),
        sft_pass_count=sft_pass_count,
        dpo_chosen_pass_count=dpo_chosen_pass_count,
        dpo_rejected_fail_count=dpo_rejected_fail_count,
        holdout_pass_count=holdout_pass_count,
        context_stripped_count=context_stripped_count,
        total_context_rows=total_context_rows,
        query_leakage_count=leakage_count,
        sft_intent_counts=dict(sft_intents),
        dpo_intent_counts=dict(dpo_intents),
        holdout_intent_counts=dict(holdout_intents),
        negative_category_counts=dict(neg_cats),
        bloated_prompt_tokens=bloated_tokens,
        minimal_prompt_tokens=minimal_tokens,
        dataset_tokens_saved=dataset_tokens_saved,
        errors=errors,
    )


def main() -> int:
    console = Console()
    report = validate_all_datasets()

    summary_table = Table(title="WeightBake Phase 2 Dataset Integrity & Context Stripping Audit")
    summary_table.add_column("Dataset / Check", style="cyan", no_wrap=True)
    summary_table.add_column("Result", justify="right", style="bold")
    summary_table.add_column("Status", justify="left")

    summary_table.add_row(
        "data/sft_train.jsonl (Gold Compliance)",
        f"{report.sft_pass_count}/{report.sft_count}",
        "[green]100% PASS[/green]" if report.sft_pass_count == report.sft_count else "[red]FAIL[/red]",
    )
    summary_table.add_row(
        "data/dpo_train.jsonl (Chosen Compliance)",
        f"{report.dpo_chosen_pass_count}/{report.dpo_count}",
        "[green]100% PASS[/green]" if report.dpo_chosen_pass_count == report.dpo_count else "[red]FAIL[/red]",
    )
    summary_table.add_row(
        "data/dpo_train.jsonl (Rejected Failure Rate)",
        f"{report.dpo_rejected_fail_count}/{report.dpo_count}",
        "[green]100% FAIL AS EXPECTED[/green]" if report.dpo_rejected_fail_count == report.dpo_count else "[red]FAIL[/red]",
    )
    summary_table.add_row(
        "data/test_holdout.jsonl (Gold Compliance)",
        f"{report.holdout_pass_count}/{report.holdout_count}",
        "[green]100% PASS[/green]" if report.holdout_pass_count == report.holdout_count else "[red]FAIL[/red]",
    )
    summary_table.add_row(
        "Context Stripping (10-tok Minimal Prompt)",
        f"{report.context_stripped_count}/{report.total_context_rows}",
        "[green]100% STRIPPED[/green]" if report.context_stripped_count == report.total_context_rows else "[red]FAIL[/red]",
    )
    summary_table.add_row(
        "Holdout vs. Train Query Leakage",
        f"{report.query_leakage_count}",
        "[green]ZERO LEAKAGE[/green]" if report.query_leakage_count == 0 else "[red]LEAKAGE[/red]",
    )
    summary_table.add_row(
        "Dataset Prefill Tokens Saved",
        f"{report.dataset_tokens_saved:,} tokens",
        f"[green]{report.bloated_prompt_tokens:,} -> {report.minimal_prompt_tokens} tok/sample[/green]",
    )
    console.print(summary_table)

    intent_table = Table(title="Stratified Intent Balance Across Datasets")
    intent_table.add_column("Intent", style="cyan", no_wrap=True)
    intent_table.add_column("SFT Train", justify="right")
    intent_table.add_column("DPO Train", justify="right")
    intent_table.add_column("Holdout", justify="right")
    for intent in ALLOWED_INTENTS:
        intent_table.add_row(
            intent,
            str(report.sft_intent_counts.get(intent, 0)),
            str(report.dpo_intent_counts.get(intent, 0)),
            str(report.holdout_intent_counts.get(intent, 0)),
        )
    console.print(intent_table)

    neg_table = Table(title="DPO Hard-Negative Failure Mode Distribution")
    neg_table.add_column("Failure Category", style="cyan", no_wrap=True)
    neg_table.add_column("Count", justify="right")
    neg_table.add_column("Share", justify="right")
    for cat in NEGATIVE_CATEGORIES:
        cnt = report.negative_category_counts.get(cat, 0)
        share = (cnt / report.dpo_count * 100.0) if report.dpo_count else 0.0
        neg_table.add_row(cat, str(cnt), f"{share:.1f}%")
    console.print(neg_table)

    if not report.passed:
        for err in report.errors[:15]:
            console.print(f"[red]ERROR:[/red] {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
