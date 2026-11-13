"""Async teacher distillation & procedural data synthesis engine for WeightBake."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from tqdm import tqdm

from src.data_gen.negative_miner import NEGATIVE_CATEGORIES, generate_negative
from src.data_gen.scenarios import (
    ALLOWED_INTENTS,
    SynthesizedScenario,
    generate_stratified_scenarios,
)
from src.eval.evaluator import evaluate_response

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOATED_PROMPT_PATH = PROJECT_ROOT / "prompts" / "bloated_system_prompt.txt"
MINIMAL_PROMPT_PATH = PROJECT_ROOT / "prompts" / "minimal_system_prompt.txt"
GOLDEN_EVAL_PATH = PROJECT_ROOT / "data" / "golden_eval.jsonl"
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "raw_cache"


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _load_golden_queries(path: Path = GOLDEN_EVAL_PATH) -> set[str]:
    if not path.exists():
        return set()
    queries: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                queries.add(json.loads(stripped)["user_query"].strip())
    return queries


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return cached
    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rec = json.loads(stripped)
                cached[rec["scenario_id"]] = rec
    return cached


async def _distill_single_scenario(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    bloated_prompt: str,
    scenario: SynthesizedScenario,
    cache_path: Path,
    cache_lock: asyncio.Lock,
    max_validation_retries: int = 3,
) -> str:
    """Call Teacher LLM with bloated_system_prompt.txt and rejection-sample via evaluate_response."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": bloated_prompt},
        {"role": "user", "content": scenario.user_query},
    ]

    last_errors: list[str] = []
    for attempt in range(max_validation_retries):
        async with semaphore:
            backoff = 1.0
            raw_content = ""
            for api_try in range(4):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.2 if attempt == 0 else 0.0,
                        max_tokens=300,
                    )
                    raw_content = (resp.choices[0].message.content or "").strip()
                    break
                except Exception as exc:
                    if api_try == 3:
                        raise RuntimeError(
                            f"API failure for {scenario.scenario_id}: {exc}"
                        ) from exc
                    await asyncio.sleep(backoff)
                    backoff *= 2.0

        eval_res = evaluate_response(
            raw_content,
            expected_intent=scenario.expected_intent,
            expected_tool=scenario.expected_tool,
            expected_requires_human=scenario.expected_requires_human,
        )
        if eval_res.passed_all:
            async with cache_lock:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "scenario_id": scenario.scenario_id,
                                "user_query": scenario.user_query,
                                "assistant_output": raw_content,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            return raw_content

        last_errors = eval_res.errors
        messages.append({"role": "assistant", "content": raw_content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Validation failed with errors: "
                    + "; ".join(last_errors)
                    + ". Output ONLY a corrected raw JSON object."
                ),
            }
        )

    raise RuntimeError(
        f"Teacher failed validation after {max_validation_retries} retries for "
        f"{scenario.scenario_id}: {last_errors}"
    )


async def _run_api_distillation(
    scenarios: list[SynthesizedScenario],
    model: str,
    concurrency: int,
    cache_name: str,
    bloated_prompt: str,
) -> dict[str, str]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for --mode api. "
            "Set it in .env or environment, or run with --mode procedural."
        )
    base_url = os.getenv("OPENAI_BASE_URL") or None
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    cache_path = RAW_CACHE_DIR / f"{cache_name}.jsonl"
    cached = _load_cache(cache_path)
    outputs: dict[str, str] = {
        sid: rec["assistant_output"]
        for sid, rec in cached.items()
        if evaluate_response(rec["assistant_output"]).passed_all
    }

    pending = [s for s in scenarios if s.scenario_id not in outputs]
    if not pending:
        return outputs

    semaphore = asyncio.Semaphore(concurrency)
    cache_lock = asyncio.Lock()

    async def _task(sc: SynthesizedScenario) -> tuple[str, str]:
        out = await _distill_single_scenario(
            client=client,
            semaphore=semaphore,
            model=model,
            bloated_prompt=bloated_prompt,
            scenario=sc,
            cache_path=cache_path,
            cache_lock=cache_lock,
        )
        return sc.scenario_id, out

    coros = [_task(sc) for sc in pending]
    for fut in tqdm(
        asyncio.as_completed(coros),
        total=len(coros),
        desc=f"Distilling ({cache_name})",
    ):
        sid, out = await fut
        outputs[sid] = out

    await client.close()
    return outputs


def _format_sft_rows(
    scenarios: list[SynthesizedScenario],
    minimal_prompt: str,
    distilled_outputs: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        gold = (
            distilled_outputs[sc.scenario_id]
            if distilled_outputs is not None
            else sc.gold_aligned_output
        )
        res = evaluate_response(
            gold,
            expected_intent=sc.expected_intent,
            expected_tool=sc.expected_tool,
            expected_requires_human=sc.expected_requires_human,
        )
        if not res.passed_all:
            raise ValueError(
                f"SFT sample {sc.scenario_id} failed compliance check: {res.errors}"
            )
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": minimal_prompt},
                    {"role": "user", "content": sc.user_query},
                    {"role": "assistant", "content": gold},
                ],
                "metadata": {
                    "id": sc.scenario_id,
                    "expected_intent": sc.expected_intent,
                    "expected_urgency": sc.expected_urgency,
                    "expected_requires_human": sc.expected_requires_human,
                    "expected_tool": sc.expected_tool,
                    "speech_style": sc.speech_style,
                },
            }
        )
    return rows


def _format_dpo_rows(
    scenarios: list[SynthesizedScenario],
    minimal_prompt: str,
    seed: int,
    distilled_outputs: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    # Build stratified, evenly balanced list of negative categories
    cats: list[str] = [
        NEGATIVE_CATEGORIES[i % len(NEGATIVE_CATEGORIES)] for i in range(len(scenarios))
    ]
    rng.shuffle(cats)

    rows: list[dict[str, Any]] = []
    for idx, (sc, neg_cat) in enumerate(zip(scenarios, cats)):
        chosen = (
            distilled_outputs[sc.scenario_id]
            if distilled_outputs is not None
            else sc.gold_aligned_output
        )
        chosen_res = evaluate_response(
            chosen,
            expected_intent=sc.expected_intent,
            expected_tool=sc.expected_tool,
            expected_requires_human=sc.expected_requires_human,
        )
        if not chosen_res.passed_all:
            raise ValueError(
                f"DPO chosen sample {sc.scenario_id} failed compliance check: {chosen_res.errors}"
            )

        rejected, rejected_res = generate_negative(
            gold_json=chosen,
            category=neg_cat,  # type: ignore[arg-type]
            expected_intent=sc.expected_intent,
            expected_tool=sc.expected_tool,
            expected_requires_human=sc.expected_requires_human,
            seed=seed + idx,
        )
        if rejected_res.passed_all:
            raise ValueError(f"DPO rejected sample {sc.scenario_id} passed unexpectedly.")

        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": minimal_prompt},
                    {"role": "user", "content": sc.user_query},
                ],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "metadata": {
                    "id": sc.scenario_id,
                    "expected_intent": sc.expected_intent,
                    "expected_urgency": sc.expected_urgency,
                    "expected_requires_human": sc.expected_requires_human,
                    "expected_tool": sc.expected_tool,
                    "negative_category": neg_cat,
                    "speech_style": sc.speech_style,
                },
            }
        )
    return rows


def _format_holdout_rows(scenarios: list[SynthesizedScenario]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        res = evaluate_response(
            sc.gold_aligned_output,
            expected_intent=sc.expected_intent,
            expected_tool=sc.expected_tool,
            expected_requires_human=sc.expected_requires_human,
        )
        if not res.passed_all:
            raise ValueError(
                f"Holdout sample {sc.scenario_id} failed compliance check: {res.errors}"
            )
        rows.append(
            {
                "id": sc.scenario_id,
                "user_query": sc.user_query,
                "expected_intent": sc.expected_intent,
                "expected_urgency": sc.expected_urgency,
                "expected_requires_human": sc.expected_requires_human,
                "expected_tool": sc.expected_tool,
                "speech_style": sc.speech_style,
                "gold_aligned_output": sc.gold_aligned_output,
            }
        )
    return rows


def synthesize_all(
    mode: str = "procedural",
    model: str = "gpt-4o-mini",
    sft_count: int = 800,
    dpo_count: int = 800,
    holdout_count: int = 80,
    concurrency: int = 10,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate SFT, DPO, and holdout JSONL datasets with strict context stripping."""
    out_dir = output_dir or (PROJECT_ROOT / "data")
    num_intents = len(ALLOWED_INTENTS)
    for name, total in (("sft_count", sft_count), ("dpo_count", dpo_count), ("holdout_count", holdout_count)):
        if total % num_intents != 0:
            raise ValueError(f"{name}={total} must be divisible by {num_intents} intents.")

    minimal_prompt = _load_prompt(MINIMAL_PROMPT_PATH)
    bloated_prompt = _load_prompt(BLOATED_PROMPT_PATH)

    forbidden = _load_golden_queries()

    # 1. Generate stratified holdout set first (disjoint holdout pool)
    holdout_scenarios = generate_stratified_scenarios(
        per_intent=holdout_count // num_intents,
        split="holdout",
        seed=seed + 1000,
        forbidden_queries=forbidden,
        start_offset=0,
    )
    forbidden.update(s.user_query.strip() for s in holdout_scenarios)

    # 2. Generate stratified SFT training scenarios
    sft_scenarios = generate_stratified_scenarios(
        per_intent=sft_count // num_intents,
        split="train",
        seed=seed,
        forbidden_queries=forbidden,
        start_offset=0,
    )
    forbidden.update(s.user_query.strip() for s in sft_scenarios)

    # 3. Generate stratified DPO training scenarios (zero overlap with SFT & holdout)
    dpo_scenarios = generate_stratified_scenarios(
        per_intent=dpo_count // num_intents,
        split="train",
        seed=seed + 2000,
        forbidden_queries=forbidden,
        start_offset=5000,
    )

    sft_distilled: dict[str, str] | None = None
    dpo_distilled: dict[str, str] | None = None
    if mode == "api":
        sft_distilled = asyncio.run(
            _run_api_distillation(
                scenarios=sft_scenarios,
                model=model,
                concurrency=concurrency,
                cache_name="sft_teacher_cache",
                bloated_prompt=bloated_prompt,
            )
        )
        dpo_distilled = asyncio.run(
            _run_api_distillation(
                scenarios=dpo_scenarios,
                model=model,
                concurrency=concurrency,
                cache_name="dpo_teacher_cache",
                bloated_prompt=bloated_prompt,
            )
        )
    elif mode != "procedural":
        raise ValueError(f"Unsupported mode: {mode}")

    sft_rows = _format_sft_rows(sft_scenarios, minimal_prompt, sft_distilled)
    dpo_rows = _format_dpo_rows(dpo_scenarios, minimal_prompt, seed, dpo_distilled)
    holdout_rows = _format_holdout_rows(holdout_scenarios)

    sft_path = out_dir / "sft_train.jsonl"
    dpo_path = out_dir / "dpo_train.jsonl"
    holdout_path = out_dir / "test_holdout.jsonl"

    _write_jsonl(sft_path, sft_rows)
    _write_jsonl(dpo_path, dpo_rows)
    _write_jsonl(holdout_path, holdout_rows)

    return {
        "sft": sft_path,
        "dpo": dpo_path,
        "holdout": holdout_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WeightBake context-distilled SFT & DPO dataset synthesizer."
    )
    parser.add_argument(
        "--mode",
        choices=["procedural", "api"],
        default="procedural",
        help="Synthesis mode: procedural (deterministic fallback) or api (Teacher LLM).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Teacher LLM model ID when --mode api is used.",
    )
    parser.add_argument("--sft-count", type=int, default=800)
    parser.add_argument("--dpo-count", type=int, default=800)
    parser.add_argument("--holdout-count", type=int, default=80)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    console = Console()
    console.print(
        f"[cyan]Running WeightBake synthesizer in [bold]{args.mode}[/bold] mode...[/cyan]"
    )
    paths = synthesize_all(
        mode=args.mode,
        model=args.model,
        sft_count=args.sft_count,
        dpo_count=args.dpo_count,
        holdout_count=args.holdout_count,
        concurrency=args.concurrency,
        seed=args.seed,
    )
    for key, path in paths.items():
        console.print(f"[green]Wrote {key} dataset:[/green] {path}")


if __name__ == "__main__":
    main()
