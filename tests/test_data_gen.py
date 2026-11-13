"""Unit and integration tests for Phase 2 WeightBake context-distilled data engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import load_dataset

from src.data_gen.negative_miner import NEGATIVE_CATEGORIES, generate_negative
from src.data_gen.synthesizer import (
    BLOATED_PROMPT_PATH,
    MINIMAL_PROMPT_PATH,
    synthesize_all,
)
from src.data_gen.validate_datasets import DATA_DIR, validate_all_datasets


def _sample_gold() -> str:
    return json.dumps(
        {
            "intent": "account_security",
            "urgency": "critical",
            "requires_human": True,
            "tool_call": {
                "name": "lock_account_mfa",
                "arguments": {"account_id": "AC-2019", "lock_level": "hard_freeze"},
            },
            "internal_reasoning": "Active MFA breach on AC-2019; apply hard freeze.",
            "spoken_response": "Account AC-2019 is frozen immediately while our security team reviews access.",
        },
        separators=(",", ":"),
    )


@pytest.mark.parametrize(
    "category,expected_failed_check",
    [
        ("markdown_fenced", "no_markdown_wrapper"),
        ("sycophantic_spoken", "zero_sycophancy"),
        ("word_count_overflow", "spoken_word_count_5_to_22"),
        ("tts_unsafe_chars", "tts_safe_punctuation"),
        ("schema_drift", "strict_schema"),
        ("routing_misclassification", "routing_accuracy"),
    ],
)
def test_negative_miner_all_6_categories(
    category: str, expected_failed_check: str
) -> None:
    gold = _sample_gold()
    for seed in range(10):
        flawed, result = generate_negative(
            gold_json=gold,
            category=category,  # type: ignore[arg-type]
            expected_intent="account_security",
            expected_tool="lock_account_mfa",
            expected_requires_human=True,
            seed=seed,
        )
        assert isinstance(flawed, str) and flawed != gold
        assert result.passed_all is False
        assert result.checks[expected_failed_check] is False


def test_context_stripping_and_trl_format_in_temp_dir(tmp_path: Path) -> None:
    paths = synthesize_all(
        mode="procedural",
        sft_count=16,
        dpo_count=16,
        holdout_count=8,
        seed=123,
        output_dir=tmp_path,
    )
    minimal_prompt = MINIMAL_PROMPT_PATH.read_text(encoding="utf-8").strip()
    bloated_prompt = BLOATED_PROMPT_PATH.read_text(encoding="utf-8").strip()

    report = validate_all_datasets(data_dir=tmp_path)
    assert report.passed is True, f"Validation errors: {report.errors}"
    assert report.sft_count == 16
    assert report.dpo_count == 16
    assert report.holdout_count == 8
    assert report.query_leakage_count == 0
    assert report.context_stripped_count == 32

    # Verify Hugging Face datasets / TRL format compatibility
    sft_ds = load_dataset("json", data_files=str(paths["sft"]), split="train")
    assert "messages" in sft_ds.column_names
    for row in sft_ds:
        msgs = row["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[0]["content"] == minimal_prompt
        assert bloated_prompt[:40] not in msgs[0]["content"]

    dpo_ds = load_dataset("json", data_files=str(paths["dpo"]), split="train")
    assert {"prompt", "chosen", "rejected"}.issubset(set(dpo_ds.column_names))
    for row in dpo_ds:
        assert row["prompt"][0]["content"] == minimal_prompt
        assert bloated_prompt[:40] not in row["prompt"][0]["content"]
        assert row["chosen"][0]["role"] == "assistant"
        assert row["rejected"][0]["role"] == "assistant"


def test_generated_phase2_datasets_integrity() -> None:
    sft_path = DATA_DIR / "sft_train.jsonl"
    dpo_path = DATA_DIR / "dpo_train.jsonl"
    holdout_path = DATA_DIR / "test_holdout.jsonl"

    assert sft_path.exists(), "Run synthesizer to generate data/sft_train.jsonl"
    assert dpo_path.exists(), "Run synthesizer to generate data/dpo_train.jsonl"
    assert holdout_path.exists(), "Run synthesizer to generate data/test_holdout.jsonl"

    report = validate_all_datasets(data_dir=DATA_DIR)
    assert report.passed is True, f"Validation errors: {report.errors}"
    assert report.sft_count == 800
    assert report.dpo_count == 800
    assert report.holdout_count == 80
    assert report.sft_pass_count == 800
    assert report.dpo_chosen_pass_count == 800
    assert report.dpo_rejected_fail_count == 800
    assert report.holdout_pass_count == 80
    assert report.context_stripped_count == 1600
    assert report.query_leakage_count == 0
    assert set(report.negative_category_counts.keys()) == set(NEGATIVE_CATEGORIES)
