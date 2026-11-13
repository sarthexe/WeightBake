"""Comprehensive unit tests for WeightBake deterministic evaluator and Phase 1 artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.eval.evaluator import (
    BANNED_SYCOPHANCY_PHRASES,
    FORBIDDEN_TTS_CHARS,
    evaluate_response,
)
from src.eval.run_baseline_check import (
    MIN_BLOATED_TOKENS,
    count_prompt_tokens,
    load_golden_dataset,
)


def _valid_payload() -> dict:
    return {
        "intent": "billing_dispute",
        "urgency": "high",
        "requires_human": False,
        "tool_call": {
            "name": "lookup_invoice",
            "arguments": {"account_id": "AC-8841", "charge_amount": 499.5},
        },
        "internal_reasoning": "Duplicate charge reported on AC-8841; run lookup.",
        "spoken_response": "I am checking account AC-8841 now to verify the duplicate charge.",
    }


def test_compliant_output_passes_all_checks() -> None:
    raw = json.dumps(_valid_payload())
    res = evaluate_response(
        raw,
        expected_intent="billing_dispute",
        expected_tool="lookup_invoice",
        expected_requires_human=False,
    )
    assert res.passed_all is True
    assert res.score == pytest.approx(1.0)
    assert all(res.checks.values())
    assert res.errors == []


def test_markdown_wrapper_fails() -> None:
    raw = f"```json\n{json.dumps(_valid_payload())}\n```"
    res = evaluate_response(raw)
    assert res.passed_all is False
    assert res.checks["no_markdown_wrapper"] is False
    assert res.checks["valid_json"] is False


def test_extra_top_level_json_key_fails_strict_schema() -> None:
    payload = _valid_payload()
    payload["unexpected_extra"] = "bad"
    res = evaluate_response(
        json.dumps(payload),
        expected_intent="billing_dispute",
        expected_tool="lookup_invoice",
        expected_requires_human=False,
    )
    assert res.passed_all is False
    assert res.checks["valid_json"] is True
    assert res.checks["strict_schema"] is False
    assert res.checks["routing_accuracy"] is True


def test_missing_top_level_key_fails_strict_schema() -> None:
    payload = _valid_payload()
    del payload["internal_reasoning"]
    res = evaluate_response(json.dumps(payload))
    assert res.passed_all is False
    assert res.checks["strict_schema"] is False


@pytest.mark.parametrize(
    "tool_name,bad_args",
    [
        ("lookup_invoice", {"account_id": "AC-1"}),  # missing charge_amount
        ("lookup_invoice", {"account_id": "AC-1", "charge_amount": "499.5"}),  # str not float
        ("lookup_invoice", {"account_id": "AC-1", "charge_amount": 10.0, "extra": 1}),
        ("run_diagnostics", {"service": "api", "error_code": "503"}),  # wrong key name
        ("lock_account_mfa", {"account_id": "AC-1", "lock_level": "freeze"}),  # bad enum
        ("modify_subscription", {"action": "extend", "target_tier": "pro"}),  # bad action
        ("track_shipment", {"order_id": 12345}),  # int instead of str
        ("escalate_specialist", {"department": "legal", "priority_code": "1"}),  # str priority
        ("escalate_specialist", {"department": "legal", "priority_code": True}),  # bool priority
        ("escalate_specialist", {"department": "finance", "priority_code": 1}),  # bad dept
        ("none", {"reason": "extra"}),  # non-empty args for none
    ],
)
def test_invalid_tool_arguments_fail_strict_schema(
    tool_name: str, bad_args: dict
) -> None:
    payload = _valid_payload()
    payload["tool_call"] = {"name": tool_name, "arguments": bad_args}
    res = evaluate_response(json.dumps(payload))
    assert res.passed_all is False
    assert res.checks["valid_json"] is True
    assert res.checks["strict_schema"] is False


def test_25_word_spoken_response_fails_word_count() -> None:
    payload = _valid_payload()
    payload["spoken_response"] = " ".join(["word"] * 25)
    res = evaluate_response(json.dumps(payload))
    assert res.passed_all is False
    assert res.checks["spoken_word_count_5_to_22"] is False


def test_4_word_spoken_response_fails_word_count() -> None:
    payload = _valid_payload()
    payload["spoken_response"] = "Checking your invoice now."
    res = evaluate_response(json.dumps(payload))
    assert res.passed_all is False
    assert res.checks["spoken_word_count_5_to_22"] is False


def test_spoken_word_count_boundaries_5_and_22_pass() -> None:
    for n in (5, 22):
        payload = _valid_payload()
        payload["spoken_response"] = " ".join(["word"] * n)
        res = evaluate_response(json.dumps(payload))
        assert res.checks["spoken_word_count_5_to_22"] is True


@pytest.mark.parametrize("phrase", BANNED_SYCOPHANCY_PHRASES)
def test_banned_sycophancy_phrases_fail(phrase: str) -> None:
    payload = _valid_payload()
    payload["spoken_response"] = f"{phrase.capitalize()} we are checking your billing invoice now."
    res = evaluate_response(json.dumps(payload))
    assert res.passed_all is False
    assert res.checks["zero_sycophancy"] is False


def test_word_boundary_prevents_false_positive_on_ensure() -> None:
    payload = _valid_payload()
    payload["spoken_response"] = "We will ensure your billing invoice is verified immediately."
    res = evaluate_response(json.dumps(payload))
    assert res.checks["zero_sycophancy"] is True
    assert res.passed_all is True


@pytest.mark.parametrize("bad_char", [":", "*", "#", "`", "(", ")", "/", "@", "~"])
def test_forbidden_tts_punctuation_fails(bad_char: str) -> None:
    assert bad_char in FORBIDDEN_TTS_CHARS
    payload = _valid_payload()
    payload["spoken_response"] = f"Checking account AC-8841{bad_char} please hold for invoice lookup."
    res = evaluate_response(json.dumps(payload))
    assert res.passed_all is False
    assert res.checks["tts_safe_punctuation"] is False


def test_wrong_intent_fails_routing_accuracy() -> None:
    raw = json.dumps(_valid_payload())
    res = evaluate_response(
        raw,
        expected_intent="technical_outage",
        expected_tool="lookup_invoice",
        expected_requires_human=False,
    )
    assert res.passed_all is False
    assert res.checks["strict_schema"] is True
    assert res.checks["routing_accuracy"] is False


def test_wrong_tool_and_requires_human_fail_routing_accuracy() -> None:
    payload = deepcopy(_valid_payload())
    res = evaluate_response(
        json.dumps(payload),
        expected_intent="billing_dispute",
        expected_tool="escalate_specialist",
        expected_requires_human=True,
    )
    assert res.passed_all is False
    assert res.checks["routing_accuracy"] is False


def test_golden_eval_dataset_and_prompt_tokens() -> None:
    bloated_tokens, minimal_tokens, reduction_pct = count_prompt_tokens()
    assert bloated_tokens >= MIN_BLOATED_TOKENS
    assert minimal_tokens <= 15
    assert reduction_pct > 98.0

    rows = load_golden_dataset()
    assert len(rows) == 16

    intents = {row["expected_intent"] for row in rows}
    assert len(intents) == 8

    for row in rows:
        unaligned = evaluate_response(
            row["mock_unaligned_output"],
            expected_intent=row["expected_intent"],
            expected_tool=row["expected_tool"],
            expected_requires_human=row["expected_requires_human"],
        )
        assert unaligned.passed_all is False, f"{row['id']} unaligned unexpectedly passed"

        gold = evaluate_response(
            row["gold_aligned_output"],
            expected_intent=row["expected_intent"],
            expected_tool=row["expected_tool"],
            expected_requires_human=row["expected_requires_human"],
        )
        assert gold.passed_all is True, f"{row['id']} gold failed: {gold.errors}"
