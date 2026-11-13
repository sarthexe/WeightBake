"""Hard-negative perturbation engine for WeightBake DPO preference pairs."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from typing import Any, Literal

from src.eval.evaluator import ComplianceResult, evaluate_response

NegativeCategory = Literal[
    "markdown_fenced",
    "sycophantic_spoken",
    "word_count_overflow",
    "tts_unsafe_chars",
    "schema_drift",
    "routing_misclassification",
]

NEGATIVE_CATEGORIES: tuple[NegativeCategory, ...] = (
    "markdown_fenced",
    "sycophantic_spoken",
    "word_count_overflow",
    "tts_unsafe_chars",
    "schema_drift",
    "routing_misclassification",
)

_SYCOPHANCY_PREFIXES: tuple[str, ...] = (
    "Certainly, I am happy to help.",
    "Of course, I'm sorry about this issue.",
    "I understand your frustration and I apologize.",
    "Absolutely, I would be glad to help.",
    "Thank you for reaching out about this.",
)

_VERBOSE_TAILS: tuple[str, ...] = (
    "Please remain on the line while our automated enterprise gateway coordinates across internal systems and logs every diagnostic step carefully for your complete visibility today.",
    "Our support platform is currently processing your details through multiple verification layers and will continue monitoring the entire workflow until every single record has been confirmed.",
    "We appreciate your patience while the backend operations queue completes its full verification cycle and updates your workspace status across all regional clusters right away.",
)


def _perturb_markdown_fenced(gold_json: str, rng: random.Random) -> str:
    pretty = json.dumps(json.loads(gold_json), indent=2)
    style = rng.randrange(3)
    if style == 0:
        return f"```json\n{pretty}\n```"
    if style == 1:
        return f"Sure! Here is the JSON response:\n```json\n{pretty}\n```"
    return f"```\n{gold_json}\n```"


def _perturb_sycophantic_spoken(gold_json: str, rng: random.Random) -> str:
    data = json.loads(gold_json)
    prefix = rng.choice(_SYCOPHANCY_PREFIXES)
    original = data["spoken_response"].strip()
    # Keep total word count <= 22 if possible so failure specifically targets sycophancy
    combined_words = f"{prefix} {original}".split()
    if len(combined_words) > 22:
        combined_words = combined_words[:21]
        data["spoken_response"] = " ".join(combined_words).rstrip(",;") + "."
    else:
        data["spoken_response"] = f"{prefix} {original}"
    return json.dumps(data, separators=(",", ":"))


def _perturb_word_count_overflow(gold_json: str, rng: random.Random) -> str:
    data = json.loads(gold_json)
    original = data["spoken_response"].rstrip(".!?")
    tail = rng.choice(_VERBOSE_TAILS)
    words = f"{original} right now. {tail}".split()
    # Guarantee 26-40 words
    target_len = rng.randint(26, 38)
    if len(words) < target_len:
        words.extend(["please", "remain", "available", "for", "confirmation"] * 3)
    words = words[:target_len]
    data["spoken_response"] = " ".join(words).rstrip(",;") + "."
    return json.dumps(data, separators=(",", ":"))


def _perturb_tts_unsafe_chars(gold_json: str, rng: random.Random) -> str:
    data = json.loads(gold_json)
    base_words = data["spoken_response"].rstrip(".!?").split()[:14]
    original = " ".join(base_words)
    style = rng.randrange(4)
    if style == 0:
        data["spoken_response"] = f"*Status*: {original} right now."
    elif style == 1:
        data["spoken_response"] = f"{original} (ticket #404 / live)."
    elif style == 2:
        data["spoken_response"] = f"Update: {original} via support@gateway."
    else:
        data["spoken_response"] = f"[{original}] please hold; processing."
    return json.dumps(data, separators=(",", ":"))


def _perturb_schema_drift(gold_json: str, rng: random.Random) -> str:
    data: dict[str, Any] = json.loads(gold_json)
    style = rng.randrange(3)
    if style == 0:
        data["confidence"] = 0.98
    elif style == 1:
        data.pop("internal_reasoning", None)
    else:
        tool_name = data["tool_call"]["name"]
        args = deepcopy(data["tool_call"]["arguments"])
        if tool_name == "none":
            data["tool_call"]["arguments"] = {"reason": "no_action"}
        elif tool_name == "lookup_invoice":
            data["tool_call"]["arguments"] = {"account": args.get("account_id", "AC-1")}
        elif tool_name == "run_diagnostics":
            data["tool_call"]["arguments"] = {"service": args.get("service_name", "api")}
        elif tool_name == "lock_account_mfa":
            data["tool_call"]["arguments"] = {
                "account_id": args.get("account_id", "AC-1"),
                "lock_level": "freeze",
            }
        elif tool_name == "modify_subscription":
            data["tool_call"]["arguments"] = {"action": "extend", "target_tier": "pro"}
        elif tool_name == "track_shipment":
            data["tool_call"]["arguments"] = {"tracking_id": args.get("order_id", "ORD-1")}
        elif tool_name == "escalate_specialist":
            data["tool_call"]["arguments"] = {
                "department": args.get("department", "legal"),
                "priority_code": "P1",
            }
    return json.dumps(data, separators=(",", ":"))


def _perturb_routing_misclassification(
    gold_json: str,
    expected_intent: str | None,
    expected_tool: str | None,
    expected_requires_human: bool | None,
    rng: random.Random,
) -> str:
    data: dict[str, Any] = json.loads(gold_json)
    actual_intent = expected_intent or data["intent"]
    actual_tool = expected_tool or data["tool_call"]["name"]
    actual_human = (
        expected_requires_human
        if expected_requires_human is not None
        else data["requires_human"]
    )

    alt_intents = [
        i
        for i in (
            "billing_dispute",
            "technical_outage",
            "account_security",
            "subscription_change",
            "order_logistics",
            "compliance_legal",
            "general_inquiry",
            "adversarial_out_of_scope",
        )
        if i != actual_intent
    ]

    style = rng.randrange(3)
    if style == 0:
        data["intent"] = rng.choice(alt_intents)
    elif style == 1:
        if actual_tool != "none":
            data["tool_call"] = {"name": "none", "arguments": {}}
        else:
            data["tool_call"] = {
                "name": "escalate_specialist",
                "arguments": {"department": "engineering_oncall", "priority_code": 2},
            }
    else:
        data["requires_human"] = not actual_human
        if actual_intent in ("technical_outage", "account_security", "compliance_legal"):
            data["requires_human"] = False
            data["intent"] = "general_inquiry" if actual_human is False else actual_intent

    return json.dumps(data, separators=(",", ":"))


def generate_negative(
    gold_json: str,
    category: NegativeCategory,
    expected_intent: str | None = None,
    expected_tool: str | None = None,
    expected_requires_human: bool | None = None,
    seed: int = 42,
) -> tuple[str, ComplianceResult]:
    """Generate a single hard-negative rejected response and verify it fails evaluation."""
    rng = random.Random(seed)

    if category == "markdown_fenced":
        flawed = _perturb_markdown_fenced(gold_json, rng)
    elif category == "sycophantic_spoken":
        flawed = _perturb_sycophantic_spoken(gold_json, rng)
    elif category == "word_count_overflow":
        flawed = _perturb_word_count_overflow(gold_json, rng)
    elif category == "tts_unsafe_chars":
        flawed = _perturb_tts_unsafe_chars(gold_json, rng)
    elif category == "schema_drift":
        flawed = _perturb_schema_drift(gold_json, rng)
    elif category == "routing_misclassification":
        flawed = _perturb_routing_misclassification(
            gold_json,
            expected_intent=expected_intent,
            expected_tool=expected_tool,
            expected_requires_human=expected_requires_human,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown negative category: {category}")

    result = evaluate_response(
        flawed,
        expected_intent=expected_intent,
        expected_tool=expected_tool,
        expected_requires_human=expected_requires_human,
    )
    if result.passed_all:
        raise RuntimeError(
            f"Hard negative ({category}) unexpectedly passed evaluate_response: {flawed}"
        )
    return flawed, result
