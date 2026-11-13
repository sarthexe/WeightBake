"""Deterministic evaluation harness for WeightBake voice router outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

IntentType = Literal[
    "billing_dispute",
    "technical_outage",
    "account_security",
    "subscription_change",
    "order_logistics",
    "compliance_legal",
    "general_inquiry",
    "adversarial_out_of_scope",
]

UrgencyType = Literal["low", "medium", "high", "critical"]

ToolNameType = Literal[
    "lookup_invoice",
    "run_diagnostics",
    "lock_account_mfa",
    "modify_subscription",
    "track_shipment",
    "escalate_specialist",
    "none",
]

BANNED_SYCOPHANCY_PHRASES: tuple[str, ...] = (
    "certainly",
    "sure",
    "of course",
    "absolutely",
    "happy to help",
    "glad to help",
    "i understand your frustration",
    "i apologize",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "great question",
    "thank you for reaching out",
)

_SYCOPHANCY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (phrase, re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE))
    for phrase in BANNED_SYCOPHANCY_PHRASES
)

FORBIDDEN_TTS_CHARS: frozenset[str] = frozenset("*#`_[](){}:;/\\|<>@~")


class LookupInvoiceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: StrictStr
    charge_amount: StrictFloat | StrictInt


class RunDiagnosticsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_name: StrictStr
    error_code: StrictStr


class LockAccountMfaArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: StrictStr
    lock_level: Literal["soft_reset", "hard_freeze"]


class ModifySubscriptionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["upgrade", "downgrade", "pause", "cancel"]
    target_tier: StrictStr


class TrackShipmentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: StrictStr


class EscalateSpecialistArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    department: Literal["legal", "security", "billing_tier2", "engineering_oncall"]
    priority_code: Literal[1, 2, 3]

    @field_validator("priority_code", mode="before")
    @classmethod
    def reject_bool_priority_code(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("priority_code must be an integer (1, 2, or 3), not bool")
        return v


class NoneArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "lookup_invoice": LookupInvoiceArgs,
    "run_diagnostics": RunDiagnosticsArgs,
    "lock_account_mfa": LockAccountMfaArgs,
    "modify_subscription": ModifySubscriptionArgs,
    "track_shipment": TrackShipmentArgs,
    "escalate_specialist": EscalateSpecialistArgs,
    "none": NoneArgs,
}


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ToolNameType
    arguments: dict[str, Any]

    @model_validator(mode="after")
    def validate_tool_arguments(self) -> ToolCall:
        arg_model = TOOL_ARG_MODELS[self.name]
        validated = arg_model.model_validate(self.arguments)
        self.arguments = validated.model_dump()
        return self


class VoiceRouterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: IntentType
    urgency: UrgencyType
    requires_human: StrictBool
    tool_call: ToolCall
    internal_reasoning: StrictStr = Field(..., min_length=1)
    spoken_response: StrictStr = Field(..., min_length=1)


class ComplianceResult(BaseModel):
    passed_all: bool
    score: float
    checks: dict[str, bool]
    errors: list[str]


def evaluate_response(
    raw_output: str,
    expected_intent: str | None = None,
    expected_tool: str | None = None,
    expected_requires_human: bool | None = None,
) -> ComplianceResult:
    """Evaluate a raw model output against all 7 deterministic voice-router checks."""
    errors: list[str] = []
    checks: dict[str, bool] = {
        "no_markdown_wrapper": False,
        "valid_json": False,
        "strict_schema": False,
        "spoken_word_count_5_to_22": False,
        "zero_sycophancy": False,
        "tts_safe_punctuation": False,
        "routing_accuracy": False,
    }

    stripped = raw_output.strip() if isinstance(raw_output, str) else ""

    # 1. no_markdown_wrapper
    has_wrapper = (
        stripped.startswith("```")
        or stripped.endswith("```")
        or "```json" in stripped.lower()
    )
    checks["no_markdown_wrapper"] = not has_wrapper
    if has_wrapper:
        errors.append("no_markdown_wrapper: Output contains markdown code fence wrapper.")

    # 2. valid_json
    parsed: dict[str, Any] | None = None
    if stripped:
        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, dict):
                parsed = loaded
                checks["valid_json"] = True
            else:
                errors.append(
                    f"valid_json: Expected top-level JSON object, got {type(loaded).__name__}."
                )
        except json.JSONDecodeError as exc:
            errors.append(f"valid_json: JSON parse error ({exc.msg} at pos {exc.pos}).")
    else:
        errors.append("valid_json: Output is empty.")

    # 3. strict_schema
    validated_model: VoiceRouterOutput | None = None
    if parsed is not None:
        try:
            validated_model = VoiceRouterOutput.model_validate(parsed)
            checks["strict_schema"] = True
        except ValidationError as exc:
            first_err = exc.errors()[0]
            loc = ".".join(str(part) for part in first_err.get("loc", ()))
            msg = first_err.get("msg", "validation error")
            errors.append(f"strict_schema: {loc}: {msg}")

    # Extract spoken_response for checks 4-6 if available
    spoken_response: str | None = None
    if validated_model is not None:
        spoken_response = validated_model.spoken_response
    elif parsed is not None and isinstance(parsed.get("spoken_response"), str):
        spoken_response = parsed["spoken_response"]

    if spoken_response is not None:
        # 4. spoken_word_count_5_to_22
        words = spoken_response.strip().split()
        word_count = len(words)
        if 5 <= word_count <= 22:
            checks["spoken_word_count_5_to_22"] = True
        else:
            errors.append(
                f"spoken_word_count_5_to_22: Word count is {word_count} (must be 5..22)."
            )

        # 5. zero_sycophancy
        matched_phrases = [
            phrase
            for phrase, pattern in _SYCOPHANCY_PATTERNS
            if pattern.search(spoken_response)
        ]
        if not matched_phrases:
            checks["zero_sycophancy"] = True
        else:
            errors.append(
                f"zero_sycophancy: Found banned filler phrase(s): {', '.join(matched_phrases)}."
            )

        # 6. tts_safe_punctuation
        bad_chars = sorted({ch for ch in spoken_response if ch in FORBIDDEN_TTS_CHARS})
        if not bad_chars:
            checks["tts_safe_punctuation"] = True
        else:
            errors.append(
                f"tts_safe_punctuation: Forbidden TTS character(s) found: {''.join(bad_chars)}."
            )
    else:
        errors.append("spoken_response: Missing or invalid string; cannot evaluate TTS rules.")

    # 7. routing_accuracy
    if parsed is not None:
        actual_intent = parsed.get("intent")
        tool_obj = parsed.get("tool_call")
        actual_tool = tool_obj.get("name") if isinstance(tool_obj, dict) else None
        actual_requires_human = parsed.get("requires_human")

        routing_ok = True
        if expected_intent is not None and actual_intent != expected_intent:
            routing_ok = False
            errors.append(
                f"routing_accuracy: Expected intent '{expected_intent}', got '{actual_intent}'."
            )
        if expected_tool is not None and actual_tool != expected_tool:
            routing_ok = False
            errors.append(
                f"routing_accuracy: Expected tool '{expected_tool}', got '{actual_tool}'."
            )
        if expected_requires_human is not None and (
            not isinstance(actual_requires_human, bool)
            or actual_requires_human is not expected_requires_human
        ):
            routing_ok = False
            errors.append(
                f"routing_accuracy: Expected requires_human={expected_requires_human}, "
                f"got {actual_requires_human!r}."
            )
        checks["routing_accuracy"] = routing_ok
    else:
        errors.append("routing_accuracy: Cannot verify routing without valid JSON object.")

    passed_all = all(checks.values())
    score = sum(1 for passed in checks.values() if passed) / len(checks)
    return ComplianceResult(
        passed_all=passed_all,
        score=score,
        checks=checks,
        errors=errors,
    )
