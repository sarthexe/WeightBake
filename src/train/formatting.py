"""ChatML formatting, assistant-only loss masking, and SFT dataset splitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tiktoken
from datasets import Dataset, DatasetDict

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASSISTANT_HEADER = f"{IM_START}assistant\n"


def format_chatml_messages(
    messages: list[dict[str, str]], add_generation_prompt: bool = False
) -> str:
    """Format conversational messages into exact Qwen2.5 / ChatML format."""
    if not messages:
        raise ValueError("messages list must not be empty.")

    parts: list[str] = []
    for idx, msg in enumerate(messages):
        if "role" not in msg or "content" not in msg:
            raise ValueError(f"Message at index {idx} missing 'role' or 'content'.")
        role = msg["role"].strip()
        content = msg["content"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported ChatML role '{role}' at index {idx}.")
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")

    if add_generation_prompt:
        parts.append(ASSISTANT_HEADER)

    return "".join(parts)


def build_masked_sft_labels(
    input_ids: list[int],
    assistant_header_ids: list[int],
    im_end_id: int | None = None,
) -> list[int]:
    """Mask prompt tokens (before & including assistant header) with -100."""
    if not assistant_header_ids:
        raise ValueError("assistant_header_ids must not be empty.")

    header_len = len(assistant_header_ids)
    match_start: int | None = None

    # Find the last occurrence of assistant_header_ids in input_ids
    for i in range(len(input_ids) - header_len, -1, -1):
        if input_ids[i : i + header_len] == assistant_header_ids:
            match_start = i
            break

    if match_start is None:
        raise ValueError(
            "Assistant header sequence was not found in input_ids; "
            "cannot safely compute assistant-only SFT loss labels."
        )

    response_start = match_start + header_len
    labels: list[int] = [-100] * len(input_ids)

    for idx in range(response_start, len(input_ids)):
        tok = input_ids[idx]
        labels[idx] = tok
        if im_end_id is not None and tok == im_end_id:
            # Keep closing <|im_end|> supervised and mask any trailing tokens/padding
            break

    return labels


def load_and_split_sft_dataset(
    train_file: str | Path,
    val_split_ratio: float = 0.05,
    seed: int = 3407,
) -> DatasetDict:
    """Load SFT JSONL file and deterministically split into train and validation."""
    path = Path(train_file)
    if not path.exists():
        raise FileNotFoundError(f"SFT training file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    if not records:
        raise ValueError(f"SFT training file is empty: {path}")

    full_ds = Dataset.from_list(records)
    split = full_ds.train_test_split(
        test_size=val_split_ratio,
        seed=seed,
        shuffle=True,
    )
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def get_dry_run_chatml_tokenizer() -> tiktoken.Encoding:
    """Build a tiktoken ChatML encoding with <|im_start|> and <|im_end|> special tokens."""
    base = tiktoken.get_encoding("cl100k_base")
    im_start_id = base.n_vocab
    im_end_id = base.n_vocab + 1
    special_tokens = {
        **base._special_tokens,
        IM_START: im_start_id,
        IM_END: im_end_id,
    }
    return tiktoken.Encoding(
        name="cl100k_chatml",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special_tokens,
    )
