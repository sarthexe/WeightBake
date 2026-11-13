"""DPO dataset ChatML formatting, context-stripping checks, and Bradley-Terry loss utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict

from src.train.formatting import IM_END, format_chatml_messages

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINIMAL_PROMPT_PATH = PROJECT_ROOT / "prompts" / "minimal_system_prompt.txt"
BLOATED_PROMPT_PATH = PROJECT_ROOT / "prompts" / "bloated_system_prompt.txt"


def load_and_format_dpo_dataset(
    train_file: str | Path,
    val_split_ratio: float = 0.05,
    seed: int = 3407,
) -> DatasetDict:
    """Load DPO JSONL triplets, verify minimal prompt context stripping, and format as ChatML."""
    path = Path(train_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"DPO training file not found: {path}")

    minimal_prompt = MINIMAL_PROMPT_PATH.read_text(encoding="utf-8").strip()
    bloated_snippet = (
        BLOATED_PROMPT_PATH.read_text(encoding="utf-8").strip()[:80]
        if BLOATED_PROMPT_PATH.exists()
        else "SYSTEM IDENTITY AND RUNTIME ARCHITECTURE"
    )

    formatted_records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            row: dict[str, Any] = json.loads(stripped)
            prompt_msgs = row.get("prompt", [])
            chosen_msgs = row.get("chosen", [])
            rejected_msgs = row.get("rejected", [])

            if len(prompt_msgs) < 2 or len(chosen_msgs) != 1 or len(rejected_msgs) != 1:
                raise ValueError(f"Invalid DPO triplet structure at row {idx}.")

            sys_content = prompt_msgs[0].get("content", "").strip()
            if bloated_snippet in sys_content or sys_content != minimal_prompt:
                raise ValueError(
                    f"Context stripping violation at DPO row {idx}: "
                    f"expected minimal prompt, got {sys_content[:60]!r}."
                )

            prompt_str = format_chatml_messages(prompt_msgs, add_generation_prompt=True)
            chosen_str = f"{chosen_msgs[0]['content']}{IM_END}\n"
            rejected_str = f"{rejected_msgs[0]['content']}{IM_END}\n"

            formatted_records.append(
                {
                    "prompt": prompt_str,
                    "chosen": chosen_str,
                    "rejected": rejected_str,
                }
            )

    if not formatted_records:
        raise ValueError(f"DPO training file is empty: {path}")

    full_ds = Dataset.from_list(formatted_records)
    split = full_ds.train_test_split(
        test_size=val_split_ratio,
        seed=seed,
        shuffle=True,
    )
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def compute_dpo_loss_and_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute exact Bradley-Terry DPO sigmoid loss and TRL-compatible reward telemetry."""
    logits = beta * (
        (policy_chosen_logps - ref_chosen_logps)
        - (policy_rejected_logps - ref_rejected_logps)
    )
    loss = -F.logsigmoid(logits).mean()

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    reward_margin = chosen_rewards - rejected_rewards
    reward_accuracy = (reward_margin > 0).float().mean()

    return {
        "loss": loss,
        "chosen_rewards": chosen_rewards.mean(),
        "rejected_rewards": rejected_rewards.mean(),
        "reward_margin": reward_margin.mean(),
        "reward_accuracy": reward_accuracy,
    }
