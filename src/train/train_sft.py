"""Production SFT training script with assistant-only loss masking & dry-run audit."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from src.train.config import SFTExperimentConfig
from src.train.formatting import (
    ASSISTANT_HEADER,
    IM_END,
    build_masked_sft_labels,
    format_chatml_messages,
    get_dry_run_chatml_tokenizer,
    load_and_split_sft_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOATED_PROMPT_PATH = PROJECT_ROOT / "prompts" / "bloated_system_prompt.txt"


class SFTDryRunSummary(BaseModel):
    train_samples: int
    val_samples: int
    bloated_prompt_leakage_count: int
    total_tokens: int
    masked_tokens: int
    supervised_tokens: int
    masked_pct: float
    supervised_pct: float
    mean_tokens_per_sample: float
    max_tokens_per_sample: int
    effective_batch_size: int
    total_optimization_steps: int


def run_dry_run(
    cfg: SFTExperimentConfig,
    max_steps: int | None = None,
    console: Console | None = None,
) -> SFTDryRunSummary:
    """Validate config, dataset split, context stripping, ChatML formatting, and -100 masking."""
    out_console = console or Console()

    train_path = Path(cfg.data.train_file)
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path

    ds_dict = load_and_split_sft_dataset(
        train_file=train_path,
        val_split_ratio=cfg.data.val_split_ratio,
        seed=cfg.training.seed,
    )
    train_ds = ds_dict["train"]
    val_ds = ds_dict["validation"]

    bloated_snippet = (
        BLOATED_PROMPT_PATH.read_text(encoding="utf-8").strip()[:80]
        if BLOATED_PROMPT_PATH.exists()
        else "SYSTEM IDENTITY AND RUNTIME ARCHITECTURE"
    )

    enc = get_dry_run_chatml_tokenizer()
    assistant_header_ids = enc.encode(ASSISTANT_HEADER, allowed_special="all")
    im_end_id = enc.encode(IM_END, allowed_special="all")[0]

    total_tokens = 0
    masked_tokens = 0
    supervised_tokens = 0
    max_tokens_seen = 0
    leakage_count = 0

    for split_name in ("train", "validation"):
        for row in ds_dict[split_name]:
            messages = row["messages"]
            sys_content = messages[0]["content"]
            if bloated_snippet in sys_content:
                leakage_count += 1

            rendered = format_chatml_messages(messages, add_generation_prompt=False)
            input_ids = enc.encode(rendered, allowed_special="all")
            if len(input_ids) > cfg.max_seq_length:
                raise ValueError(
                    f"Formatted sample exceeds max_seq_length={cfg.max_seq_length}: {len(input_ids)} tokens."
                )

            labels = build_masked_sft_labels(
                input_ids=input_ids,
                assistant_header_ids=assistant_header_ids,
                im_end_id=im_end_id,
            )

            sample_masked = sum(1 for lbl in labels if lbl == -100)
            sample_supervised = sum(1 for lbl in labels if lbl != -100)

            total_tokens += len(input_ids)
            masked_tokens += sample_masked
            supervised_tokens += sample_supervised
            if len(input_ids) > max_tokens_seen:
                max_tokens_seen = len(input_ids)

    if leakage_count > 0:
        raise ValueError(
            f"Detected bloated system prompt leakage in {leakage_count} SFT samples!"
        )

    total_samples = len(train_ds) + len(val_ds)
    effective_batch = (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
    )
    computed_steps = (
        math.ceil(len(train_ds) / effective_batch) * cfg.training.num_train_epochs
    )
    total_steps = max_steps if max_steps is not None else computed_steps

    masked_pct = (masked_tokens / total_tokens) * 100.0 if total_tokens else 0.0
    supervised_pct = (supervised_tokens / total_tokens) * 100.0 if total_tokens else 0.0
    mean_tokens = total_tokens / total_samples if total_samples else 0.0

    summary = SFTDryRunSummary(
        train_samples=len(train_ds),
        val_samples=len(val_ds),
        bloated_prompt_leakage_count=leakage_count,
        total_tokens=total_tokens,
        masked_tokens=masked_tokens,
        supervised_tokens=supervised_tokens,
        masked_pct=masked_pct,
        supervised_pct=supervised_pct,
        mean_tokens_per_sample=mean_tokens,
        max_tokens_per_sample=max_tokens_seen,
        effective_batch_size=effective_batch,
        total_optimization_steps=total_steps,
    )

    table = Table(title="WeightBake SFT Dry-Run & Assistant-Only Masking Audit")
    table.add_column("Metric / Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold")

    table.add_row("Model (Primary / Fallback)", f"{cfg.model_name} / {cfg.fallback_model_name}")
    table.add_row("LoRA Config", f"r={cfg.lora.r}, alpha={cfg.lora.lora_alpha}, dropout={cfg.lora.lora_dropout}, modules={len(cfg.lora.target_modules)}")
    table.add_row("Dataset Split (Train / Val)", f"{summary.train_samples} / {summary.val_samples}")
    table.add_row("Bloated Prompt Leakage", f"[green]{summary.bloated_prompt_leakage_count} (100% Minimal Context)[/green]")
    table.add_row("Tokens / Sample (Mean / Max)", f"{summary.mean_tokens_per_sample:.1f} / {summary.max_tokens_per_sample} (limit {cfg.max_seq_length})")
    table.add_row("Masked Prompt Tokens (-100)", f"{summary.masked_tokens:,} ({summary.masked_pct:.1f}%)")
    table.add_row("Supervised Assistant Tokens", f"[green]{summary.supervised_tokens:,} ({summary.supervised_pct:.1f}%)[/green]")
    table.add_row("Effective Batch Size", f"{cfg.training.per_device_train_batch_size} x {cfg.training.gradient_accumulation_steps} = {summary.effective_batch_size}")
    table.add_row("Total Optimization Steps", f"{summary.total_optimization_steps} ({cfg.training.num_train_epochs} epochs)")
    table.add_row("Learning Rate / Optimizer", f"{cfg.training.learning_rate} ({cfg.training.lr_scheduler_type}) / {cfg.training.optim}")
    out_console.print(table)

    return summary


def run_training(
    cfg: SFTExperimentConfig,
    backend: str = "unsloth",
    max_steps: int | None = None,
) -> None:
    """Execute full GPU QLoRA SFT training with assistant-only loss masking."""
    train_path = Path(cfg.data.train_file)
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path

    ds_dict = load_and_split_sft_dataset(
        train_file=train_path,
        val_split_ratio=cfg.data.val_split_ratio,
        seed=cfg.training.seed,
    )

    def _add_text_field(batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": [
                format_chatml_messages(msgs, add_generation_prompt=False)
                for msgs in batch["messages"]
            ]
        }

    train_ds = ds_dict["train"].map(_add_text_field, batched=True)
    val_ds = ds_dict["validation"].map(_add_text_field, batched=True)

    output_dir = Path(cfg.training.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "unsloth":
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=None,
            load_in_4bit=cfg.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.r,
            target_modules=cfg.lora.target_modules,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=cfg.lora.random_state,
        )

        sft_args = SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
            num_train_epochs=cfg.training.num_train_epochs,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            learning_rate=cfg.training.learning_rate,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            warmup_ratio=cfg.training.warmup_ratio,
            weight_decay=cfg.training.weight_decay,
            optim=cfg.training.optim,
            logging_steps=cfg.training.logging_steps,
            save_strategy=cfg.training.save_strategy,
            eval_strategy="epoch",
            seed=cfg.training.seed,
            max_steps=max_steps if max_steps is not None else -1,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            args=sft_args,
        )
        if cfg.training.train_on_responses_only:
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
        trainer.train()
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    elif backend == "trl":
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        tokenizer = AutoTokenizer.from_pretrained(cfg.fallback_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(cfg.fallback_model_name)
        peft_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            target_modules=cfg.lora.target_modules,
            task_type="CAUSAL_LM",
        )

        assistant_header_ids = tokenizer.encode(
            ASSISTANT_HEADER, add_special_tokens=False
        )
        im_end_ids = tokenizer.encode(IM_END, add_special_tokens=False)
        im_end_id = im_end_ids[0] if len(im_end_ids) == 1 else None

        def _tokenize_and_mask(row: dict[str, Any]) -> dict[str, Any]:
            enc = tokenizer(
                row["text"],
                truncation=True,
                max_length=cfg.max_seq_length,
                add_special_tokens=False,
            )
            labels = build_masked_sft_labels(
                enc["input_ids"],
                assistant_header_ids=assistant_header_ids,
                im_end_id=im_end_id,
            )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }

        tokenized_train = train_ds.map(_tokenize_and_mask, remove_columns=train_ds.column_names)
        tokenized_val = val_ds.map(_tokenize_and_mask, remove_columns=val_ds.column_names)

        sft_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=cfg.training.num_train_epochs,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            learning_rate=cfg.training.learning_rate,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            warmup_ratio=cfg.training.warmup_ratio,
            weight_decay=cfg.training.weight_decay,
            logging_steps=cfg.training.logging_steps,
            save_strategy=cfg.training.save_strategy,
            eval_strategy="epoch",
            seed=cfg.training.seed,
            max_steps=max_steps if max_steps is not None else -1,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            peft_config=peft_cfg,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            args=sft_args,
        )
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WeightBake SFT training pipeline with assistant-only loss masking."
    )
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument("--backend", choices=["unsloth", "trl"], default="unsloth")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = SFTExperimentConfig.from_yaml(args.config)
    if args.dry_run:
        run_dry_run(cfg=cfg, max_steps=args.max_steps)
    else:
        run_training(cfg=cfg, backend=args.backend, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
