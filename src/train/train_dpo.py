"""Direct Preference Optimization (DPO) alignment trainer for WeightBake."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from src.train.config import DPOExperimentConfig
from src.train.dpo_utils import compute_dpo_loss_and_metrics, load_and_format_dpo_dataset
from src.train.formatting import get_dry_run_chatml_tokenizer
from src.train.merge_weights import merge_lora_to_16bit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DPODryRunSummary(BaseModel):
    train_samples: int
    val_samples: int
    mean_prompt_tokens: float
    max_prompt_tokens: int
    mean_chosen_tokens: float
    mean_rejected_tokens: float
    max_total_tokens: int
    effective_batch_size: int
    total_optimization_steps: int
    sanity_loss: float
    sanity_reward_margin: float
    sanity_reward_accuracy: float


def run_dpo_dry_run(
    cfg: DPOExperimentConfig,
    max_steps: int | None = None,
    console: Console | None = None,
) -> DPODryRunSummary:
    """Validate DPO config, dataset formatting, token budgets, and Bradley-Terry math."""
    out_console = console or Console()

    ds_dict = load_and_format_dpo_dataset(
        train_file=cfg.data.train_file,
        val_split_ratio=cfg.data.val_split_ratio,
        seed=cfg.dpo.seed,
    )
    train_ds = ds_dict["train"]
    val_ds = ds_dict["validation"]

    enc = get_dry_run_chatml_tokenizer()
    prompt_lens: list[int] = []
    chosen_lens: list[int] = []
    rejected_lens: list[int] = []
    max_total_tokens = 0

    for split_name in ("train", "validation"):
        for row in ds_dict[split_name]:
            p_len = len(enc.encode(row["prompt"], allowed_special="all"))
            c_len = len(enc.encode(row["chosen"], allowed_special="all"))
            r_len = len(enc.encode(row["rejected"], allowed_special="all"))

            if p_len > cfg.max_prompt_length:
                raise ValueError(
                    f"Prompt length {p_len} exceeds max_prompt_length={cfg.max_prompt_length}."
                )
            total_len = p_len + max(c_len, r_len)
            if total_len > cfg.max_seq_length:
                raise ValueError(
                    f"Triplet length {total_len} exceeds max_seq_length={cfg.max_seq_length}."
                )

            prompt_lens.append(p_len)
            chosen_lens.append(c_len)
            rejected_lens.append(r_len)
            if total_len > max_total_tokens:
                max_total_tokens = total_len

    # Numerical sanity check on synthetic log-probabilities
    policy_chosen = torch.tensor([-12.0, -10.5, -14.2, -9.8], dtype=torch.float32)
    policy_rejected = torch.tensor([-28.0, -22.5, -31.0, -24.0], dtype=torch.float32)
    ref_chosen = torch.tensor([-18.0, -15.0, -18.5, -14.5], dtype=torch.float32)
    ref_rejected = torch.tensor([-17.0, -15.5, -19.0, -15.0], dtype=torch.float32)

    metrics = compute_dpo_loss_and_metrics(
        policy_chosen_logps=policy_chosen,
        policy_rejected_logps=policy_rejected,
        ref_chosen_logps=ref_chosen,
        ref_rejected_logps=ref_rejected,
        beta=cfg.dpo.beta,
    )
    for name, tensor in metrics.items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise RuntimeError(f"Numerical instability detected in DPO metric '{name}'.")

    effective_batch = (
        cfg.dpo.per_device_train_batch_size * cfg.dpo.gradient_accumulation_steps
    )
    computed_steps = math.ceil(len(train_ds) / effective_batch) * cfg.dpo.num_train_epochs
    if max_steps is not None:
        total_steps = max_steps
    elif cfg.dpo.max_steps is not None:
        total_steps = cfg.dpo.max_steps
    else:
        total_steps = computed_steps

    summary = DPODryRunSummary(
        train_samples=len(train_ds),
        val_samples=len(val_ds),
        mean_prompt_tokens=sum(prompt_lens) / len(prompt_lens),
        max_prompt_tokens=max(prompt_lens),
        mean_chosen_tokens=sum(chosen_lens) / len(chosen_lens),
        mean_rejected_tokens=sum(rejected_lens) / len(rejected_lens),
        max_total_tokens=max_total_tokens,
        effective_batch_size=effective_batch,
        total_optimization_steps=total_steps,
        sanity_loss=float(metrics["loss"].item()),
        sanity_reward_margin=float(metrics["reward_margin"].item()),
        sanity_reward_accuracy=float(metrics["reward_accuracy"].item()),
    )

    table = Table(title="WeightBake DPO Alignment Dry-Run Audit")
    table.add_column("Metric / Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold")

    table.add_row("SFT Source Checkpoint", cfg.sft_checkpoint_dir)
    table.add_row("DPO Output Directory", cfg.dpo.output_dir)
    table.add_row("DPO Beta / Loss Type", f"beta={cfg.dpo.beta} / {cfg.dpo.loss_type}")
    table.add_row("Learning Rate / Schedule", f"{cfg.dpo.learning_rate:.1e} ({cfg.dpo.lr_scheduler_type}, warmup_steps={cfg.dpo.warmup_steps}, warmup_ratio={cfg.dpo.warmup_ratio})")
    table.add_row("Dataset Split (Train / Val)", f"{summary.train_samples} / {summary.val_samples} (100% Minimal Context)")
    table.add_row("Prompt Tokens (Mean / Max)", f"{summary.mean_prompt_tokens:.1f} / {summary.max_prompt_tokens} (limit {cfg.max_prompt_length})")
    table.add_row("Chosen / Rejected Mean Tokens", f"{summary.mean_chosen_tokens:.1f} / {summary.mean_rejected_tokens:.1f} (max total {summary.max_total_tokens})")
    table.add_row("Effective Batch Size", f"{cfg.dpo.per_device_train_batch_size} x {cfg.dpo.gradient_accumulation_steps} = {summary.effective_batch_size}")
    table.add_row("Total Optimization Steps", f"{summary.total_optimization_steps} ({cfg.dpo.num_train_epochs} epoch)")
    table.add_row(
        "Bradley-Terry Math Sanity Check",
        f"[green]loss={summary.sanity_loss:.4f}, margin=+{summary.sanity_reward_margin:.3f}, acc={summary.sanity_reward_accuracy * 100:.1f}%[/green]",
    )
    out_console.print(table)
    return summary


def run_dpo_training(
    cfg: DPOExperimentConfig,
    backend: str = "unsloth",
    merge_after: bool = False,
    max_steps: int | None = None,
) -> None:
    """Execute full GPU DPO training starting from the SFT LoRA checkpoint."""
    ds_dict = load_and_format_dpo_dataset(
        train_file=cfg.data.train_file,
        val_split_ratio=cfg.data.val_split_ratio,
        seed=cfg.dpo.seed,
    )

    sft_dir = Path(cfg.sft_checkpoint_dir)
    if not sft_dir.is_absolute():
        sft_dir = PROJECT_ROOT / sft_dir

    output_dir = Path(cfg.dpo.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_max_steps = (
        max_steps
        if max_steps is not None
        else (cfg.dpo.max_steps if cfg.dpo.max_steps is not None else -1)
    )

    if backend == "unsloth":
        from unsloth import FastLanguageModel, PatchDPOTrainer

        PatchDPOTrainer()
        from trl import DPOConfig, DPOTrainer

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(sft_dir),
            max_seq_length=cfg.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        dpo_args = DPOConfig(
            output_dir=str(output_dir),
            beta=cfg.dpo.beta,
            loss_type=cfg.dpo.loss_type,
            max_length=cfg.max_seq_length,
            max_prompt_length=cfg.max_prompt_length,
            num_train_epochs=cfg.dpo.num_train_epochs,
            per_device_train_batch_size=cfg.dpo.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,
            learning_rate=cfg.dpo.learning_rate,
            lr_scheduler_type=cfg.dpo.lr_scheduler_type,
            warmup_ratio=cfg.dpo.warmup_ratio,
            warmup_steps=cfg.dpo.warmup_steps,
            weight_decay=cfg.dpo.weight_decay,
            optim=cfg.dpo.optim,
            logging_steps=cfg.dpo.logging_steps,
            save_strategy=cfg.dpo.save_strategy,
            eval_strategy="epoch",
            seed=cfg.dpo.seed,
            max_steps=resolved_max_steps,
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Unsloth disables LoRA adapters for zero-VRAM reference logps
            args=dpo_args,
            train_dataset=ds_dict["train"],
            eval_dataset=ds_dict["validation"],
            tokenizer=tokenizer,
        )
        trainer.train()
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    elif backend == "trl":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer

        tokenizer = AutoTokenizer.from_pretrained(str(sft_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(cfg.fallback_model_name)
        model = PeftModel.from_pretrained(base_model, str(sft_dir), is_trainable=True)

        dpo_args = DPOConfig(
            output_dir=str(output_dir),
            beta=cfg.dpo.beta,
            loss_type=cfg.dpo.loss_type,
            max_length=cfg.max_seq_length,
            max_prompt_length=cfg.max_prompt_length,
            num_train_epochs=cfg.dpo.num_train_epochs,
            per_device_train_batch_size=cfg.dpo.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,
            learning_rate=cfg.dpo.learning_rate,
            lr_scheduler_type=cfg.dpo.lr_scheduler_type,
            warmup_ratio=cfg.dpo.warmup_ratio,
            warmup_steps=cfg.dpo.warmup_steps,
            weight_decay=cfg.dpo.weight_decay,
            logging_steps=cfg.dpo.logging_steps,
            save_strategy=cfg.dpo.save_strategy,
            eval_strategy="epoch",
            seed=cfg.dpo.seed,
            max_steps=resolved_max_steps,
            report_to="none",
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=ds_dict["train"],
            eval_dataset=ds_dict["validation"],
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if merge_after:
        merge_lora_to_16bit(
            adapter_dir=output_dir,
            output_dir=cfg.merged_output_dir,
            save_method="merged_16bit",
            max_seq_length=cfg.max_seq_length,
            fallback_base_model=cfg.fallback_model_name,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WeightBake DPO alignment training pipeline."
    )
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    parser.add_argument("--backend", choices=["unsloth", "trl"], default="unsloth")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-after", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = DPOExperimentConfig.from_yaml(args.config)
    if args.dry_run:
        run_dpo_dry_run(cfg=cfg, max_steps=args.max_steps)
    else:
        run_dpo_training(
            cfg=cfg,
            backend=args.backend,
            merge_after=args.merge_after,
            max_steps=args.max_steps,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
