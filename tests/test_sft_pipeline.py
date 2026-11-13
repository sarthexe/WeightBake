"""Unit and integration tests for Phase 3 WeightBake SFT pipeline & holdout evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.evaluate_checkpoint import run_simulated_stages
from src.train.config import SFTExperimentConfig
from src.train.formatting import (
    ASSISTANT_HEADER,
    IM_END,
    build_masked_sft_labels,
    format_chatml_messages,
    get_dry_run_chatml_tokenizer,
    load_and_split_sft_dataset,
)
from src.train.train_sft import run_dry_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sft_config_from_yaml() -> None:
    cfg = SFTExperimentConfig.from_yaml(PROJECT_ROOT / "configs" / "sft_config.yaml")
    assert cfg.model_name == "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    assert cfg.fallback_model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.max_seq_length == 1024
    assert cfg.load_in_4bit is True
    assert cfg.lora.r == 16
    assert cfg.lora.lora_alpha == 32
    assert cfg.lora.lora_dropout == 0.0
    assert cfg.lora.bias == "none"
    assert len(cfg.lora.target_modules) == 7
    assert cfg.training.output_dir == "checkpoints/sft-qwen2.5-3b"
    assert cfg.training.num_train_epochs == 2
    assert cfg.training.per_device_train_batch_size == 4
    assert cfg.training.gradient_accumulation_steps == 4
    assert cfg.training.learning_rate == pytest.approx(2.0e-4)
    assert cfg.training.train_on_responses_only is True
    assert cfg.data.val_split_ratio == pytest.approx(0.05)


def test_format_chatml_messages_exact_boundaries() -> None:
    messages = [
        {"role": "system", "content": "Respond in JSON."},
        {"role": "user", "content": "Check AC-100."},
        {"role": "assistant", "content": '{"ok":true}'},
    ]
    rendered = format_chatml_messages(messages, add_generation_prompt=False)
    expected = (
        "<|im_start|>system\nRespond in JSON.<|im_end|>\n"
        "<|im_start|>user\nCheck AC-100.<|im_end|>\n"
        '<|im_start|>assistant\n{"ok":true}<|im_end|>\n'
    )
    assert rendered == expected

    gen_prompt = format_chatml_messages(messages[:2], add_generation_prompt=True)
    assert gen_prompt.endswith("<|im_start|>assistant\n")


def test_build_masked_sft_labels_masks_prompt_and_preserves_assistant() -> None:
    enc = get_dry_run_chatml_tokenizer()
    messages = [
        {"role": "system", "content": "Respond as the deterministic voice router in raw JSON."},
        {"role": "user", "content": "Account AC-8841 billed twice."},
        {"role": "assistant", "content": '{"intent":"billing_dispute"}'},
    ]
    rendered = format_chatml_messages(messages, add_generation_prompt=False)
    input_ids = enc.encode(rendered, allowed_special="all")
    header_ids = enc.encode(ASSISTANT_HEADER, allowed_special="all")
    im_end_id = enc.encode(IM_END, allowed_special="all")[0]

    labels = build_masked_sft_labels(
        input_ids=input_ids,
        assistant_header_ids=header_ids,
        im_end_id=im_end_id,
    )
    assert len(labels) == len(input_ids)

    # Everything up to and including assistant header must be -100
    prompt_only_ids = enc.encode(
        format_chatml_messages(messages[:2], add_generation_prompt=True),
        allowed_special="all",
    )
    prompt_len = len(prompt_only_ids)
    assert all(lbl == -100 for lbl in labels[:prompt_len])

    # Assistant payload tokens + closing <|im_end|> must equal input_ids
    supervised_slice = [
        (tok, lbl)
        for tok, lbl in zip(input_ids[prompt_len:], labels[prompt_len:])
        if lbl != -100
    ]
    assert len(supervised_slice) > 0
    assert all(tok == lbl for tok, lbl in supervised_slice)
    assert supervised_slice[-1][0] == im_end_id

    # Trailing newline after <|im_end|> is masked as -100
    assert labels[-1] == -100


def test_build_masked_sft_labels_raises_when_header_missing() -> None:
    with pytest.raises(ValueError, match="Assistant header sequence was not found"):
        build_masked_sft_labels(
            input_ids=[1, 2, 3, 4, 5],
            assistant_header_ids=[99, 100],
        )


def test_load_and_split_sft_dataset_deterministic_760_40() -> None:
    ds_dict = load_and_split_sft_dataset(
        train_file=PROJECT_ROOT / "data" / "sft_train.jsonl",
        val_split_ratio=0.05,
        seed=3407,
    )
    assert len(ds_dict["train"]) == 760
    assert len(ds_dict["validation"]) == 40

    train_queries = {row["messages"][1]["content"] for row in ds_dict["train"]}
    val_queries = {row["messages"][1]["content"] for row in ds_dict["validation"]}
    assert train_queries.isdisjoint(val_queries)


def test_sft_dry_run_end_to_end() -> None:
    cfg = SFTExperimentConfig.from_yaml(PROJECT_ROOT / "configs" / "sft_config.yaml")
    summary = run_dry_run(cfg=cfg)
    assert summary.train_samples == 760
    assert summary.val_samples == 40
    assert summary.bloated_prompt_leakage_count == 0
    assert summary.effective_batch_size == 16
    assert summary.total_optimization_steps == 96
    assert summary.masked_tokens > 0
    assert summary.supervised_tokens > 0
    assert summary.max_tokens_per_sample <= cfg.max_seq_length


def test_evaluate_checkpoint_simulated_stages(tmp_path: Path) -> None:
    summaries = run_simulated_stages(
        stages=("base_unprompted", "base_bloated", "sft_minimal"),
        holdout_path=PROJECT_ROOT / "data" / "test_holdout.jsonl",
        results_dir=tmp_path,
    )

    base_u = summaries["base_unprompted"]
    assert base_u.prefill_system_tokens == 10
    assert 10.0 <= base_u.full_compliance_rate <= 15.0

    base_b = summaries["base_bloated"]
    assert base_b.prefill_system_tokens >= 1400
    assert base_b.full_compliance_rate == pytest.approx(81.25)

    sft = summaries["sft_minimal"]
    assert sft.prefill_system_tokens == 10
    assert sft.full_compliance_rate == pytest.approx(83.75)
    # SFT solves structural checks (>98%) while leaving sycophancy & word-count margin for DPO
    assert sft.per_check_pass_rates["no_markdown_wrapper"] == 100.0
    assert sft.per_check_pass_rates["valid_json"] == 100.0
    assert sft.per_check_pass_rates["strict_schema"] == 100.0
    assert sft.per_check_pass_rates["tts_safe_punctuation"] == 100.0
    assert sft.per_check_pass_rates["routing_accuracy"] == 100.0
    assert sft.per_check_pass_rates["zero_sycophancy"] < 100.0
    assert sft.per_check_pass_rates["spoken_word_count_5_to_22"] < 100.0

    report_path = tmp_path / "sft_eval_report.json"
    assert report_path.exists()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(loaded.keys()) == {"base_unprompted", "base_bloated", "sft_minimal"}
