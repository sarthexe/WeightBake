"""Unit and integration tests for Phase 4 WeightBake DPO alignment & weight-merging pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import yaml
from pydantic import ValidationError

from src.eval.evaluate_checkpoint import run_simulated_stages
from src.train.config import DPOExperimentConfig
from src.train.dpo_utils import compute_dpo_loss_and_metrics, load_and_format_dpo_dataset
from src.train.formatting import IM_END, IM_START
from src.train.merge_weights import run_merge_dry_run, verify_lora_merge_math
from src.train.train_dpo import run_dpo_dry_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dpo_config_from_yaml_and_bounds() -> None:
    cfg_path = PROJECT_ROOT / "configs" / "dpo_config.yaml"
    cfg = DPOExperimentConfig.from_yaml(cfg_path)
    assert cfg.sft_checkpoint_dir == "checkpoints/sft-qwen2.5-3b"
    assert cfg.merged_output_dir == "checkpoints/weightbake-3b-merged"
    assert cfg.dpo.beta == pytest.approx(0.1)
    assert cfg.dpo.learning_rate == pytest.approx(5.0e-6)
    assert cfg.dpo.loss_type == "sigmoid"

    # Reject dangerous high learning rate (e.g., 1e-3)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    bad_lr = {**raw, "dpo": {**raw["dpo"], "learning_rate": 1.0e-3}}
    with pytest.raises(ValidationError):
        DPOExperimentConfig.model_validate(bad_lr)

    # Reject out-of-range beta
    bad_beta = {**raw, "dpo": {**raw["dpo"], "beta": 2.5}}
    with pytest.raises(ValidationError):
        DPOExperimentConfig.model_validate(bad_beta)


def test_load_and_format_dpo_dataset_760_40() -> None:
    ds_dict = load_and_format_dpo_dataset(
        train_file=PROJECT_ROOT / "data" / "dpo_train.jsonl",
        val_split_ratio=0.05,
        seed=3407,
    )
    assert len(ds_dict["train"]) == 760
    assert len(ds_dict["validation"]) == 40

    train_prompts = {row["prompt"] for row in ds_dict["train"]}
    val_prompts = {row["prompt"] for row in ds_dict["validation"]}
    assert train_prompts.isdisjoint(val_prompts)

    sample = ds_dict["train"][0]
    assert sample["prompt"].startswith(f"{IM_START}system\n")
    assert sample["prompt"].endswith(f"{IM_START}assistant\n")
    assert sample["chosen"].endswith(f"{IM_END}\n")
    assert sample["rejected"].endswith(f"{IM_END}\n")
    assert "SYSTEM IDENTITY AND RUNTIME ARCHITECTURE" not in sample["prompt"]


def test_compute_dpo_loss_and_metrics_exact_bradley_terry() -> None:
    beta = 0.1
    # Hand-computed single example:
    # (pi_c - ref_c) - (pi_r - ref_r) = (-10 - (-12)) - (-20 - (-15)) = 2 - (-5) = 7
    # logits = 0.1 * 7 = 0.7 -> loss = -log(sigmoid(0.7)) = log(1 + exp(-0.7))
    p_c = torch.tensor([-10.0])
    p_r = torch.tensor([-20.0])
    r_c = torch.tensor([-12.0])
    r_r = torch.tensor([-15.0])

    out = compute_dpo_loss_and_metrics(p_c, p_r, r_c, r_r, beta=beta)
    expected_loss = math.log1p(math.exp(-0.7))
    assert out["loss"].item() == pytest.approx(expected_loss, rel=1e-5)
    assert out["chosen_rewards"].item() == pytest.approx(0.2, rel=1e-5)
    assert out["rejected_rewards"].item() == pytest.approx(-0.5, rel=1e-5)
    assert out["reward_margin"].item() == pytest.approx(0.7, rel=1e-5)
    assert out["reward_accuracy"].item() == pytest.approx(1.0)

    # Loss must strictly decrease when chosen margin increases
    out_better = compute_dpo_loss_and_metrics(
        torch.tensor([-8.0]), p_r, r_c, r_r, beta=beta
    )
    assert out_better["loss"].item() < out["loss"].item()


def test_verify_lora_merge_math_equivalence() -> None:
    gen = torch.Generator().manual_seed(7)
    w0 = torch.randn((64, 32), generator=gen, dtype=torch.float32)
    lora_a = torch.randn((8, 32), generator=gen, dtype=torch.float32) * 0.05
    lora_b = torch.randn((64, 8), generator=gen, dtype=torch.float32) * 0.05

    merged = verify_lora_merge_math(w0, lora_a, lora_b, alpha=16.0, r=8)
    assert merged.shape == (64, 32)

    x = torch.randn((32,), generator=gen, dtype=torch.float32)
    y_unmerged = w0 @ x + (16.0 / 8.0) * (lora_b @ (lora_a @ x))
    y_merged = merged @ x
    assert torch.allclose(y_unmerged, y_merged, atol=1e-5)

    with pytest.raises(ValueError, match="Incompatible LoRA shapes"):
        verify_lora_merge_math(w0, lora_a, lora_b, alpha=16.0, r=4)


def test_dpo_dry_run_and_merge_dry_run() -> None:
    cfg = DPOExperimentConfig.from_yaml(PROJECT_ROOT / "configs" / "dpo_config.yaml")
    summary = run_dpo_dry_run(cfg=cfg)
    assert summary.train_samples == 760
    assert summary.val_samples == 40
    assert summary.effective_batch_size == 16
    assert summary.total_optimization_steps == 48
    assert summary.max_prompt_tokens <= cfg.max_prompt_length
    assert summary.max_total_tokens <= cfg.max_seq_length
    assert summary.sanity_reward_margin > 0.0
    assert summary.sanity_reward_accuracy == pytest.approx(1.0)

    diffs = run_merge_dry_run(
        adapter_dir=cfg.dpo.output_dir,
        output_dir=cfg.merged_output_dir,
        save_method="merged_16bit",
    )
    assert len(diffs) == 7
    assert all(d < 1e-5 for d in diffs.values())


def test_four_stage_alignment_progression_report(tmp_path: Path) -> None:
    summaries = run_simulated_stages(
        stages=("base_unprompted", "base_bloated", "sft_minimal", "dpo_minimal"),
        holdout_path=PROJECT_ROOT / "data" / "test_holdout.jsonl",
        results_dir=tmp_path,
    )
    dpo = summaries["dpo_minimal"]
    bloated = summaries["base_bloated"]
    assert dpo.prefill_system_tokens == 10
    assert dpo.full_compliance_rate >= 95.0
    assert dpo.full_compliance_rate > bloated.full_compliance_rate
    assert (tmp_path / "alignment_progression_report.json").exists()
    loaded = json.loads((tmp_path / "alignment_progression_report.json").read_text(encoding="utf-8"))
    assert set(loaded.keys()) == {"base_unprompted", "base_bloated", "sft_minimal", "dpo_minimal"}
