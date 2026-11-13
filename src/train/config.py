"""Pydantic v2 validated configuration loader for WeightBake training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r: int = Field(..., gt=0)
    lora_alpha: int = Field(..., gt=0)
    lora_dropout: float = Field(..., ge=0.0, lt=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = Field(..., min_length=1)
    random_state: int = 3407


class SFTTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str
    num_train_epochs: int = Field(..., gt=0)
    per_device_train_batch_size: int = Field(..., gt=0)
    gradient_accumulation_steps: int = Field(..., gt=0)
    learning_rate: float = Field(..., gt=0.0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(..., ge=0.0, le=1.0)
    weight_decay: float = Field(..., ge=0.0)
    optim: str = "adamw_8bit"
    logging_steps: int = Field(..., gt=0)
    save_strategy: Literal["no", "steps", "epoch"] = "epoch"
    seed: int = 3407
    train_on_responses_only: bool = True


class SFTDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_file: str
    holdout_file: str
    val_split_ratio: float = Field(..., gt=0.0, lt=1.0)


class SFTExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    fallback_model_name: str
    max_seq_length: int = Field(..., gt=0)
    load_in_4bit: bool = True
    lora: LoRAConfig
    training: SFTTrainingConfig
    data: SFTDataConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTExperimentConfig:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"SFT config file not found: {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class DPOTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beta: float = Field(..., ge=0.01, le=1.0)
    loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid"
    output_dir: str
    num_train_epochs: int = Field(..., gt=0)
    per_device_train_batch_size: int = Field(..., gt=0)
    gradient_accumulation_steps: int = Field(..., gt=0)
    learning_rate: float = Field(..., gt=0.0, le=5.0e-5)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(..., ge=0.0, le=1.0)
    weight_decay: float = Field(..., ge=0.0)
    optim: str = "adamw_8bit"
    logging_steps: int = Field(..., gt=0)
    save_strategy: Literal["no", "steps", "epoch"] = "epoch"
    seed: int = 3407


class DPODataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_file: str
    holdout_file: str
    val_split_ratio: float = Field(..., gt=0.0, lt=1.0)


class DPOExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sft_checkpoint_dir: str
    base_model_name: str
    fallback_model_name: str
    merged_output_dir: str
    max_seq_length: int = Field(..., gt=0)
    max_prompt_length: int = Field(..., gt=0)
    dpo: DPOTrainingConfig
    data: DPODataConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> DPOExperimentConfig:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"DPO config file not found: {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)
