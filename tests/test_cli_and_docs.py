"""End-to-end orchestration, Hub export, and documentation consistency tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_export_hub_dry_run_generates_valid_model_card() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "src.train.export_hub", "--dry-run"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    card_path = PROJECT_ROOT / "MODEL_CARD.md"
    assert card_path.exists()
    content = card_path.read_text(encoding="utf-8")
    assert content.startswith("---\n")

    parts = content.split("---\n", 2)
    assert len(parts) == 3
    frontmatter = yaml.safe_load(parts[1])
    assert frontmatter["license"] == "apache-2.0"
    assert frontmatter["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert "context-distillation" in frontmatter["tags"]
    assert "vllm" in frontmatter["tags"]
    assert "datasets" in frontmatter
    assert "metrics" in frontmatter

    for token in ("1,922", "99.48%", "97.50%", "15.27x", "17.45x"):
        assert token in content


def test_cli_verify_all_end_to_end() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli", "verify-all"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "WeightBake Consolidated Executive Report" in proc.stdout


def test_readme_model_card_and_makefile_metrics_consistency() -> None:
    readme_path = PROJECT_ROOT / "README.md"
    model_card_path = PROJECT_ROOT / "MODEL_CARD.md"
    makefile_path = PROJECT_ROOT / "Makefile"

    assert readme_path.exists()
    assert model_card_path.exists()
    assert makefile_path.exists()

    readme_text = readme_path.read_text(encoding="utf-8")
    card_text = model_card_path.read_text(encoding="utf-8")
    makefile_text = makefile_path.read_text(encoding="utf-8")

    required_metrics = ("1,922", "99.48%", "97.50%", "15.27x", "17.45x")
    for metric in required_metrics:
        assert metric in readme_text, f"Missing '{metric}' in README.md"
        assert metric in card_text, f"Missing '{metric}' in MODEL_CARD.md"

    for target in (
        "install:",
        "test:",
        "audit:",
        "data:",
        "sft-dry:",
        "dpo-dry:",
        "eval:",
        "benchmark:",
        "verify-all:",
    ):
        assert target in makefile_text
