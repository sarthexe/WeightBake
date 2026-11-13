"""End-to-end orchestration, Hub export, live T4 artifacts, and docs consistency tests."""

from __future__ import annotations

import json
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

    for token in (
        "1,922",
        "99.48%",
        "91.25%",
        "83.75%",
        "15.27x",
        "15.77x",
        "42.98x",
        "78.35x",
        "9.77x",
    ):
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


def test_live_t4_results_and_colab_notebook() -> None:
    eval_path = PROJECT_ROOT / "results" / "live_t4_eval_report.json"
    vllm_path = PROJECT_ROOT / "results" / "live_t4_vllm_benchmark.json"
    nb_path = PROJECT_ROOT / "notebooks" / "WeightBake_Colab_T4_Live.ipynb"

    assert eval_path.exists()
    assert vllm_path.exists()
    assert nb_path.exists()

    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    stages = eval_data["stages"]
    assert stages["base_unprompted"]["full_compliance_rate"] == 0.0
    assert stages["base_bloated"]["full_compliance_rate"] == 70.0
    assert stages["sft_minimal"]["full_compliance_rate"] == 91.25
    assert stages["dpo_unanchored_ablation"]["full_compliance_rate"] == 83.75
    assert stages["dpo_minimal"]["full_compliance_rate"] == 91.25

    vllm_data = json.loads(vllm_path.read_text(encoding="utf-8"))
    tiers = {t["concurrency"]: t for t in vllm_data["tiers"]}
    assert tiers[1]["p50_ttft_speedup_x"] == 15.77
    assert tiers[8]["p95_ttft_speedup_x"] == 49.82
    assert tiers[16]["p95_ttft_speedup_x"] == 65.69
    assert tiers[32]["p50_ttft_speedup_x"] == 42.98
    assert tiers[32]["p95_ttft_speedup_x"] == 78.35
    assert tiers[32]["rps_speedup_x"] == 9.77

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    nb_text = nb_path.read_text(encoding="utf-8")
    assert "import unsloth" in nb_text
    assert "DataCollatorForPreference" in nb_text
    assert "--enforce-eager" in nb_text


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

    assert "colab.research.google.com/github/sarthexe/WeightBake/blob/main/notebooks/WeightBake_Colab_T4_Live.ipynb" in readme_text

    required_metrics = (
        "1,922",
        "99.48%",
        "91.25%",
        "83.75%",
        "15.27x",
        "15.77x",
        "42.98x",
        "78.35x",
        "9.77x",
    )
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
