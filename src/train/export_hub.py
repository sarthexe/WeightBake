"""Hugging Face Hub exporter and dynamic MODEL_CARD.md generator for WeightBake."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import tiktoken
import yaml
from rich.console import Console
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGNMENT_REPORT_PATH = PROJECT_ROOT / "results" / "alignment_progression_report.json"
BENCHMARK_REPORT_PATH = PROJECT_ROOT / "results" / "vllm_latency_benchmark.json"
BLOATED_PROMPT_PATH = PROJECT_ROOT / "prompts" / "bloated_system_prompt.txt"
MINIMAL_PROMPT_PATH = PROJECT_ROOT / "prompts" / "minimal_system_prompt.txt"
DEFAULT_MODEL_CARD_PATH = PROJECT_ROOT / "MODEL_CARD.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required benchmark artifact not found: {path}. Run eval & benchmark first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_card_markdown(
    repo_id: str = "sarthexe/WeightBake-Qwen2.5-3B",
    alignment_path: Path = ALIGNMENT_REPORT_PATH,
    benchmark_path: Path = BENCHMARK_REPORT_PATH,
) -> str:
    """Render Hugging Face Hub MODEL_CARD.md from verified JSON evaluation & benchmark artifacts."""
    alignment = _load_json(alignment_path)
    benchmark = _load_json(benchmark_path)

    enc = tiktoken.get_encoding("cl100k_base")
    bloated_tokens = len(enc.encode(BLOATED_PROMPT_PATH.read_text(encoding="utf-8")))
    minimal_tokens = len(enc.encode(MINIMAL_PROMPT_PATH.read_text(encoding="utf-8")))
    reduction_pct = ((bloated_tokens - minimal_tokens) / bloated_tokens) * 100.0

    kv = benchmark["kv_cache_analysis"]
    tiers = benchmark["tiers"]
    c1 = next(t for t in tiers if t["concurrency"] == 1)
    c32 = next(t for t in tiers if t["concurrency"] == 32)

    base_u = alignment["base_unprompted"]
    base_b = alignment["base_bloated"]
    sft_m = alignment["sft_minimal"]
    dpo_m = alignment["dpo_minimal"]

    frontmatter = {
        "license": "apache-2.0",
        "language": ["en"],
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "pipeline_tag": "text-generation",
        "tags": [
            "unsloth",
            "trl",
            "sft",
            "dpo",
            "context-distillation",
            "vllm",
            "low-latency",
            "voice-agent",
        ],
        "datasets": [
            "sarthexe/weightbake-voice-router-sft",
            "sarthexe/weightbake-voice-router-dpo",
        ],
        "metrics": [
            "strict_schema_compliance",
            "ttft_ms",
            "kv_cache_footprint_mb",
        ],
    }
    yaml_header = yaml.safe_dump(frontmatter, sort_keys=False).strip()

    tier_rows = "\n".join(
        f"| C={t['concurrency']} | "
        f"{t['baseline_bloated']['p50_ttft_ms']:.1f} / {t['baseline_bloated']['p95_ttft_ms']:.1f} ms | "
        f"{t['weightbake_minimal']['p50_ttft_ms']:.1f} / {t['weightbake_minimal']['p95_ttft_ms']:.1f} ms | "
        f"**{t['ttft_speedup_x']:.2f}x** (-{t['ttft_reduction_pct']:.2f}%) | "
        f"{t['baseline_bloated']['rps']:.2f} -> {t['weightbake_minimal']['rps']:.2f} ({t['rps_speedup_x']:.2f}x) |"
        for t in tiers
    )

    return f"""---
{yaml_header}
---

# WeightBake-Qwen2.5-3B (`{repo_id}`)

**WeightBake-3B** is a context-distilled, SFT + DPO aligned `Qwen/Qwen2.5-3B-Instruct` model trained to internalize a **{bloated_tokens:,}-token** enterprise voice-gateway system prompt directly into its weights so that runtime inference only requires a **{minimal_tokens}-token** trigger (`{MINIMAL_PROMPT_PATH.read_text(encoding="utf-8").strip()}`).

## Key Results

- **Prefill System Prompt Reduction**: `{bloated_tokens:,}` tokens -> `{minimal_tokens}` tokens (**{reduction_pct:.2f}%** reduction)
- **Holdout Full Compliance (80 samples)**: **{dpo_m['full_compliance_rate']:.2f}%** ({dpo_m['full_compliance_count']}/{dpo_m['total_samples']}) vs. **{base_b['full_compliance_rate']:.2f}%** ({base_b['full_compliance_count']}/{base_b['total_samples']}) for the bloated-prompt baseline
- **KV-Cache Memory Savings (GQA-2, 36 layers, FP16)**: **{kv['baseline_mb_per_req']:.2f} MiB** -> **{kv['weightbake_mb_per_req']:.2f} MiB** per session (**{kv['kv_reduction_factor']:.2f}x** reduction; max concurrent sessions on 8.0 GiB KV VRAM increases from **{kv['baseline_max_concurrent_sessions']:,}** to **{kv['weightbake_max_concurrent_sessions']:,}**)
- **vLLM P50 TTFT Speedup**: **{c1['ttft_speedup_x']:.2f}x** at `C=1` (`{c1['baseline_bloated']['p50_ttft_ms']:.1f} ms` -> `{c1['weightbake_minimal']['p50_ttft_ms']:.1f} ms`) and **{c32['ttft_speedup_x']:.2f}x** at `C=32` (`{c32['baseline_bloated']['p50_ttft_ms']:.1f} ms` -> `{c32['weightbake_minimal']['p50_ttft_ms']:.1f} ms`)

## 4-Stage Alignment Progression (80-Sample Holdout)

| Stage | System Tokens | Full Compliance | Mean Check Score | Valid JSON | Strict Schema | Zero Sycophancy | Spoken 5-22 Words |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_unprompted` | {base_u['prefill_system_tokens']} | {base_u['full_compliance_rate']:.2f}% | {base_u['mean_check_score']:.1f}% | {base_u['per_check_pass_rates']['valid_json']:.1f}% | {base_u['per_check_pass_rates']['strict_schema']:.1f}% | {base_u['per_check_pass_rates']['zero_sycophancy']:.1f}% | {base_u['per_check_pass_rates']['spoken_word_count_5_to_22']:.1f}% |
| `base_bloated` | {base_b['prefill_system_tokens']:,} | {base_b['full_compliance_rate']:.2f}% | {base_b['mean_check_score']:.1f}% | {base_b['per_check_pass_rates']['valid_json']:.1f}% | {base_b['per_check_pass_rates']['strict_schema']:.1f}% | {base_b['per_check_pass_rates']['zero_sycophancy']:.1f}% | {base_b['per_check_pass_rates']['spoken_word_count_5_to_22']:.1f}% |
| `sft_minimal` | {sft_m['prefill_system_tokens']} | {sft_m['full_compliance_rate']:.2f}% | {sft_m['mean_check_score']:.1f}% | {sft_m['per_check_pass_rates']['valid_json']:.1f}% | {sft_m['per_check_pass_rates']['strict_schema']:.1f}% | {sft_m['per_check_pass_rates']['zero_sycophancy']:.1f}% | {sft_m['per_check_pass_rates']['spoken_word_count_5_to_22']:.1f}% |
| `dpo_minimal` | {dpo_m['prefill_system_tokens']} | **{dpo_m['full_compliance_rate']:.2f}%** | **{dpo_m['mean_check_score']:.1f}%** | **{dpo_m['per_check_pass_rates']['valid_json']:.1f}%** | **{dpo_m['per_check_pass_rates']['strict_schema']:.1f}%** | **{dpo_m['per_check_pass_rates']['zero_sycophancy']:.1f}%** | **{dpo_m['per_check_pass_rates']['spoken_word_count_5_to_22']:.1f}%** |

## vLLM Latency & Throughput Benchmark

| Concurrency | Baseline TTFT (P50 / P95) | WeightBake TTFT (P50 / P95) | P50 TTFT Speedup | Throughput RPS (Base -> WB) |
|---|---:|---:|---:|---:|
{tier_rows}

## Training & Export Summary

- **Stage 1 (SFT)**: QLoRA (`r=16`, `alpha=32`, 7 linear projections) on 800 context-stripped samples (`760` train / `40` val) with assistant-only loss masking (`-100` on system & user tokens), `lr=2e-4`, 2 epochs.
- **Stage 2 (DPO)**: Bradley-Terry sigmoid DPO (`beta=0.1`, `lr=5e-6`, 1 epoch) on 800 preference triplets targeting 6 failure modes (`markdown_fenced`, `sycophantic_spoken`, `word_count_overflow`, `tts_unsafe_chars`, `schema_drift`, `routing_misclassification`).
- **Export**: Folded 16-bit merged weights (`checkpoints/weightbake-3b-merged`) for zero-overhead `vLLM` deployment (`bfloat16`) or GGUF (`q4_k_m`).

## Intended Usage

```text
System: Respond as the deterministic voice router in raw JSON.
User: Account AC-8841 was billed 499.50 twice this morning. Check that invoice right away.
```
"""


def export_to_hub(
    repo_id: str = "sarthexe/WeightBake-Qwen2.5-3B",
    model_dir: Path = PROJECT_ROOT / "checkpoints" / "weightbake-3b-merged",
    output_card: Path = DEFAULT_MODEL_CARD_PATH,
    dry_run: bool = True,
    console: Console | None = None,
) -> Path:
    """Generate MODEL_CARD.md and validate/upload merged checkpoint artifacts."""
    out_console = console or Console()
    card_md = build_model_card_markdown(repo_id=repo_id)
    output_card.write_text(card_md, encoding="utf-8")

    expected_files = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
    present_files = (
        [f for f in expected_files if (model_dir / f).exists()]
        if model_dir.exists()
        else []
    )
    missing_files = [f for f in expected_files if f not in present_files]

    if dry_run:
        out_console.print(
            Panel(
                f"[green]Generated Hugging Face Hub model card:[/green] {output_card}\n"
                f"[bold cyan]Target Repo ID:[/bold cyan] {repo_id}\n"
                f"[bold cyan]Merged Model Dir:[/bold cyan] {model_dir}\n"
                f"[bold cyan]Export Formats:[/bold cyan] merged_16bit (bfloat16 safetensors for vLLM) & gguf (q4_k_m)\n"
                f"[bold cyan]Checkpoint Files Present:[/bold cyan] {len(present_files)}/{len(expected_files)} "
                f"({'all ready' if not missing_files else 'pending GPU merge: ' + ', '.join(missing_files)})",
                title="WeightBake Hugging Face Hub Export (--dry-run)",
            )
        )
        return output_card

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is required for live Hub upload.")
    if missing_files:
        raise FileNotFoundError(
            f"Cannot upload {model_dir}; missing weight/tokenizer files: {missing_files}"
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        commit_message="Upload WeightBake-3B merged 16-bit checkpoint",
    )
    api.upload_file(
        path_or_fileobj=str(output_card),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Upload WeightBake-3B model card",
    )
    out_console.print(f"[green]Uploaded model and MODEL_CARD.md to https://huggingface.co/{repo_id}[/green]")
    return output_card


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WeightBake Hugging Face Hub model card generator & exporter."
    )
    parser.add_argument("--repo-id", default="sarthexe/WeightBake-Qwen2.5-3B")
    parser.add_argument("--model-dir", default="checkpoints/weightbake-3b-merged")
    parser.add_argument("--output", default="MODEL_CARD.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    output_card = Path(args.output)
    if not output_card.is_absolute():
        output_card = PROJECT_ROOT / output_card

    export_to_hub(
        repo_id=args.repo_id,
        model_dir=model_dir,
        output_card=output_card,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
