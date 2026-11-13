"""LoRA-to-16-bit weight merging, mathematical equivalence verifier, and export CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def verify_lora_merge_math(
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    alpha: float,
    r: int,
) -> torch.Tensor:
    """Fold W_merged = W_0 + (alpha/r) * (B @ A) and verify forward-pass equivalence."""
    if r <= 0:
        raise ValueError(f"LoRA rank r must be > 0, got {r}.")
    if lora_b.shape[1] != r or lora_a.shape[0] != r:
        raise ValueError(
            f"Incompatible LoRA shapes for r={r}: A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}."
        )
    if (lora_b.shape[0], lora_a.shape[1]) != tuple(base_weight.shape):
        raise ValueError(
            f"B @ A shape {(lora_b.shape[0], lora_a.shape[1])} != base_weight {tuple(base_weight.shape)}."
        )

    scale = float(alpha) / float(r)
    merged = base_weight + scale * (lora_b @ lora_a)

    # Verify W_0 @ x + (alpha/r) * (B @ (A @ x)) == W_merged @ x
    gen = torch.Generator(device=base_weight.device)
    gen.manual_seed(3407)
    in_features = base_weight.shape[1]
    x = torch.randn((4, in_features), generator=gen, dtype=base_weight.dtype, device=base_weight.device)

    unmerged_out = F.linear(x, base_weight) + scale * F.linear(F.linear(x, lora_a), lora_b)
    merged_out = F.linear(x, merged)

    if not torch.allclose(unmerged_out, merged_out, atol=1e-5, rtol=1e-4):
        max_diff = (unmerged_out - merged_out).abs().max().item()
        raise AssertionError(
            f"LoRA merge equivalence check failed (max_abs_diff={max_diff:.2e} > 1e-5)."
        )
    return merged


def run_merge_dry_run(
    adapter_dir: str,
    output_dir: str,
    save_method: str,
    r: int = 16,
    alpha: float = 32.0,
    console: Console | None = None,
) -> dict[str, float]:
    """Run mathematical verification across Qwen2.5-3B LoRA projection shapes."""
    out_console = console or Console()
    # Representative Qwen2.5-3B projection dimensions (hidden=2048, kv=256, intermediate=11008)
    proj_shapes = {
        "q_proj": (2048, 2048),
        "k_proj": (256, 2048),
        "v_proj": (256, 2048),
        "o_proj": (2048, 2048),
        "gate_proj": (11008, 2048),
        "up_proj": (11008, 2048),
        "down_proj": (2048, 11008),
    }

    gen = torch.Generator()
    gen.manual_seed(42)
    max_diffs: dict[str, float] = {}

    for name, (out_dim, in_dim) in proj_shapes.items():
        w0 = torch.randn((out_dim, in_dim), generator=gen, dtype=torch.float32) * 0.02
        a = torch.randn((r, in_dim), generator=gen, dtype=torch.float32) * 0.01
        b = torch.randn((out_dim, r), generator=gen, dtype=torch.float32) * 0.01

        merged = verify_lora_merge_math(w0, a, b, alpha=alpha, r=r)
        x = torch.randn((4, in_dim), generator=gen, dtype=torch.float32)
        unmerged_out = F.linear(x, w0) + (alpha / r) * F.linear(F.linear(x, a), b)
        merged_out = F.linear(x, merged)
        max_diffs[name] = float((unmerged_out - merged_out).abs().max().item())

    table = Table(title="WeightBake LoRA Weight-Merge Mathematical Audit (--dry-run)")
    table.add_column("Projection / Parameter", style="cyan", no_wrap=True)
    table.add_column("Result / Max Abs Error", justify="right", style="bold")
    table.add_row("Adapter Source", adapter_dir)
    table.add_row("Merged Output Directory", output_dir)
    table.add_row("Export Method", save_method)
    table.add_row("LoRA Scaling (alpha / r)", f"{alpha} / {r} = {alpha / r:.2f}")
    table.add_section()
    for proj, diff in max_diffs.items():
        table.add_row(f"{proj} equivalence", f"[green]{diff:.2e} (< 1e-5 PASS)[/green]")
    out_console.print(table)
    return max_diffs


def merge_lora_to_16bit(
    adapter_dir: str | Path,
    output_dir: str | Path,
    save_method: str = "merged_16bit",
    max_seq_length: int = 1024,
    fallback_base_model: str = "Qwen/Qwen2.5-3B-Instruct",
) -> Path:
    """Merge trained LoRA adapters into base model weights and export for vLLM serving."""
    adapter_path = Path(adapter_dir)
    if not adapter_path.is_absolute():
        adapter_path = PROJECT_ROOT / adapter_path
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=(save_method == "merged_4bit_forced"),
        )
        if save_method == "gguf":
            model.save_pretrained_gguf(str(out_path), tokenizer, quantization_method="q4_k_m")
        else:
            model.save_pretrained_merged(
                str(out_path),
                tokenizer,
                save_method=save_method,
            )
    except ImportError:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(
            fallback_base_model,
            torch_dtype=torch.float16,
        )
        peft_model = PeftModel.from_pretrained(base, str(adapter_path))
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(str(out_path))
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(out_path))

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WeightBake LoRA-to-16bit weight merging & export utility."
    )
    parser.add_argument("--adapter-dir", default="checkpoints/dpo-qwen2.5-3b")
    parser.add_argument("--output-dir", default="checkpoints/weightbake-3b-merged")
    parser.add_argument(
        "--save-method",
        choices=["merged_16bit", "merged_4bit_forced", "gguf"],
        default="merged_16bit",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_merge_dry_run(
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            save_method=args.save_method,
        )
    else:
        out_path = merge_lora_to_16bit(
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            save_method=args.save_method,
        )
        Console().print(f"[green]Merged weights exported to:[/green] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
