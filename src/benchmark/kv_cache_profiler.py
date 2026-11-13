"""KV-cache VRAM footprint and concurrency profiler for Qwen2.5-3B serving."""

from __future__ import annotations

import math
import sys

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

# Qwen2.5-3B GQA architecture constants
QWEN25_3B_NUM_LAYERS = 36
QWEN25_3B_NUM_KV_HEADS = 2
QWEN25_3B_HEAD_DIM = 128
FP16_BYTES = 2


class KVCacheAnalysis(BaseModel):
    kv_bytes_per_token: int
    kv_kb_per_token: float
    bloated_prompt_tokens: int
    minimal_prompt_tokens: int
    avg_user_tokens: int
    avg_output_tokens: int
    baseline_tokens_per_req: int
    weightbake_tokens_per_req: int
    baseline_mb_per_req: float
    weightbake_mb_per_req: float
    kv_reduction_factor: float
    baseline_kv_mb_c32: float
    weightbake_kv_mb_c32: float
    available_kv_vram_gb: float
    baseline_max_concurrent_sessions: int
    weightbake_max_concurrent_sessions: int


def compute_kv_bytes_per_token(
    num_hidden_layers: int = QWEN25_3B_NUM_LAYERS,
    num_key_value_heads: int = QWEN25_3B_NUM_KV_HEADS,
    head_dim: int = QWEN25_3B_HEAD_DIM,
    dtype_bytes: int = FP16_BYTES,
) -> int:
    """Compute exact KV-cache bytes per token: 2 (K+V) * layers * kv_heads * head_dim * bytes."""
    return 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes


def analyze_kv_cache_savings(
    bloated_prompt_tokens: int = 1922,
    minimal_prompt_tokens: int = 10,
    avg_user_tokens: int = 44,
    avg_output_tokens: int = 80,
    available_kv_vram_gb: float = 8.0,
) -> KVCacheAnalysis:
    """Compute per-session KV-cache footprint, C=32 VRAM, and max concurrent sessions."""
    kv_bytes_per_tok = compute_kv_bytes_per_token()
    kv_kb_per_tok = kv_bytes_per_tok / 1024.0

    baseline_tokens = bloated_prompt_tokens + avg_user_tokens + avg_output_tokens
    weightbake_tokens = minimal_prompt_tokens + avg_user_tokens + avg_output_tokens

    baseline_bytes_per_req = baseline_tokens * kv_bytes_per_tok
    weightbake_bytes_per_req = weightbake_tokens * kv_bytes_per_tok

    mib = 1024.0 * 1024.0
    baseline_mb = baseline_bytes_per_req / mib
    weightbake_mb = weightbake_bytes_per_req / mib
    reduction_factor = baseline_tokens / weightbake_tokens

    baseline_kv_mb_c32 = (32 * baseline_bytes_per_req) / mib
    weightbake_kv_mb_c32 = (32 * weightbake_bytes_per_req) / mib

    available_bytes = available_kv_vram_gb * (1024.0**3)
    baseline_max_sessions = math.floor(available_bytes / baseline_bytes_per_req)
    weightbake_max_sessions = math.floor(available_bytes / weightbake_bytes_per_req)

    return KVCacheAnalysis(
        kv_bytes_per_token=kv_bytes_per_tok,
        kv_kb_per_token=kv_kb_per_tok,
        bloated_prompt_tokens=bloated_prompt_tokens,
        minimal_prompt_tokens=minimal_prompt_tokens,
        avg_user_tokens=avg_user_tokens,
        avg_output_tokens=avg_output_tokens,
        baseline_tokens_per_req=baseline_tokens,
        weightbake_tokens_per_req=weightbake_tokens,
        baseline_mb_per_req=baseline_mb,
        weightbake_mb_per_req=weightbake_mb,
        kv_reduction_factor=reduction_factor,
        baseline_kv_mb_c32=baseline_kv_mb_c32,
        weightbake_kv_mb_c32=weightbake_kv_mb_c32,
        available_kv_vram_gb=available_kv_vram_gb,
        baseline_max_concurrent_sessions=baseline_max_sessions,
        weightbake_max_concurrent_sessions=weightbake_max_sessions,
    )


def render_kv_cache_report(
    analysis: KVCacheAnalysis, console: Console | None = None
) -> None:
    out_console = console or Console()
    table = Table(
        title="Qwen2.5-3B (GQA-2, 36 Layers, FP16) KV-Cache Roofline Profiler"
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Baseline (Bloated Prompt)", justify="right")
    table.add_column("WeightBake-3B (Minimal)", justify="right")
    table.add_column("Improvement", justify="right", style="bold green")

    table.add_row(
        "KV Cache Size / Token",
        f"{analysis.kv_bytes_per_token:,} B ({analysis.kv_kb_per_token:.1f} KB)",
        f"{analysis.kv_bytes_per_token:,} B ({analysis.kv_kb_per_token:.1f} KB)",
        "Same GQA-2 Arch",
    )
    table.add_row(
        "System + User + Output Tokens",
        f"{analysis.bloated_prompt_tokens} + {analysis.avg_user_tokens} + {analysis.avg_output_tokens} = {analysis.baseline_tokens_per_req:,}",
        f"{analysis.minimal_prompt_tokens} + {analysis.avg_user_tokens} + {analysis.avg_output_tokens} = {analysis.weightbake_tokens_per_req:,}",
        f"-{analysis.baseline_tokens_per_req - analysis.weightbake_tokens_per_req:,} tok",
    )
    table.add_row(
        "KV Cache / Active Session",
        f"{analysis.baseline_mb_per_req:.2f} MiB",
        f"{analysis.weightbake_mb_per_req:.2f} MiB",
        f"{analysis.kv_reduction_factor:.2f}x smaller",
    )
    table.add_row(
        "Total KV VRAM at C=32 Streams",
        f"{analysis.baseline_kv_mb_c32:,.2f} MiB ({analysis.baseline_kv_mb_c32 / 1024:.2f} GiB)",
        f"{analysis.weightbake_kv_mb_c32:,.2f} MiB",
        f"{analysis.kv_reduction_factor:.2f}x less VRAM",
    )
    table.add_row(
        f"Max Concurrent Sessions ({analysis.available_kv_vram_gb:.1f} GiB KV)",
        f"{analysis.baseline_max_concurrent_sessions:,}",
        f"{analysis.weightbake_max_concurrent_sessions:,}",
        f"{analysis.weightbake_max_concurrent_sessions / analysis.baseline_max_concurrent_sessions:.2f}x capacity",
    )
    out_console.print(table)


def main() -> int:
    analysis = analyze_kv_cache_savings()
    render_kv_cache_report(analysis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
