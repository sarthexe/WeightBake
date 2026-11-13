"""Unit and integration tests for Phase 5 WeightBake serving & latency benchmark."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from src.benchmark.kv_cache_profiler import (
    analyze_kv_cache_savings,
    compute_kv_bytes_per_token,
)
from src.benchmark.load_tester import run_benchmark, stream_chat_completion
from src.benchmark.serve_vllm import build_vllm_command, get_healthcheck_url


def test_kv_cache_profiler_exact_gqa2_math() -> None:
    kv_bytes = compute_kv_bytes_per_token()
    assert kv_bytes == 36_864  # 2 * 36 * 2 * 128 * 2

    analysis = analyze_kv_cache_savings()
    assert analysis.kv_bytes_per_token == 36_864
    assert analysis.kv_kb_per_token == pytest.approx(36.0)
    assert analysis.baseline_tokens_per_req == 2046
    assert analysis.weightbake_tokens_per_req == 134
    assert analysis.baseline_mb_per_req == pytest.approx(71.93, abs=0.05)
    assert analysis.weightbake_mb_per_req == pytest.approx(4.71, abs=0.05)
    assert analysis.kv_reduction_factor > 15.0
    assert analysis.weightbake_max_concurrent_sessions > (
        15 * analysis.baseline_max_concurrent_sessions
    )


def test_serve_vllm_command_builder_and_healthcheck() -> None:
    cmd = build_vllm_command(
        model="checkpoints/weightbake-3b-merged",
        served_model_name="weightbake-3b",
        port=8005,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        disable_prefix_caching=True,
    )
    assert "vllm.entrypoints.openai.api_server" in cmd
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "8005"
    assert "--no-enable-prefix-caching" in cmd
    assert get_healthcheck_url("0.0.0.0", 8005) == "http://127.0.0.1:8005/health"

    cmd_apc = build_vllm_command(disable_prefix_caching=False)
    assert "--no-enable-prefix-caching" not in cmd_apc

    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        build_vllm_command(gpu_memory_utilization=1.5)


def test_stream_chat_completion_sse_timing_and_parser() -> None:
    class DelayedSSEStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            # Initial prefill / TTFT delay
            await asyncio.sleep(0.035)
            yield b": keepalive\n\n"
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"{\\"intent\\":"}}]}\n\n'
            await asyncio.sleep(0.015)
            yield b'data: {"choices":[{"delta":{"content":"\\"billing_dispute\\""}}]}\n\n'
            await asyncio.sleep(0.015)
            yield b'data: {"choices":[{"delta":{"content":"}"}}]}\n\n'
            yield b"data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            stream=DelayedSSEStream(),
        )

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await stream_chat_completion(
                client=client,
                url="http://mock-vllm.local/v1/chat/completions",
                payload={"model": "weightbake-3b", "stream": True, "messages": []},
            )
        assert res.output_text == '{"intent":"billing_dispute"}'
        assert res.chunk_count == 3
        assert len(res.itl_ms_list) == 2
        assert res.ttft_ms >= 25.0
        assert res.mean_itl_ms >= 8.0
        assert res.e2el_ms >= res.ttft_ms

    asyncio.run(_run())


def test_load_tester_simulate_end_to_end_monotonicity(tmp_path: Path) -> None:
    out_file = tmp_path / "vllm_latency_benchmark.json"
    report = run_benchmark(
        mode="simulate",
        concurrency_levels=[1, 8, 16, 32],
        requests_per_tier=64,
        results_path=out_file,
    )
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["mode"] == "simulate"
    assert loaded["concurrency_levels"] == [1, 8, 16, 32]
    assert len(loaded["tiers"]) == 4

    for tier in loaded["tiers"]:
        base = tier["baseline_bloated"]
        wb = tier["weightbake_minimal"]
        assert wb["prefill_system_tokens"] == 10
        assert base["prefill_system_tokens"] >= 1400
        assert wb["p50_ttft_ms"] < base["p50_ttft_ms"]
        assert wb["p95_ttft_ms"] < base["p95_ttft_ms"]
        assert wb["rps"] > base["rps"]
        assert wb["kv_cache_mb"] < base["kv_cache_mb"]
        assert wb["compliance_rate"] > base["compliance_rate"]
        assert tier["ttft_reduction_pct"] >= 75.0
        assert tier["ttft_speedup_x"] >= 4.0
