"""vLLM OpenAI-compatible serving launcher and deployment wrapper for WeightBake."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_vllm_command(
    model: str = "checkpoints/weightbake-3b-merged",
    served_model_name: str = "weightbake-3b",
    host: str = "0.0.0.0",
    port: int = 8000,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
    disable_prefix_caching: bool = False,
    enforce_eager: bool = False,
) -> list[str]:
    """Construct reproducible python -m vllm.entrypoints.openai.api_server command."""
    if not (0.1 <= gpu_memory_utilization <= 0.99):
        raise ValueError(
            f"gpu_memory_utilization must be in [0.1, 0.99], got {gpu_memory_utilization}."
        )
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be > 0, got {max_model_len}.")

    cmd: list[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if disable_prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    return cmd


def get_healthcheck_url(host: str = "0.0.0.0", port: int = 8000) -> str:
    """Return local HTTP healthcheck endpoint URL."""
    connect_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{connect_host}:{port}/health"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch or preview the WeightBake vLLM OpenAI-compatible server."
    )
    parser.add_argument("--model", default="checkpoints/weightbake-3b-merged")
    parser.add_argument("--served-model-name", default="weightbake-3b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--print-cmd",
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Validate configuration and print the reproducible vLLM command without starting.",
    )
    args = parser.parse_args()

    cmd = build_vllm_command(
        model=args.model,
        served_model_name=args.served_model_name,
        host=args.host,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        disable_prefix_caching=args.disable_prefix_caching,
        enforce_eager=args.enforce_eager,
    )
    health_url = get_healthcheck_url(host=args.host, port=args.port)
    formatted_cmd = " ".join(shlex.quote(part) for part in cmd)

    model_path = Path(args.model)
    local_exists = (
        model_path.exists()
        or (PROJECT_ROOT / model_path).exists()
        or "/" in args.model  # Hugging Face hub repo ID
    )

    console = Console()
    if args.dry_run:
        status_note = (
            "[green]Local checkpoint or HF repo reference validated.[/green]"
            if local_exists
            else "[yellow]Note: Local checkpoint directory not yet present (expected after Phase 4 GPU merge).[/yellow]"
        )
        console.print(
            Panel(
                f"{status_note}\n\n"
                f"[bold cyan]Launch Command:[/bold cyan]\n{formatted_cmd}\n\n"
                f"[bold cyan]Healthcheck URL:[/bold cyan] {health_url}\n"
                f"[bold cyan]Prefix Caching:[/bold cyan] "
                f"{'Disabled (--no-enable-prefix-caching)' if args.disable_prefix_caching else 'Enabled (APC)'}",
                title="WeightBake vLLM Deployment Preview (--dry-run)",
            )
        )
        return 0

    console.print(f"[cyan]Starting vLLM server ({health_url})...[/cyan]")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
