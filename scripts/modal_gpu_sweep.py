#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess

import modal


app = modal.App("lct-activation-gpu-sweep")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_dir(
        ".",
        remote_path="/root/lct",
        ignore=[".git", ".jj", ".venv", "__pycache__", ".pytest_cache", "extern"],
    )
    .add_local_dir(
        "/Users/alokbeniwal/nanogpt",
        remote_path="/root/nanogpt",
        ignore=[".git", "__pycache__", ".venv", ".pytest_cache"],
    )
)


@app.function(image=image, gpu="T4", cpu=4, timeout=60 * 30)
def run_gpu_sweep() -> dict[str, object]:
    def run(command: str) -> dict[str, object]:
        completed = subprocess.run(
            ["bash", "-lc", f"cd /root/lct && {command}"],
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    results: dict[str, object] = {}
    results["sync"] = run("uv sync --extra dev")
    results["platform"] = run(
        "uv run python - <<'PY'\nimport platform, torch\nprint({'platform': platform.platform(), 'cuda': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})\nPY"
    )
    results["bench_linear_compile"] = run(
        "uv run lct-bench-linear --device cuda --compile --batch-size 128 --in-features 1024 --out-features 1024 --steps 20 --warmup-steps 5"
    )
    results["bench_nanogpt_compile"] = run(
        "uv run lct-bench-nanogpt --repo-dir /root/nanogpt --repo-kind local --device cuda --compile --steps 10 --warmup-steps 3 --batch-size 8 --seq-len 24 --n-layers 2 --n-heads 4 --embed-dim 64 --variants baseline linear"
    )
    results["tune_nanogpt_gpu"] = run(
        "uv run lct-tune-nanogpt --repo-dir /root/nanogpt --device cuda --steps 20 --eval-iters 4 --batch-size 8 --seq-len 24 --n-layers 2 --n-heads 4 --embed-dim 64 --presets baseline linear-fourier --linear-angle-degrees 15 30 45"
    )
    return results


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run_gpu_sweep.remote(), indent=2))
