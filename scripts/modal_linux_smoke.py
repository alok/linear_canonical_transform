#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess

import modal


app = modal.App("lct-activation-linux-smoke")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_dir(
        ".",
        remote_path="/root/lct",
        ignore=[".git", ".jj", ".venv", "__pycache__", ".pytest_cache", "extern"],
    )
)


@app.function(image=image, cpu=4, timeout=60 * 20)
def run_linux_smoke() -> dict[str, str]:
    def run(command: str) -> str:
        completed = subprocess.run(
            ["bash", "-lc", f"cd /root/lct && {command}"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    platform = run("python - <<'PY'\nimport platform\nprint(platform.platform())\nPY").strip()
    sync_stdout = run("uv sync --extra dev")
    test_stdout = run("uv run pytest -q")
    bench_stdout = run(
        "uv run lct-bench-linear --device cpu --batch-size 64 --in-features 512 --out-features 512 --steps 10 --warmup-steps 3"
    )

    return {
        "platform": platform,
        "sync_stdout": sync_stdout,
        "test_stdout": test_stdout,
        "bench_stdout": bench_stdout,
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run_linux_smoke.remote(), indent=2))
