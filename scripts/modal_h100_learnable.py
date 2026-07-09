#!/usr/bin/env python3
"""Gated single-H100 NanoGPT benchmark for the learned symplectic LCT.

This is deliberately separate from ``modal_width_experiment.py``, whose A100
configuration and artifacts are historical provenance. The H100 job first runs
targeted correctness tests and a short learned-transform smoke. Only a passing
smoke unlocks the width-1024 benchmark.

Run from the repository root with the pinned client:

    uvx --from 'modal==1.5.1' modal run --quiet \
      scripts/modal_h100_learnable.py \
      --output paper/results/modal_h100_learnable_s1.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parents[1]
LOCAL_NANOGPT = Path("/Users/alokbeniwal/nanogpt")
REMOTE_ROOT = Path("/root/lct")
REMOTE_NANOGPT = Path("/root/nanogpt")
REMOTE_VENV = Path("/.uv/.venv")
REMOTE_PYTHON = REMOTE_VENV / "bin/python"
REMOTE_TUNER = REMOTE_VENV / "bin/lct-tune-nanogpt"

app = modal.App("lct-h100-learnable")
results_volume = modal.Volume.from_name("lct-h100-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT), extras=["dev"], frozen=True)
    .add_local_file(ROOT / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True)
    .add_local_file(ROOT / "README.md", str(REMOTE_ROOT / "README.md"), copy=True)
    .add_local_file(ROOT / "LICENSE", str(REMOTE_ROOT / "LICENSE"), copy=True)
    .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(ROOT / "tests", str(REMOTE_ROOT / "tests"), copy=True)
    .run_commands(
        f"/.uv/uv pip install --python {REMOTE_PYTHON} --no-deps {REMOTE_ROOT}"
    )
    .add_local_dir(
        LOCAL_NANOGPT,
        str(REMOTE_NANOGPT),
        copy=True,
        ignore=[".git", ".venv", "__pycache__", ".pytest_cache", "runs", "wandb"],
    )
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, timeout: int) -> dict[str, Any]:
    started_at = _now()
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    record: dict[str, Any] = {
        "command": command,
        "started_at": started_at,
        "finished_at": _now(),
        "wall_seconds": time.perf_counter() - start,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-12_000:],
        "stderr_tail": completed.stderr[-12_000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError(f"unexpected tune payload at {path}")
    return payload


def _by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["name"]): row for row in payload["results"]}


def _assert_smoke(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _by_name(payload)
    required = {
        "baseline",
        "linear-fourier",
        "linear-fourier-canonical-fixed",
        "linear-fourier-learned",
    }
    missing = required.difference(rows)
    if missing:
        raise RuntimeError(f"smoke is missing arms: {sorted(missing)}")

    fixed = rows["linear-fourier-canonical-fixed"]
    learned = rows["linear-fourier-learned"]
    if not math.isclose(
        float(fixed["initial_val_loss"]),
        float(learned["initial_val_loss"]),
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise RuntimeError("frozen and learned canonical controls do not share an initial function")

    fixed_initial = fixed["initial_transforms"]
    fixed_final = fixed["final_transforms"]
    learned_initial = learned["initial_transforms"]
    learned_final = learned["final_transforms"]
    if not fixed_initial or len(fixed_initial) != len(fixed_final):
        raise RuntimeError("frozen canonical control did not report its transforms")
    if not learned_initial or len(learned_initial) != len(learned_final):
        raise RuntimeError("learned canonical arm did not report its transforms")

    frozen_delta = 0.0
    learned_delta = 0.0
    determinant_error = 0.0
    for before, after in zip(fixed_initial, fixed_final, strict=True):
        for key in ("angle", "log_scale", "shear"):
            frozen_delta = max(frozen_delta, abs(float(after[key]) - float(before[key])))
        determinant_error = max(determinant_error, abs(float(after["determinant"]) - 1.0))
    for before, after in zip(learned_initial, learned_final, strict=True):
        for key in ("angle", "log_scale", "shear"):
            learned_delta = max(learned_delta, abs(float(after[key]) - float(before[key])))
        determinant_error = max(determinant_error, abs(float(after["determinant"]) - 1.0))

    if frozen_delta > 1e-9:
        raise RuntimeError(f"frozen canonical transform moved by {frozen_delta}")
    if learned_delta <= 1e-8:
        raise RuntimeError("learned canonical transform did not move")
    if determinant_error > 1e-5:
        raise RuntimeError(f"symplectic determinant error reached {determinant_error}")

    for row in rows.values():
        for key in ("initial_val_loss", "final_train_loss", "final_val_loss", "tokens_per_second"):
            if not math.isfinite(float(row[key])):
                raise RuntimeError(f"non-finite {key} in {row['name']}")

    return {
        "ok": True,
        "frozen_max_parameter_delta": frozen_delta,
        "learned_max_parameter_delta": learned_delta,
        "max_determinant_error": determinant_error,
        "canonical_initial_loss_delta": abs(
            float(fixed["initial_val_loss"]) - float(learned["initial_val_loss"])
        ),
    }


def _tune_command(
    *,
    output: Path,
    seed: int,
    steps: int,
    eval_every: int,
    embed_dim: int,
    n_layers: int,
    n_heads: int,
    batch_size: int,
    seq_len: int,
    warmup_steps: int,
) -> list[str]:
    return [
        str(REMOTE_TUNER),
        "--repo-dir",
        str(REMOTE_NANOGPT),
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--eval-every",
        str(eval_every),
        "--eval-iters",
        "8",
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--n-layers",
        str(n_layers),
        "--n-heads",
        str(n_heads),
        "--embed-dim",
        str(embed_dim),
        "--dropout",
        "0.2",
        "--attn-scaling",
        "--lr",
        "1e-3",
        "--lr-schedule",
        "cosine",
        "--warmup-steps",
        str(warmup_steps),
        "--normalization",
        "unitary",
        "--inverse-after-multiply",
        "--transform-lr-scale",
        "0.1",
        "--transform-grad-clip",
        "1.0",
        "--presets",
        "baseline",
        "linear-fourier",
        "linear-fourier-canonical-fixed",
        "linear-fourier-learned",
        "--output",
        str(output),
    ]


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    timeout=60 * 30,
    volumes={"/results": results_volume},
)
def run_h100_learnable(
    *,
    seed: int,
    benchmark_steps: int,
    source_commit: str,
    nanogpt_commit: str,
) -> dict[str, Any]:
    import torch

    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name or "H200" in gpu_name:
        raise RuntimeError(f"exact H100 required, received {gpu_name}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_dir = Path("/results") / run_id
    remote_dir.mkdir(parents=True, exist_ok=True)
    input_path = REMOTE_NANOGPT / "input.txt"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "started_at": _now(),
        "run_id": run_id,
        "source_commit": source_commit,
        "nanogpt_commit": nanogpt_commit,
        "seed": seed,
        "benchmark_steps": benchmark_steps,
        "hardware": {
            "gpu": gpu_name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        },
        "dataset": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": _sha256(input_path),
        },
    }

    tests = _run(
        [
            str(REMOTE_PYTHON),
            "-m",
            "pytest",
            "-q",
            "tests/test_lct_core.py",
            "tests/test_lct_linear.py",
            "tests/test_nanogpt_patch.py",
        ],
        timeout=180,
    )
    payload["tests"] = tests

    smoke_path = remote_dir / "smoke.json"
    smoke_command = _tune_command(
        output=smoke_path,
        seed=seed,
        steps=20,
        eval_every=10,
        embed_dim=256,
        n_layers=2,
        n_heads=4,
        batch_size=8,
        seq_len=128,
        warmup_steps=2,
    )
    payload["smoke_command"] = _run(smoke_command, timeout=300)
    payload["smoke"] = _load_payload(smoke_path)
    payload["smoke_assertions"] = _assert_smoke(payload["smoke"])

    benchmark_path = remote_dir / "benchmark.json"
    benchmark_command = _tune_command(
        output=benchmark_path,
        seed=seed,
        steps=benchmark_steps,
        eval_every=max(1, benchmark_steps // 5),
        embed_dim=1024,
        n_layers=4,
        n_heads=8,
        batch_size=32,
        seq_len=256,
        warmup_steps=max(1, benchmark_steps // 10),
    )
    payload["benchmark_command"] = _run(benchmark_command, timeout=60 * 24)
    payload["benchmark"] = _load_payload(benchmark_path)
    payload["benchmark_assertions"] = _assert_smoke(payload["benchmark"])
    payload["finished_at"] = _now()

    consolidated = remote_dir / "modal_h100_learnable.json"
    consolidated.write_text(json.dumps(payload, indent=2) + "\n")
    results_volume.commit()
    payload["remote_result_path"] = str(consolidated)
    return payload


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


@app.local_entrypoint()
def main(
    output: str = "paper/results/modal_h100_learnable_s1.json",
    seed: int = 1,
    benchmark_steps: int = 500,
) -> None:
    if benchmark_steps < 20:
        raise ValueError("benchmark_steps must be at least 20")
    payload = run_h100_learnable.remote(
        seed=seed,
        benchmark_steps=benchmark_steps,
        source_commit=_git_head(ROOT),
        nanogpt_commit=_git_head(LOCAL_NANOGPT),
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
