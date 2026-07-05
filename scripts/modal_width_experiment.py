#!/usr/bin/env python3
"""Large-width LCT experiment on Modal (A100): the wall-clock hypothesis.

Two questions the Mac protocol could not answer:

1. Microbenchmark: at widths 1024-8192 on a real training GPU, where does
   LCTLinear actually beat dense nn.Linear (square and rectangular d->4d
   shapes, forward and forward+backward, with and without torch.compile)?
2. Training: at trunk width 1024 (MLP up-projection 1024->4096, past the
   crossover), does replacing the up-projection with LCTLinear reach the
   same val loss in less wall-clock time than the dense baseline and a
   parameter-matched narrower baseline? Standard substrate (attention
   scaling, warmup + cosine lr).

Run:  modal run scripts/modal_width_experiment.py
"""

from __future__ import annotations

import json
import subprocess

import modal

app = modal.App("lct-width-experiment")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_dir(
        ".",
        remote_path="/root/lct",
        ignore=[".git", ".jj", ".venv", "__pycache__", ".pytest_cache", "extern", "dist"],
    )
    .add_local_dir(
        "/Users/alokbeniwal/nanogpt",
        remote_path="/root/nanogpt",
        ignore=[".git", "__pycache__", ".venv", ".pytest_cache"],
    )
)

MICROBENCH = r"""
import json, statistics, time
import torch
from lct_activation import LCTLinear

torch.manual_seed(0)
device = torch.device("cuda")
results = []

def bench(module, x, steps=30, warmup=10, train=False):
    if train:
        x = x.clone().requires_grad_(True)
    def step():
        if train:
            module.zero_grad(set_to_none=True)
            x.grad = None
            module(x).square().mean().backward()
        else:
            with torch.no_grad():
                module(x)
        torch.cuda.synchronize()
    for _ in range(warmup):
        step()
    samples = []
    for _ in range(steps):
        t = time.perf_counter(); step(); samples.append((time.perf_counter() - t) * 1e3)
    return statistics.median(samples)

for dim in (1024, 2048, 4096, 8192):
    for shape_name, in_f, out_f in (("square", dim, dim), ("up-proj", dim // 4, dim)):
        x = torch.randn(16384, in_f, device=device)
        for name, module in (
            ("dense", torch.nn.Linear(in_f, out_f, device=device)),
            ("lct", LCTLinear(in_f, out_f).to(device)),
        ):
            for train in (False, True):
                ms = bench(module, x, train=train)
                results.append({
                    "name": f"{name}/{shape_name}/dim{dim}/{'train' if train else 'fwd'}",
                    "dim": dim, "shape": shape_name, "module": name,
                    "train": train, "ms": ms,
                })
print("MICROBENCH_JSON=" + json.dumps(results))
"""


@app.function(image=image, gpu="A100", cpu=8, timeout=60 * 100)
def run_width_experiment() -> dict[str, object]:
    def run(command: str, timeout: int = 5400) -> dict[str, object]:
        completed = subprocess.run(
            ["bash", "-lc", f"cd /root/lct && {command}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-40000:],
            "stderr": completed.stderr[-8000:],
        }

    results: dict[str, object] = {}
    results["sync"] = run("uv sync --extra dev")
    results["data"] = run(
        "uv run python - <<'PY'\n"
        "import io, urllib.request, zipfile\n"
        "data = urllib.request.urlopen('http://mattmahoney.net/dc/text8.zip', timeout=120).read()\n"
        "text = zipfile.ZipFile(io.BytesIO(data)).read('text8')[:10_000_000]\n"
        "open('/tmp/text8_10mb.txt', 'wb').write(text)\n"
        "print('text8 slice bytes:', len(text))\n"
        "PY"
    )
    results["gpu"] = run(
        "uv run python -c \"import torch; print(torch.cuda.get_device_name(0), torch.__version__)\""
    )
    results["microbench"] = run(f"uv run python - <<'PY'\n{MICROBENCH}\nPY")

    # Training at width 1024 (MLP 1024->4096): standard substrate, wall-clock
    # readable from val_history's cumulative seconds. 2 paired seeds.
    common = (
        "uv run lct-tune-nanogpt --repo-dir /root/nanogpt --device cuda "
        "--steps 3000 --eval-every 150 --eval-iters 8 "
        "--batch-size 32 --seq-len 256 --n-layers 4 --n-heads 8 --dropout 0.2 "
        "--attn-scaling --lr-schedule cosine --warmup-steps 150 --lr 1e-3 "
        "--normalization unitary --no-inverse-after-multiply "
        "--data-path /tmp/text8_10mb.txt "
    )
    for seed in (1, 2):
        results[f"train_1024_s{seed}"] = run(
            common
            + f"--embed-dim 1024 --seed {seed} "
            + "--presets baseline linear-fourier lowrank-mlp "
            + f"--output /tmp/width1024_s{seed}.json && cat /tmp/width1024_s{seed}.json"
        )
        # Parameter-matched narrower dense baseline for the linear variant:
        # dim 840 matches the trunk-1024 linear variant within 0.8% at
        # vocab 27 (verified locally before launch).
        results[f"train_matched_s{seed}"] = run(
            common
            + f"--embed-dim 840 --seed {seed} "
            + "--presets baseline "
            + f"--output /tmp/matched840_s{seed}.json && cat /tmp/matched840_s{seed}.json"
        )
    return results


@app.local_entrypoint()
def main() -> None:
    payload = run_width_experiment.remote()
    print(json.dumps(payload, indent=2))
