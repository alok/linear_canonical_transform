#!/usr/bin/env python3
"""Benchmark LCT layers locally on a Mac: torch CPU, torch MPS, and MLX.

Compares the LCT activation and structured linear layer against their
conventional baselines (GELU, dense Linear) on every backend available on
this machine, for forward and forward+backward passes.

Usage:

    uv run python scripts/bench_mac_local.py
    uv run python scripts/bench_mac_local.py --dims 1024 4096 --steps 50 \
        --output paper/results/bench_mac_local.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from lct_activation import LCTActivation, LCTLinear

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn

    import lct_activation.mlx as mlx_lct

    HAS_MLX = True
except ImportError:  # pragma: no cover - benchmark guard
    HAS_MLX = False
    mx = mlx_nn = mlx_lct = None  # type: ignore[assignment]


def _median_ms(fn: Callable[[], None], *, steps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(steps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _torch_sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":  # pragma: no cover - not expected locally
        torch.cuda.synchronize()


def bench_torch_module(
    module: nn.Module,
    x: torch.Tensor,
    *,
    steps: int,
    warmup: int,
) -> tuple[float, float]:
    device = x.device
    module = module.to(device)

    def fwd() -> None:
        with torch.no_grad():
            module(x)
        _torch_sync(device)

    def fwd_bwd() -> None:
        module.zero_grad(set_to_none=True)
        y = module(x_grad)
        y.square().mean().backward()
        _torch_sync(device)

    x_grad = x.clone().requires_grad_(True)
    fwd_ms = _median_ms(fwd, steps=steps, warmup=warmup)
    fwd_bwd_ms = _median_ms(fwd_bwd, steps=steps, warmup=warmup)
    return fwd_ms, fwd_bwd_ms


def bench_mlx_module(module, x, *, steps: int, warmup: int) -> tuple[float, float]:
    from mlx.utils import tree_flatten

    def fwd() -> None:
        mx.eval(module(x))

    # Differentiate w.r.t. the input always (like torch's backward through
    # x.requires_grad_) and w.r.t. parameters when the module has any.
    params = module.trainable_parameters()
    has_params = len(tree_flatten(params)) > 0

    def loss_fn(p, inp):
        if has_params:
            module.update(p)
        return mx.mean(mx.square(module(inp)))

    argnums = (0, 1) if has_params else (1,)
    value_and_grad = mx.value_and_grad(loss_fn, argnums=argnums)

    def fwd_bwd() -> None:
        out = value_and_grad(params, x)
        mx.eval(out)

    fwd_ms = _median_ms(fwd, steps=steps, warmup=warmup)
    fwd_bwd_ms = _median_ms(fwd_bwd, steps=steps, warmup=warmup)
    return fwd_ms, fwd_bwd_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dims", type=int, nargs="+", default=[768, 1024, 2048, 4096])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["torch-cpu", "torch-mps", "mlx"],
        choices=["torch-cpu", "torch-mps", "mlx"],
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = args.batch_size * args.seq_len
    results: list[dict[str, object]] = []

    frameworks = list(args.frameworks)
    if "torch-mps" in frameworks and not torch.backends.mps.is_available():
        print("torch-mps requested but MPS is unavailable; skipping")
        frameworks.remove("torch-mps")
    if "mlx" in frameworks and not HAS_MLX:
        print("mlx requested but not installed; skipping")
        frameworks.remove("mlx")

    for dim in args.dims:
        for framework in frameworks:
            if framework.startswith("torch"):
                device = torch.device("cpu" if framework == "torch-cpu" else "mps")
                torch.manual_seed(0)
                x_act = torch.randn(args.batch_size, args.seq_len, dim, device=device)
                x_lin = torch.randn(tokens, dim, device=device)
                modules: list[tuple[str, nn.Module, torch.Tensor]] = [
                    ("gelu", nn.GELU(), x_act),
                    ("lct-activation", LCTActivation(dim), x_act),
                    ("linear", nn.Linear(dim, dim), x_lin),
                    ("lct-linear", LCTLinear(dim, dim), x_lin),
                ]
                for name, module, x in modules:
                    fwd_ms, fwd_bwd_ms = bench_torch_module(
                        module, x, steps=args.steps, warmup=args.warmup_steps
                    )
                    results.append(
                        {
                            "name": f"{framework}/{name}/dim{dim}",
                            "framework": framework,
                            "module": name,
                            "dim": dim,
                            "fwd_ms": fwd_ms,
                            "fwd_bwd_ms": fwd_bwd_ms,
                            "tokens_per_second": tokens / (fwd_ms / 1e3),
                        }
                    )
            else:
                mx.random.seed(0)
                x_act = mx.random.normal((args.batch_size, args.seq_len, dim))
                x_lin = mx.random.normal((tokens, dim))
                mx.eval(x_act, x_lin)
                mlx_modules = [
                    ("gelu", mlx_nn.GELU(), x_act),
                    ("lct-activation", mlx_lct.LCTActivation(dim), x_act),
                    ("linear", mlx_nn.Linear(dim, dim), x_lin),
                    ("lct-linear", mlx_lct.LCTLinear(dim, dim), x_lin),
                ]
                for name, module, x in mlx_modules:
                    fwd_ms, fwd_bwd_ms = bench_mlx_module(
                        module, x, steps=args.steps, warmup=args.warmup_steps
                    )
                    results.append(
                        {
                            "name": f"mlx/{name}/dim{dim}",
                            "framework": "mlx",
                            "module": name,
                            "dim": dim,
                            "fwd_ms": fwd_ms,
                            "fwd_bwd_ms": fwd_bwd_ms,
                            "tokens_per_second": tokens / (fwd_ms / 1e3),
                        }
                    )

    payload: dict[str, object] = {
        "benchmark": "bench_mac_local",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "mlx_version": mx.__version__ if HAS_MLX else None,
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "dims": args.dims,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
        },
        "results": results,
    }

    header = f"{'name':<40} {'fwd ms':>10} {'fwd+bwd ms':>12} {'tokens/s':>14}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['name']:<40} {row['fwd_ms']:>10.3f} {row['fwd_bwd_ms']:>12.3f} "
            f"{row['tokens_per_second']:>14,.0f}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
