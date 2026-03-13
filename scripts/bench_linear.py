#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch

from lct_activation import LCTLinear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark nn.Linear against LCTLinear on random inputs."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_module(module: torch.nn.Module, x: torch.Tensor, *, steps: int, warmup_steps: int) -> float:
    for _ in range(warmup_steps):
        module(x)
    synchronize(x.device)

    start = time.perf_counter()
    for _ in range(steps):
        module(x)
    synchronize(x.device)
    end = time.perf_counter()
    return (end - start) / steps * 1_000.0


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    x = torch.randn(args.batch_size, args.in_features, device=device)
    dense = torch.nn.Linear(args.in_features, args.out_features).to(device)
    lct = LCTLinear(args.in_features, args.out_features).to(device)

    dense_ms = benchmark_module(dense, x, steps=args.steps, warmup_steps=args.warmup_steps)
    lct_ms = benchmark_module(lct, x, steps=args.steps, warmup_steps=args.warmup_steps)

    print(
        {
            "device": str(device),
            "batch_size": args.batch_size,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "dense_ms": round(dense_ms, 4),
            "lct_ms": round(lct_ms, 4),
            "lct_over_dense": round(lct_ms / dense_ms, 4) if dense_ms else None,
        }
    )


if __name__ == "__main__":
    main()
