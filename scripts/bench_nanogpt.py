#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lct_activation.integrations.nanogpt import (  # noqa: E402
    DEFAULT_LOCAL_NANOGPT_REPO,
    build_local_nanogpt,
    build_upstream_nanogpt,
    infer_nanogpt_repo_kind,
    make_lct_activation_factory,
    make_lct_linear_factory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark baseline and LCT NanoGPT variants on random tokens."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_LOCAL_NANOGPT_REPO,
        help="NanoGPT checkout to benchmark. Defaults to the local /Users/alokbeniwal/nanogpt repo.",
    )
    parser.add_argument(
        "--repo-kind",
        choices=("auto", "local", "upstream"),
        default="auto",
        help="Repo layout to use. 'auto' infers from the checkout.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device spec. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--vocab-size", type=int, default=65)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "activation", "linear"],
        choices=["baseline", "activation", "linear", "hybrid"],
        help="Model variants to benchmark.",
    )
    parser.add_argument("--a", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=0.0)
    parser.add_argument("--bias-init", type=float, default=0.1)
    parser.add_argument("--residual-mix", type=float, default=0.0)
    parser.add_argument(
        "--inverse-after-multiply",
        dest="inverse_after_multiply",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether LCTLinear should map back with the inverse transform after spectral mixing.",
    )
    parser.add_argument(
        "--bias",
        dest="bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to request bias parameters for upstream NanoGPT configs.",
    )
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    *,
    repo_dir: Path,
    repo_kind: str,
    variant: str,
    device: torch.device,
    seed: int,
    warmup_steps: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    embed_dim: int,
    dropout: float,
    vocab_size: int,
    bias: bool,
    activation_factory,
    linear_factory,
) -> dict[str, object]:
    torch.manual_seed(seed)

    if repo_kind == "local":
        model, _namespace = build_local_nanogpt(
            repo_dir,
            variant=variant,
            activation_factory=activation_factory,
            linear_factory=linear_factory,
            batch_size=batch_size,
            ctx_len=seq_len,
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            drop_frac=dropout,
            vocab_size=vocab_size,
            device=device,
        )
    elif repo_kind == "upstream":
        model, _model_module = build_upstream_nanogpt(
            repo_dir,
            variant=variant,
            activation_factory=activation_factory,
            linear_factory=linear_factory,
            block_size=seq_len,
            vocab_size=vocab_size,
            n_layer=n_layers,
            n_head=n_heads,
            n_embd=embed_dim,
            dropout=dropout,
            bias=bias,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported repo kind: {repo_kind}")

    model.eval()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    with torch.inference_mode():
        for _ in range(warmup_steps):
            model(input_ids, targets)

        sync_device(device)
        start = time.perf_counter()
        for _ in range(steps):
            model(input_ids, targets)
        sync_device(device)
        elapsed = time.perf_counter() - start

    total_tokens = batch_size * seq_len * steps
    return {
        "variant": variant,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "parameter_count": sum(param.numel() for param in model.parameters()),
    }


def main() -> None:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    repo_kind = infer_nanogpt_repo_kind(repo_dir) if args.repo_kind == "auto" else args.repo_kind
    device = resolve_device(args.device)
    activation_factory = make_lct_activation_factory(
        a=args.a,
        b=args.b,
        c=args.c,
        bias_init=args.bias_init,
        residual_mix=args.residual_mix,
    )
    linear_factory = make_lct_linear_factory(
        a=args.a,
        b=args.b,
        c=args.c,
        inverse_after_multiply=args.inverse_after_multiply,
    )

    results = []
    for idx, variant in enumerate(args.variants):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        results.append(
            benchmark_once(
                repo_dir=repo_dir,
                repo_kind=repo_kind,
                variant=variant,
                device=device,
                seed=args.seed + idx * 1000,
                warmup_steps=args.warmup_steps,
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                embed_dim=args.embed_dim,
                dropout=args.dropout,
                vocab_size=args.vocab_size,
                bias=args.bias,
                activation_factory=activation_factory,
                linear_factory=linear_factory,
            )
        )

    summary = {
        "repo_dir": str(repo_dir),
        "repo_kind": repo_kind,
        "device": str(device),
        "results": results,
    }
    baseline = next((item for item in results if item["variant"] == "baseline"), None)
    if baseline is not None:
        summary["speed_ratios_vs_baseline"] = {
            item["variant"]: item["tokens_per_second"] / baseline["tokens_per_second"]
            for item in results
            if item["variant"] != "baseline"
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
