#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from lct_activation.integrations.nanogpt import (
    DEFAULT_LOCAL_NANOGPT_REPO,
    build_local_nanogpt,
    infer_nanogpt_repo_kind,
    make_lct_activation_factory,
    make_lct_linear_factory,
)


@dataclass
class TrialSpec:
    name: str
    variant: str
    activation_kwargs: dict[str, float | bool | int]
    linear_kwargs: dict[str, float | bool | int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small NanoGPT ablation sweep for baseline vs LCT variants."
    )
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_LOCAL_NANOGPT_REPO)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--vocab-size", type=int, default=65)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=[
            "baseline",
            "activation-fourier",
            "activation-frft45",
            "linear-fourier",
            "linear-frft45",
            "hybrid-fourier",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/results/nanogpt_local_tune.json"),
    )
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def trial_spec(name: str) -> TrialSpec:
    frft45 = math.pi / 4.0
    frft_a = math.cos(frft45)
    frft_b = math.sin(frft45)
    frft_c = -frft_b

    presets: dict[str, TrialSpec] = {
        "baseline": TrialSpec("baseline", "baseline", {}, {}),
        "activation-fourier": TrialSpec(
            "activation-fourier",
            "activation",
            {"a": 0.0, "b": 1.0, "c": 0.0, "bias_init": 0.1, "residual_mix": 0.0, "dense_threshold": 128},
            {},
        ),
        "activation-frft45": TrialSpec(
            "activation-frft45",
            "activation",
            {"a": frft_a, "b": frft_b, "c": frft_c, "bias_init": 0.1, "residual_mix": 0.1, "dense_threshold": 128},
            {},
        ),
        "linear-fourier": TrialSpec(
            "linear-fourier",
            "linear",
            {},
            {"a": 0.0, "b": 1.0, "c": 0.0, "dense_threshold": 32, "inverse_after_multiply": True},
        ),
        "linear-frft45": TrialSpec(
            "linear-frft45",
            "linear",
            {},
            {"a": frft_a, "b": frft_b, "c": frft_c, "dense_threshold": 32, "inverse_after_multiply": True},
        ),
        "hybrid-fourier": TrialSpec(
            "hybrid-fourier",
            "hybrid",
            {"a": 0.0, "b": 1.0, "c": 0.0, "bias_init": 0.1, "residual_mix": 0.0, "dense_threshold": 128},
            {"a": 0.0, "b": 1.0, "c": 0.0, "dense_threshold": 32, "inverse_after_multiply": True},
        ),
    }
    if name not in presets:
        raise KeyError(f"Unknown preset: {name}")
    return presets[name]


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, get_batch, split: str, eval_iters: int, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_iters):
        xb, yb = get_batch(split)
        xb = xb.to(device)
        yb = yb.to(device)
        _logits, loss = model(xb, yb)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def run_trial(args: argparse.Namespace, spec: TrialSpec, *, device: torch.device, seed_offset: int) -> dict[str, object]:
    torch.manual_seed(args.seed + seed_offset)

    activation_factory = make_lct_activation_factory(**spec.activation_kwargs)
    linear_factory = make_lct_linear_factory(**spec.linear_kwargs)

    model, namespace = build_local_nanogpt(
        args.repo_dir,
        variant=spec.variant,
        activation_factory=activation_factory,
        linear_factory=linear_factory,
        batch_size=args.batch_size,
        ctx_len=args.seq_len,
        n_heads=args.n_heads,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        drop_frac=args.dropout,
        vocab_size=args.vocab_size,
        device=device,
    )

    get_batch = namespace["get_batch"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    initial_val_loss = evaluate_loss(model, get_batch, "val", args.eval_iters, device)

    model.train()
    sync_device(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        xb, yb = get_batch("train")
        xb = xb.to(device)
        yb = yb.to(device)
        _logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    sync_device(device)
    elapsed = time.perf_counter() - start

    final_train_loss = evaluate_loss(model, get_batch, "train", args.eval_iters, device)
    final_val_loss = evaluate_loss(model, get_batch, "val", args.eval_iters, device)

    return {
        "name": spec.name,
        "variant": spec.variant,
        "activation_kwargs": spec.activation_kwargs,
        "linear_kwargs": spec.linear_kwargs,
        "initial_val_loss": initial_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "steps": args.steps,
        "tokens_per_second": (args.batch_size * args.seq_len * args.steps) / elapsed,
        "parameter_count": sum(param.numel() for param in model.parameters()),
    }


def main() -> None:
    args = parse_args()
    args.repo_dir = args.repo_dir.resolve()
    if infer_nanogpt_repo_kind(args.repo_dir) != "local":
        raise SystemExit("tune_nanogpt_lct.py currently expects the local /Users/alokbeniwal/nanogpt layout.")

    device = resolve_device(args.device)
    specs = [trial_spec(name) for name in args.presets]
    results = [run_trial(args, spec, device=device, seed_offset=i * 1000) for i, spec in enumerate(specs)]
    results.sort(key=lambda item: float(item["final_val_loss"]))

    payload = {
        "device": str(device),
        "repo_dir": str(args.repo_dir),
        "config": {
            "steps": args.steps,
            "eval_iters": args.eval_iters,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "embed_dim": args.embed_dim,
            "dropout": args.dropout,
            "vocab_size": args.vocab_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
