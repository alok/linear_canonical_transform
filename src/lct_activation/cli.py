from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .integrations.nanogpt import (
    DEFAULT_LOCAL_NANOGPT_REPO,
    DEFAULT_UPSTREAM_NANOGPT_REPO,
    DEFAULT_UPSTREAM_NANOGPT_URL,
    build_local_nanogpt,
    build_upstream_nanogpt,
    infer_nanogpt_repo_kind,
    make_lct_activation_factory,
    make_lct_linear_factory,
    run_upstream_train,
)
from .layers import LCTLinear


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _maybe_compile(module: torch.nn.Module, *, enabled: bool, device: torch.device, mode: str) -> torch.nn.Module:
    if not enabled or not hasattr(torch, "compile"):
        return module
    if device.type != "cuda":
        return module
    return torch.compile(module, mode=mode)


def _benchmark_module(module: torch.nn.Module, x: torch.Tensor, *, steps: int, warmup_steps: int) -> float:
    for _ in range(warmup_steps):
        module(x)
    _sync_device(x.device)

    start = time.perf_counter()
    for _ in range(steps):
        module(x)
    _sync_device(x.device)
    end = time.perf_counter()
    return (end - start) / steps * 1_000.0


def parse_bench_linear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark nn.Linear against LCTLinear on random inputs.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("forward", "train"), default="forward")
    parser.add_argument("--compile", dest="compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    return parser.parse_args()


def bench_linear_main() -> None:
    args = parse_bench_linear_args()
    device = _resolve_device(args.device)

    x = torch.randn(args.batch_size, args.in_features, device=device)
    dense = torch.nn.Linear(args.in_features, args.out_features).to(device)
    lct = LCTLinear(args.in_features, args.out_features).to(device)
    dense = _maybe_compile(dense, enabled=args.compile, device=device, mode=args.compile_mode)
    lct = _maybe_compile(lct, enabled=args.compile, device=device, mode=args.compile_mode)

    if args.mode == "forward":
        dense_ms = _benchmark_module(dense, x, steps=args.steps, warmup_steps=args.warmup_steps)
        lct_ms = _benchmark_module(lct, x, steps=args.steps, warmup_steps=args.warmup_steps)
    else:
        target = torch.randn(args.batch_size, args.out_features, device=device)

        def bench_train_step(module: torch.nn.Module) -> float:
            params = [param for param in module.parameters() if param.requires_grad]
            for _ in range(args.warmup_steps):
                out = module(x)
                loss = (out - target).square().mean()
                for param in params:
                    param.grad = None
                loss.backward()
            _sync_device(device)

            start = time.perf_counter()
            for _ in range(args.steps):
                out = module(x)
                loss = (out - target).square().mean()
                for param in params:
                    param.grad = None
                loss.backward()
            _sync_device(device)
            end = time.perf_counter()
            return (end - start) / args.steps * 1_000.0

        dense_ms = bench_train_step(dense)
        lct_ms = bench_train_step(lct)

    print(
        {
            "device": str(device),
            "batch_size": args.batch_size,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "mode": args.mode,
            "compiled": bool(args.compile and device.type == "cuda"),
            "dense_ms": round(dense_ms, 4),
            "lct_ms": round(lct_ms, 4),
            "lct_over_dense": round(lct_ms / dense_ms, 4) if dense_ms else None,
        }
    )


def parse_bench_nanogpt_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline and LCT NanoGPT variants on random tokens.")
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_LOCAL_NANOGPT_REPO)
    parser.add_argument("--repo-kind", choices=("auto", "local", "upstream"), default="auto")
    parser.add_argument("--device", default="auto")
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
        "--bias",
        dest="bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to request bias parameters for upstream NanoGPT configs.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "activation", "linear"],
        choices=["baseline", "activation", "linear", "hybrid"],
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
    )
    parser.add_argument("--compile", dest="compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    return parser.parse_args()


def _benchmark_nanogpt_once(
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
    compile_enabled: bool,
    compile_mode: str,
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
        model, _module = build_upstream_nanogpt(
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

    model = _maybe_compile(model, enabled=compile_enabled, device=device, mode=compile_mode)

    model.eval()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    with torch.inference_mode():
        for _ in range(warmup_steps):
            model(input_ids, targets)
        _sync_device(device)
        start = time.perf_counter()
        for _ in range(steps):
            model(input_ids, targets)
        _sync_device(device)
        elapsed = time.perf_counter() - start

    total_tokens = batch_size * seq_len * steps
    return {
        "variant": variant,
        "compiled": bool(compile_enabled and device.type == "cuda"),
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "parameter_count": sum(param.numel() for param in model.parameters()),
    }


def bench_nanogpt_main() -> None:
    args = parse_bench_nanogpt_args()
    repo_dir = args.repo_dir.resolve()
    repo_kind = infer_nanogpt_repo_kind(repo_dir) if args.repo_kind == "auto" else args.repo_kind
    device = _resolve_device(args.device)
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
            _benchmark_nanogpt_once(
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
                compile_enabled=args.compile,
                compile_mode=args.compile_mode,
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


def parse_train_nanogpt_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Patch NanoGPT's MLP block with LCT variants and run upstream train.py in-process."
    )
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_UPSTREAM_NANOGPT_REPO)
    parser.add_argument("--repo-url", default=DEFAULT_UPSTREAM_NANOGPT_URL)
    parser.add_argument("--clone-if-missing", action="store_true")
    parser.add_argument("--variant", choices=("activation", "linear", "hybrid"), default="activation")
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
    )
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    train_args = args.train_args
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    return args, train_args


def train_nanogpt_main() -> None:
    args, train_args = parse_train_nanogpt_args()
    repo_dir = args.repo_dir.resolve()

    if repo_dir.exists():
        repo_kind = infer_nanogpt_repo_kind(repo_dir)
        if repo_kind != "upstream":
            raise SystemExit(
                f"{repo_dir} is a local/source-sliced NanoGPT variant, not an upstream train.py checkout."
            )

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

    run_upstream_train(
        repo_dir=repo_dir,
        train_argv=train_args,
        clone_if_missing=args.clone_if_missing,
        repo_url=args.repo_url,
        variant=args.variant,
        activation_factory=activation_factory,
        linear_factory=linear_factory,
    )


@dataclass
class TrialSpec:
    name: str
    variant: str
    activation_kwargs: dict[str, float | bool | int]
    linear_kwargs: dict[str, float | bool | int]


def _default_trial_specs(name: str) -> TrialSpec:
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


def _angle_trial_spec(angle_deg: float) -> TrialSpec:
    theta = math.radians(angle_deg)
    return TrialSpec(
        f"linear-frft{int(angle_deg):02d}",
        "linear",
        {},
        {
            "a": math.cos(theta),
            "b": math.sin(theta),
            "c": -math.sin(theta),
            "dense_threshold": 32,
            "inverse_after_multiply": True,
        },
    )


def parse_tune_nanogpt_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small NanoGPT ablation sweep for baseline vs LCT variants.")
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
    parser.add_argument("--compile", dest="compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
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
        "--linear-angle-degrees",
        nargs="*",
        type=float,
        default=[],
        help="Extra FRFT angles in degrees to evaluate as linear-only variants.",
    )
    parser.add_argument("--output", type=Path, default=Path("paper/results/nanogpt_local_tune.json"))
    return parser.parse_args()


@torch.no_grad()
def _evaluate_loss(model: torch.nn.Module, get_batch, split: str, eval_iters: int, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_iters):
        xb, yb = get_batch(split)
        xb = xb.to(device)
        yb = yb.to(device)
        _logits, loss = model(xb, yb)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def _run_tune_trial(args: argparse.Namespace, spec: TrialSpec, *, device: torch.device, seed_offset: int) -> dict[str, object]:
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
    model = _maybe_compile(model, enabled=args.compile, device=device, mode=args.compile_mode)

    get_batch = namespace["get_batch"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    initial_val_loss = _evaluate_loss(model, get_batch, "val", args.eval_iters, device)

    model.train()
    _sync_device(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        xb, yb = get_batch("train")
        xb = xb.to(device)
        yb = yb.to(device)
        _logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    _sync_device(device)
    elapsed = time.perf_counter() - start

    final_train_loss = _evaluate_loss(model, get_batch, "train", args.eval_iters, device)
    final_val_loss = _evaluate_loss(model, get_batch, "val", args.eval_iters, device)

    return {
        "name": spec.name,
        "variant": spec.variant,
        "compiled": bool(args.compile and device.type == "cuda"),
        "activation_kwargs": spec.activation_kwargs,
        "linear_kwargs": spec.linear_kwargs,
        "initial_val_loss": initial_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "steps": args.steps,
        "tokens_per_second": (args.batch_size * args.seq_len * args.steps) / elapsed,
        "parameter_count": sum(param.numel() for param in model.parameters()),
    }


def tune_nanogpt_main() -> None:
    args = parse_tune_nanogpt_args()
    args.repo_dir = args.repo_dir.resolve()
    if infer_nanogpt_repo_kind(args.repo_dir) != "local":
        raise SystemExit("lct-tune-nanogpt currently expects the local /Users/alokbeniwal/nanogpt layout.")

    device = _resolve_device(args.device)
    specs = [_default_trial_specs(name) for name in args.presets]
    specs.extend(_angle_trial_spec(angle) for angle in args.linear_angle_degrees)
    results = [_run_tune_trial(args, spec, device=device, seed_offset=i * 1000) for i, spec in enumerate(specs)]
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
