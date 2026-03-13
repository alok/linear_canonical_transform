#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lct_activation.integrations.nanogpt import (  # noqa: E402
    DEFAULT_UPSTREAM_NANOGPT_REPO,
    DEFAULT_UPSTREAM_NANOGPT_URL,
    infer_nanogpt_repo_kind,
    make_lct_activation_factory,
    make_lct_linear_factory,
    run_upstream_train,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Patch NanoGPT's MLP activation with nonlinear LCT and run upstream train.py in-process."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_UPSTREAM_NANOGPT_REPO,
        help="Upstream NanoGPT checkout containing train.py/model.py.",
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_UPSTREAM_NANOGPT_URL,
        help="Repository URL used when --clone-if-missing is set.",
    )
    parser.add_argument(
        "--clone-if-missing",
        action="store_true",
        help="Clone the upstream NanoGPT repository before training if it is missing.",
    )
    parser.add_argument(
        "--variant",
        choices=("activation", "linear", "hybrid"),
        default="activation",
        help="Which LCT patch to apply to NanoGPT's MLP block.",
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
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are forwarded to NanoGPT train.py.",
    )
    args = parser.parse_args()
    train_args = args.train_args
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    return args, train_args


def main() -> None:
    args, train_args = parse_args()
    repo_dir = args.repo_dir.resolve()

    if repo_dir.exists():
        repo_kind = infer_nanogpt_repo_kind(repo_dir)
        if repo_kind != "upstream":
            raise SystemExit(
                f"{repo_dir} is a local/source-sliced NanoGPT variant, not an upstream train.py checkout. "
                "Use an upstream repo for training, or run scripts/bench_nanogpt.py for the local benchmark path."
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


if __name__ == "__main__":
    main()
