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

    run_upstream_train(
        repo_dir=repo_dir,
        train_argv=train_args,
        clone_if_missing=args.clone_if_missing,
        repo_url=args.repo_url,
    )


if __name__ == "__main__":
    main()
