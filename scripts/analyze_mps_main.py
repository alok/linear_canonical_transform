#!/usr/bin/env python3
"""Apply the pre-registered decision rule to the Mac MPS main-run artifacts.

See paper/experiments/mps_shakespeare_protocol.md. Reads
paper/results/mps_main_group1_s{seed}.json (baseline dim 256 + linear
configs, paired within invocation) and mps_main_matched212_s{seed}.json
(parameter-matched baseline), computes per-seed paired deltas of best-val,
and reports the rule outcomes plus wall-clock context.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_runs(result_dir: Path, seeds: list[int]) -> dict[str, dict[int, dict]]:
    """-> {config_name: {seed: trial_record}}"""
    runs: dict[str, dict[int, dict]] = {}
    for seed in seeds:
        g1 = json.loads((result_dir / f"mps_main_group1_s{seed}.json").read_text())
        for r in g1["results"]:
            runs.setdefault(r["name"], {})[seed] = r
        g2 = json.loads((result_dir / f"mps_main_matched212_s{seed}.json").read_text())
        for r in g2["results"]:
            runs.setdefault("matched-baseline-212", {})[seed] = r
    return runs


def paired_deltas(
    runs: dict[str, dict[int, dict]], config: str, control: str, seeds: list[int]
) -> list[float]:
    return [
        runs[config][s]["best_val_loss"] - runs[control][s]["best_val_loss"]
        for s in seeds
    ]


def fmt_deltas(deltas: list[float]) -> str:
    per_seed = " ".join(f"{d:+.4f}" for d in deltas)
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else float("nan")
    return f"mean {mean:+.4f} sd {sd:.4f} per-seed [{per_seed}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("paper/results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    runs = load_runs(args.result_dir, args.seeds)
    configs = sorted(runs)
    lct_configs = [c for c in configs if c.startswith(("linear", "hybrid", "activation"))]

    print("== Per-config summary (best-val over deterministic full-val curve) ==")
    for name in configs:
        by_seed = runs[name]
        bests = [by_seed[s]["best_val_loss"] for s in args.seeds]
        finals = [by_seed[s]["final_val_loss"] for s in args.seeds]
        tps = statistics.mean(by_seed[s]["tokens_per_second"] for s in args.seeds)
        params = by_seed[args.seeds[0]]["parameter_count"]
        print(
            f"  {name:22s} params={params:>9,} tok/s={tps:>8,.0f} "
            f"best-val mean {statistics.mean(bests):.4f} sd {statistics.stdev(bests):.4f} "
            f"(final mean {statistics.mean(finals):.4f})"
        )

    print("\n== Decision rule ==")
    for config in lct_configs:
        print(f"\n  {config}:")
        for control in ("matched-baseline-212", "baseline"):
            deltas = paired_deltas(runs, config, control, args.seeds)
            mean = statistics.mean(deltas)
            consistent = all(d < 0 for d in deltas) or all(d > 0 for d in deltas)
            wins = mean < -args.threshold and all(d < 0 for d in deltas)
            label = (
                "REAL IMPROVEMENT" if wins
                else ("consistent regression" if consistent and mean > args.threshold
                      else "no detectable effect")
            )
            print(f"    vs {control:22s} {fmt_deltas(deltas)} -> {label}")

    print("\n== Wall-clock context (train-only, eval excluded) ==")
    base_tps = statistics.mean(runs["baseline"][s]["tokens_per_second"] for s in args.seeds)
    for config in lct_configs:
        tps = statistics.mean(runs[config][s]["tokens_per_second"] for s in args.seeds)
        print(f"  {config:22s} {tps:>8,.0f} tok/s = {tps / base_tps:.2f}x baseline throughput")

    print("\n== Val-vs-step curves (mean over seeds) ==")
    steps = [row[0] for row in runs["baseline"][args.seeds[0]]["val_history"]]
    header = "  step  " + "  ".join(f"{name[:14]:>14s}" for name in configs)
    print(header)
    for i, step in enumerate(steps):
        row = [
            statistics.mean(runs[name][s]["val_history"][i][1] for s in args.seeds)
            if i < len(runs[name][args.seeds[0]]["val_history"]) else float("nan")
            for name in configs
        ]
        print(f"  {step:>5d} " + "  ".join(f"{v:>14.4f}" for v in row))


if __name__ == "__main__":
    main()
