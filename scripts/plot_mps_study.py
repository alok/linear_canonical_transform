#!/usr/bin/env python3
"""Render the MPS NanoGPT study figure from checked-in artifacts.

Run with an ephemeral matplotlib (not a project dependency):

    uv run --with matplotlib python scripts/plot_mps_study.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("paper/results")
OUT = Path("paper/figures/mps_study.png")

# Okabe-Ito palette, fixed per entity across panels (color follows the entity).
COLORS = {
    "baseline-256": "#000000",
    "matched-212": "#E69F00",
    "linear-fourier": "#0072B2",
    "linear-frft30": "#56B4E9",
    "linear-frft45": "#009E73",
    "activation-fourier": "#CC79A7",
    "baseline-172": "#D55E00",
}


def curve(record: dict) -> tuple[list[int], list[float]]:
    rows = record["val_history"]
    return [row[0] for row in rows], [row[1] for row in rows]


def mean_curve(records: list[dict]) -> tuple[list[int], list[float]]:
    steps = [row[0] for row in records[0]["val_history"]]
    values = [
        statistics.mean(r["val_history"][i][1] for r in records)
        for i in range(len(steps))
    ]
    return steps, values


def main() -> None:
    seeds = [1, 2, 3, 4]
    group1 = {
        seed: json.loads((RESULTS / f"mps_main_group1_s{seed}.json").read_text())["results"]
        for seed in seeds
    }
    matched = [
        json.loads((RESULTS / f"mps_main_matched212_s{seed}.json").read_text())["results"][0]
        for seed in seeds
    ]
    by_name = lambda name: [next(r for r in group1[s] if r["name"] == name) for s in seeds]

    extended = {
        r["name"]: r
        for r in json.loads((RESULTS / "mps_extended_s1.json").read_text())["results"]
    }
    ext_matched = json.loads(
        (RESULTS / "mps_extended_matched212_s1.json").read_text()
    )["results"][0]
    healthy = {
        r["name"]: r
        for r in json.loads((RESULTS / "mps_healthy212_s1.json").read_text())["results"]
    }
    matched172 = json.loads((RESULTS / "mps_matched172_s1.json").read_text())["results"][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    panel1 = [
        ("baseline-256", mean_curve(by_name("baseline")), "-"),
        ("matched-212", mean_curve(matched), "-"),
        ("linear-fourier", mean_curve(by_name("linear-fourier")), "-"),
        ("linear-frft30", mean_curve(by_name("linear-frft30")), "-"),
        ("linear-frft45", mean_curve(by_name("linear-frft45")), "-"),
    ]
    for name, (steps, values), style in panel1:
        ax1.plot(steps, values, style, color=COLORS[name], lw=1.8, label=name)
    ax1.set_title("(a) Main: mean of 4 paired seeds, 2000 steps", fontsize=10)

    panel2 = [
        ("baseline-256", curve(extended["baseline"]), "-"),
        ("matched-212", curve(ext_matched), "-"),
        ("linear-fourier", curve(extended["linear-fourier"]), "-"),
        ("activation-fourier", curve(extended["activation-fourier"]), "-"),
        ("baseline-172", curve(matched172), "-"),
        ("linear-fourier", curve(healthy["linear-fourier"]), "--"),
    ]
    labels = {"linear-fourier--": "linear-fourier@212"}
    for name, (steps, values), style in panel2:
        label = labels.get(name + style, name)
        ax2.plot(steps, values, style, color=COLORS[name], lw=1.8, label=label)
    ax2.set_title("(b) Extended horizon: seed 1, 5000 steps", fontsize=10)

    for ax in (ax1, ax2):
        ax.set_xlabel("training step")
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    ax1.set_ylabel("val loss (deterministic full-split, nats/char)")

    fig.suptitle(
        "LCT layers in NanoGPT on tinyshakespeare (Apple-silicon MPS): "
        "paired evidence and horizon effects",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT}")

    # Second figure: the repaired-substrate replication.
    std_seeds = [1, 2, 3, 4]
    std = {
        s: {r["name"]: r for r in json.loads((RESULTS / f"std_main_group1_s{s}.json").read_text())["results"]}
        for s in std_seeds
    }
    std_matched = [
        json.loads((RESULTS / f"std_main_matched212_s{s}.json").read_text())["results"][0]
        for s in std_seeds
    ]
    ext = {
        s: {r["name"]: r for r in json.loads((RESULTS / f"std_extended_s{s}.json").read_text())["results"]}
        for s in (1, 2)
    }

    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    COLORS["lowrank-mlp"] = "#D55E00"
    COLORS["baseline"] = COLORS["baseline-256"]
    for name in ("baseline", "linear-fourier", "activation-fourier", "lowrank-mlp"):
        steps, values = mean_curve([std[s][name] for s in std_seeds])
        bx1.plot(steps, values, "-", color=COLORS.get(name, "#999999"), lw=1.8, label=name)
    steps, values = mean_curve(std_matched)
    bx1.plot(steps, values, "-", color=COLORS["matched-212"], lw=1.8, label="matched-212")
    bx1.set_title("(a) Repaired substrate: mean of 4 paired seeds, 2000 steps", fontsize=10)
    for name in ("baseline", "linear-fourier", "activation-fourier", "lowrank-mlp"):
        steps, values = mean_curve([ext[s][name] for s in (1, 2)])
        bx2.plot(steps, values, "-", color=COLORS.get(name, "#999999"), lw=1.8, label=name)
    bx2.set_title("(b) Extended horizon: mean of 2 paired seeds, 5000 steps", fontsize=10)
    for ax in (bx1, bx2):
        ax.set_xlabel("training step")
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    bx1.set_ylabel("val loss (deterministic full-split, nats/char)")
    fig2.suptitle(
        "Repaired substrate (attention scaling, warmup+cosine, lr 5e-3): "
        "the negative replicates; structure beats rank-1; horizon narrows the gap",
        fontsize=11,
    )
    fig2.tight_layout()
    out2 = OUT.parent / "std_study.png"
    fig2.savefig(out2, dpi=180)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
