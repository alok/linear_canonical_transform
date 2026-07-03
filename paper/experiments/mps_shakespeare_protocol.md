# Is the LCT layer a real NanoGPT improvement? Mac-local protocol

Pre-registered protocol for the question that matters: do LCT layers improve
a small NanoGPT on real data, under fair controls, on this machine — and can
they be tuned to?

Written before the main runs; the decision rule below is fixed. Pilot results
select configurations only and are never quoted as evidence.

## Setup

- Hardware: Apple-silicon Mac, torch MPS.
- Data: char-level tinyshakespeare (`/Users/alokbeniwal/nanogpt/input.txt`,
  1,115,394 chars, vocab 65, contiguous 90/10 train/val split, no leakage —
  verified).
- Model: local nanogpt (pre-LN, ReLU MLP, learned positional embeddings,
  no attention `1/sqrt(d)` scaling — a known nonstandard quirk shared by all
  variants; limits external validity, not internal comparisons).
- Reference config: 4 layers, 4 heads, dim 256, seq 256, batch 64,
  dropout 0.2, AdamW, constant lr, weight decay 0. 3,258,433 params.
- Harness: `lct-tune-nanogpt` after commits `48b9046` (frozen-activation fix)
  and `803ad1d` (real seeds, paired batch streams, deterministic full-val
  eval). Val loss = deterministic mean over all 435 non-overlapping ctx-256
  windows of the val split.

## Variants

- `baseline`: unmodified model.
- `linear-*`: each MLP up-projection (256 -> 1024 dense, 263k params) replaced
  by `LCTLinear` (2,051 params). Total 2,213,965 params (-32%).
- `activation-*`: MLP ReLU replaced by `LCTModReLU`. Total 3,260,501 params.
- `hybrid-*`: both. `-fourier` = (a,b,c) = (0,1,0); `-frftNN` = frft angle.
- `matched-baseline`: unmodified model at dim 212 = 2,199,777 params, within
  0.65% of the linear variant. This is the **primary control** for the linear
  family: at ~30 epochs the 32% capacity cut alone delays overfitting, so
  beating only the dim-256 baseline at final step would be confounded.

## Pilots (selection only)

- Pilot A (lr): lr in {1e-4, 3e-4, 1e-3, 3e-3} x all variant families,
  300 steps. Pick each family's best lr; if two lrs are within 0.02 nats,
  carry both to a 1-seed 2000-step tiebreak.
- Pilot B (tuning): on the top-2 LCT families at their best lr: frft angle
  {15, 30, 45}, normalization {unitary, compositional}, inverse-after-multiply
  {on, off}. 300 steps. Select top-2 configs overall.

## Main experiment

Configs: `baseline` (dim 256), `matched-baseline` (dim 212), top-2 LCT
configs. Seeds: 4 common seeds {1, 2, 3, 4}; within a seed, all configs share
identical init streams and identical training-batch sequences (paired
design). 2,000 steps, deterministic full-val every 100 steps.

## Decision rule (fixed in advance)

Let `delta(s)` = best-val(LCT config, seed s) − best-val(control, seed s).

1. **Real improvement at matched parameters**: mean `delta` vs
   `matched-baseline` < −0.01 nats with the same sign in 4/4 seeds.
2. **Real improvement at matched architecture**: same test vs the dim-256
   `baseline`. (Passing 1 but not 2 = "better use of a parameter budget";
   passing both = unambiguous win.)
3. **Wall-clock**: report tokens/sec for every run; any claimed win must not
   lose at matched wall-clock by more than it gains at matched steps
   (read off val-vs-time curves).
4. Anything smaller or sign-inconsistent is reported as "no detectable
   effect at this scale" — also a real answer.

Secondary reporting: final-val (with the overfitting caveat), val-vs-step
curves, val at the baseline's best-val step, params, throughput.

## Known threats accepted

- Single dataset/scale; conclusions are about this regime.
- Constant lr, no warmup/schedule (shared by all variants).
- MPS kernel nondeterminism adds run-to-run noise beyond seeds; the paired
  4-seed sign test is the mitigation.
- Winner's curse on the tuned LCT config is bounded by pre-registering this
  rule and never reusing pilot numbers as evidence.

## Outcome (filled in after the runs; decision rule unchanged)

- Rule 1 (vs matched-baseline-212): FAILED for all LCT configs — consistent
  regression in 4/4 seeds, mean +0.33 to +0.38 nats.
- Rule 2 (vs baseline-256): FAILED at 2000 steps (4/4 seeds, +0.11 to +0.16);
  exploratory 5000-step run shows linear-fourier crossing this (sick) control
  at ~step 2000 and winning by 0.32 nats, but still losing to matched dense
  controls at either width.
- Rule 3 (wall-clock): FAILED — 0.55–0.85x dense throughput at these widths.
- Headline: **no real improvement at this scale; not tunable into one within
  the swept space.** See paper/nanogpt_lct_note.md for the full narrative,
  including the two narrower positives (conditioning rescue of the plateaued
  dim-256 dense model; repaired activation variant passing the same-width
  baseline long-horizon).
