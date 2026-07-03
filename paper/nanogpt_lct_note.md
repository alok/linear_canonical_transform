# LCT in NanoGPT: Early Adoption Note

This note records the first end-to-end NanoGPT ablation run for the
`lct-activation` package.

## Motivation

The packaging and layer work in this repo is only interesting if the resulting
modules help on a real training loop. The immediate question is not whether the
Linear Canonical Transform is mathematically elegant; it is whether an LCT-based
layer is a useful drop-in component for a small autoregressive language model.

## Variants tested

The current NanoGPT integration supports four patch modes:

- `baseline`: no LCT patch
- `activation`: replace the MLP nonlinearity with `LCTActivation`
- `linear`: replace the first MLP projection with `LCTLinear`
- `hybrid`: replace both the first MLP projection and the nonlinearity

For the first sweep, we evaluated:

- `baseline`
- `activation-fourier`
- `activation-frft45`
- `linear-fourier`
- `linear-frft45`
- `hybrid-fourier`

## Experimental setup

Command:

```bash
uv run python scripts/tune_nanogpt_lct.py \
  --device cpu \
  --steps 20 \
  --eval-iters 4 \
  --batch-size 8 \
  --seq-len 24 \
  --n-layers 2 \
  --n-heads 4 \
  --embed-dim 64
```

Model/data path:

- local NanoGPT checkout: `/Users/alokbeniwal/nanogpt`
- dataset: the local tiny Shakespeare path used by that repo

The raw JSON artifact is stored in
[`paper/results/nanogpt_local_tune.json`](/Users/alokbeniwal/LCT/paper/results/nanogpt_local_tune.json).

A follow-up linear-only run at 40 steps is stored in
[`paper/results/nanogpt_local_tune_linear_only.json`](/Users/alokbeniwal/LCT/paper/results/nanogpt_local_tune_linear_only.json).

An FRFT angle sweep for the linear layer is stored in
[`paper/results/nanogpt_linear_angle_sweep.json`](/Users/alokbeniwal/LCT/paper/results/nanogpt_linear_angle_sweep.json).

A Linux smoke test on Modal is stored in
[`paper/results/modal_linux_smoke.json`](/Users/alokbeniwal/LCT/paper/results/modal_linux_smoke.json).

An Apple MPS sweep is stored in
[`paper/results/nanogpt_mps_sweep.json`](/Users/alokbeniwal/LCT/paper/results/nanogpt_mps_sweep.json).

A CUDA / Modal GPU sweep is stored in
[`paper/results/modal_gpu_sweep.json`](/Users/alokbeniwal/LCT/paper/results/modal_gpu_sweep.json).

A parameter-efficiency comparison on MPS is stored in
[`paper/results/nanogpt_param_efficiency_mps.json`](/Users/alokbeniwal/LCT/paper/results/nanogpt_param_efficiency_mps.json).

A normalization-mode comparison on MPS is stored in
[`paper/results/nanogpt_norm_tradeoff_mps.json`](/Users/alokbeniwal/LCT/paper/results/nanogpt_norm_tradeoff_mps.json).

## Results

| variant | final val loss | tokens/s | params |
| --- | ---: | ---: | ---: |
| `linear-frft45` | `3.8680` | `26.4k` | `77,767` |
| `linear-fourier` | `3.8768` | `42.4k` | `77,767` |
| `baseline` | `3.9415` | `45.0k` | `110,017` |
| `activation-frft45` | `3.9749` | `19.7k` | `110,283` |
| `activation-fourier` | `3.9931` | `37.0k` | `110,283` |
| `hybrid-fourier` | `4.0323` | `33.9k` | `78,033` |

## Interpretation

The current evidence points in one direction:

- `LCTLinear` looks promising.
- `LCTActivation` does not look promising yet.

In this short run, both structured linear variants beat baseline validation
loss, and they do so with fewer parameters than the baseline feedforward block.
The best early-loss result came from the FrFT-style linear variant, while the
Fourier linear variant was materially faster and only slightly worse in loss.

By contrast, the activation-only variants were slower and worse than baseline,
and the hybrid variant underperformed all of the simpler alternatives.

## Follow-up: linear-only rerun

To check that the signal was not just a 20-step fluke, we reran the top three
variants for 40 steps with the same tiny local setup:

| variant | final val loss | tokens/s |
| --- | ---: | ---: |
| `linear-frft45` | `3.6563` | `24.4k` |
| `linear-fourier` | `3.6575` | `41.7k` |
| `baseline` | `3.6892` | `42.8k` |

That rerun strengthens the same conclusion:

- the structured linear layer still outperforms baseline in loss,
- the FrFT-style version is marginally best in loss,
- the Fourier version is almost as good in loss and much better in speed.

## Current conclusion

If this line of work is going to be adopted, the right story is currently:

1. keep `LCTLinear` as the main productized layer,
2. treat `LCTActivation` as exploratory,
3. focus the next NanoGPT tuning wave on the linear variant only.

## Parameter efficiency

The strongest current paper angle is not raw standalone layer throughput. It is
that the structured linear LCT layer can reach better loss with fewer
parameters.

On a small MPS comparison:

| model | params | final val loss |
| --- | ---: | ---: |
| `baseline-56` | `85,073` | `4.2293` |
| `baseline-64` | `109,505` | `4.0362` |
| `linear64-frft15` | `77,255` | `3.8816` |
| `linear64-frft30` | `77,255` | `3.8205` |

So in this setup, the best `LCTLinear` model outperformed both the same-width
baseline and a narrower baseline with a closer parameter budget.

## Normalization tradeoff

The `unitary` vs `compositional` distinction is not just mathematical framing.
It changes the learning behavior.

On a small MPS comparison at width 64:

| mode | variant | final val loss |
| --- | --- | ---: |
| `unitary` | Fourier | `4.0246` |
| `unitary` | FrFT 15° | `3.9990` |
| `unitary` | FrFT 30° | `3.9555` |
| `compositional` | Fourier | `3.7919` |
| `compositional` | FrFT 15° | `3.8170` |
| `compositional` | FrFT 30° | `3.8445` |

So at least in this toy setting, the compositional discretization can actually
be the stronger inductive bias. That makes the finite-dimensional tradeoff a
real experimental axis for the paper, not just an implementation note.

## Angle sweep

We also swept a few additional FRFT-style angles for `LCTLinear` while keeping
the same tiny NanoGPT setup and 20-step budget:

| variant | final val loss | tokens/s |
| --- | ---: | ---: |
| `linear-frft30` | `3.7809` | `23.8k` |
| `linear-frft15` | `3.8205` | `24.7k` |
| `linear-frft60` | `3.8460` | `24.6k` |
| `linear-frft75` | `3.8636` | `24.4k` |
| `linear-frft45` | `3.8680` | `24.3k` |
| `linear-fourier` | `3.8768` | `41.4k` |
| `baseline` | `3.9415` | `38.6k` |

This changes the tuning recommendation slightly:

- the best early-loss region is currently around `15°` to `30°`, not `45°`,
- the Fourier special case remains the best speed/quality compromise,
- the FrFT family is worth tuning further instead of freezing the project at a
  single canonical setting.

## Linux smoke test

We also ran the packaged branch inside a remote Modal Linux container. The
container successfully completed:

- `uv sync --extra dev`
- the full `pytest` suite (`26 passed`)
- `lct-bench-linear` on CPU

In that Linux environment, the 512-wide CPU microbenchmark reported
`lct_over_dense = 0.8357`, meaning the current `LCTLinear` implementation was
already faster than `nn.Linear` for that setting. This matters because it
reduces the risk that the current speed story is a macOS-local artifact.

## Apple MPS

We also ran the current branch on local Apple MPS after fixing the generic LCT
path to avoid `complex128` promotion on that backend.

In the small MPS sweep:

| variant | final val loss | tokens/s |
| --- | ---: | ---: |
| `linear-frft30` | `3.8893` | `3.34k` |
| `linear-frft15` | `3.9213` | `1.96k` |
| `linear-fourier` | `3.9591` | `7.91k` |
| `baseline` | `4.0362` | `2.95k` |

The 256-wide MPS microbenchmark also gave `lct_over_dense = 0.7563`, so the
structured linear layer is already faster than `nn.Linear` for that tested MPS
setting.

## CUDA / Triton path on Linux

We then ran the packaged branch on a Modal Tesla T4 instance.

- `uv sync` installed the CUDA-enabled `torch` wheel and `triton`
- `lct-bench-linear --device cuda --compile` ran successfully
- `lct-bench-nanogpt --device cuda --compile` ran successfully
- `lct-tune-nanogpt --device cuda` ran successfully

The important caveat is that TorchInductor still warned that complex operators
were not being code-generated efficiently, so the current CUDA compile path is
using Triton where it can, but not yet for the complex-heavy parts of the LCT.
That showed up in the 1024-wide compiled microbenchmark, where `LCTLinear` was
still slower than `nn.Linear` even after the optimized Fourier backward and the
Triton pointwise path (`lct_over_dense = 2.4709` for forward, `1.5272` for a
train step).

We also tested a convolution-style Fourier backend as a more aggressive CUDA
replacement. That turned out to be a negative result for this setup: it was
slower than the FFT backend in both the standalone forward microbenchmark and
the compiled local NanoGPT benchmark. Because of that, the default backend has
been set back to `fft`, and the convolution path remains an explicit
experimental option rather than the default.

On the other hand, the actual CUDA NanoGPT tuning run still favored the linear
variants:

| variant | final val loss | tokens/s |
| --- | ---: | ---: |
| `linear-frft15` | `3.7979` | `8.31k` |
| `linear-frft30` | `3.8031` | `9.26k` |
| `linear-frft45` | `3.8157` | `14.32k` |
| `linear-fourier` | `3.8523` | `3.63k` |
| `baseline` | `3.9415` | `7.90k` |

So the current state is:

- the branch now works on Linux CUDA and macOS MPS,
- `torch.compile` is active and usable on Linux CUDA,
- but the standalone CUDA benchmark is still bottlenecked by complex-heavy ops,
  even after adding the optimized Fourier backward and the Triton pointwise
  multiply.

## Inverse-free sweep

We also tested `inverse_after_multiply=False` around the same `10°` to `40°`
region.

This did improve throughput on both MPS and CUDA, but it did **not** improve
the validation-loss frontier enough to beat the inverse-backed variants.

On CUDA, for example:

| variant | final val loss | tokens/s |
| --- | ---: | ---: |
| `linear-frft10` | `3.9195` | `4.75k` |
| `linear-frft35` | `3.9392` | `5.80k` |
| `baseline` | `4.0362` | `2.28k` |
| `linear-fourier` | `4.0714` | `4.47k` |

So the current recommendation stays the same:

- keep `inverse_after_multiply=True` for the best paper-quality result,
- treat `inverse_after_multiply=False` as a speed-biased ablation, not the main
  model configuration.

## Next tuning steps

- Sweep `LCTLinear` more finely between `10°` and `40°`.
- Test `inverse_after_multiply=False` for the structured linear layer.
- Push more of the complex path into a custom CUDA/Triton implementation if we
  want the standalone microbenchmark to beat dense `nn.Linear` on GPU too.
- Run the same ablation at larger widths, where the FFT-backed linear path is
  already faster than `nn.Linear` on CPU.
- Repeat on upstream `karpathy/nanoGPT` with a clean training config and log a
  longer loss curve, not just a 20-step snapshot.

## Gradient-fix rerun (2026-06-12)

A cross-backend parity probe against the new MLX implementation exposed an
input-gradient bug in the PyTorch fast path: the tile-expansion adjoint in
`reduce_unpacked_grad` scattered with `i // repeats` (the adjoint of
`repeat_interleave`) where the forward expands with `repeat` (`i % original`).
Every `linear-*` and `hybrid-*` result above trained the layers *upstream* of
an expanding `LCTLinear` (the MLP up-projection, `d -> 4d`) with corrupted
gradients; parameter gradients of the expanding layer itself, the `baseline`
rows, and the `activation-*` rows were unaffected.

After the fix, the two CPU evidence runs were repeated with identical
configs and seeds. The `baseline` rows reproduce bit-for-bit (validating the
comparison); the linear variants mostly improve:

| variant | old val loss | fixed val loss | delta |
|---|---|---|---|
| `linear-fourier` (20 steps) | `3.8768` | `3.8419` | `-0.035` |
| `linear-frft15` | `3.8205` | `3.7900` | `-0.031` |
| `linear-frft30` | `3.7809` | `3.7949` | `+0.014` |
| `linear-frft45` | `3.8680` | `3.8095` | `-0.059` |
| `linear-frft60` | `3.8460` | `3.8099` | `-0.036` |
| `linear-frft75` | `3.8636` | `3.8387` | `-0.025` |
| `linear-fourier` (40 steps) | `3.6575` | `3.6467` | `-0.011` |
| `linear-frft45` (40 steps) | `3.6563` | `3.6264` | `-0.030` |

Artifacts: [`nanogpt_linear_angle_sweep_gradfix.json`](results/nanogpt_linear_angle_sweep_gradfix.json),
[`nanogpt_local_tune_linear_only_gradfix.json`](results/nanogpt_local_tune_linear_only_gradfix.json).
The qualitative conclusion is unchanged (structured linear variants beat the
parameter-matched baseline at these scales) and the margins widen slightly
with correct gradients. Earlier MPS/CUDA linear-variant tables retain their
recorded values for provenance; treat them as lower bounds on the fixed
implementation. The regression test for the adjoint lives in
`tests/test_linear_input_gradients.py`.

## Fair-controls study on MPS (2026-07-03): the tiny-run wins do not survive

Full protocol and decision rule (pre-registered before the main runs):
[`paper/experiments/mps_shakespeare_protocol.md`](experiments/mps_shakespeare_protocol.md).
Substrate: char-level tinyshakespeare, local nanogpt (pre-LN, ReLU MLP, no
attention `1/sqrt(d)` scaling), 4L/4H, seq 256, batch 64, dropout 0.2, AdamW
constant lr 1e-3 (selected by pilot for every family), Apple-silicon MPS.
Harness fixes that gate this study: real seeds + paired batch streams +
deterministic full-val eval (`803ad1d`), frozen-activation fix (`48b9046`),
tile-adjoint input-gradient fix (`5e63c6f`).

### Main result (confirmatory, 4 paired seeds, 2000 steps, dim 256)

| config | params | tok/s | best-val (mean over 4 seeds) |
|---|---|---|---|
| baseline dim-256 | 3.26M | 171k | 2.2496 |
| **matched baseline dim-212** | 2.25M | **197k** | **2.0313** |
| linear-fourier (up-proj -> LCT) | 2.21M | 145k | 2.3619 |
| linear-frft30 | 2.21M | 102k | 2.4124 |
| linear-frft45 | 2.21M | 102k | 2.3729 |

Every LCT linear config regresses against both controls in 4/4 paired seeds
(vs matched-212: +0.33 to +0.38 nats). The 300-step pilot's angle ranking
(frft45 > frft30 > fourier) inverted at 2000 steps - short-horizon pilots
select the wrong config, exactly the winner's-curse failure the protocol
anticipated.

### Extended horizon (exploratory, seed 1, 5000 steps)

The identity-initialized LCT layer starts slow and takes off late:
linear-fourier crosses the dim-256 baseline near step 2000 and finishes at
**1.910 vs 2.228** - a 0.32-nat win over the *same-width* dense model, which
plateaus (best 2.226@4400). The repaired activation variant also passes the
baseline (2.119). But the matched dim-212 dense baseline stays ahead of
everything (1.754).

At the healthy width the story repeats one level down: linear-fourier@212
(1.53M params) reaches 1.914, losing to both baseline@212 (1.740) and its own
param-matched dense control baseline@172 (1.747, at 1.8x the throughput).
Notably the LCT-MLP model lands at ~1.91 at both dim 256 and dim 212 - the
structured up-projection, not the trunk width, is the capacity bottleneck in
this regime.

### Verdict

- **Not a real improvement at this scale under fair controls.** The dense way
  to spend a parameter budget (narrower width) beats the LCT way (structured
  up-projection) by ~0.17 nats at matched params and wins throughput at these
  widths (LCT crossover vs dense matmul is ~1024+ features; the MLP here is
  256-1024).
- **Not tunable into one within the swept space** (lr x angle x normalization
  x inverse): fourier + unitary + no-inverse is the best cell and still loses.
- **Two real, narrower positives**: (1) at dim 256 the dense substrate
  plateaus pathologically (no attention scaling + constant lr) and the LCT
  up-projection *rescues* it (+0.32 nats over same-width dense) - LCT acted as
  a conditioning fix; (2) the earlier "linear beats baseline" tiny-run results
  are explained: 20-40-step comparisons at dim 64 measure init transients, and
  single-seed MPS deltas below ~0.03 nats are noise (measured).
- **Where a win could still live**: widths >= 1024 where LCTLinear is
  genuinely faster than dense (2.6-5x at 4096 on MPS/MLX) so a quality-neutral
  result becomes a wall-clock win; lr schedules/warmup tailored to the
  identity-init late takeoff; other placements (attention projections); and
  per-group lr for the ~2k spectral parameters. None of these are claims -
  they are the next experiments.
