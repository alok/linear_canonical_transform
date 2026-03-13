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
still slower than `nn.Linear` (`lct_over_dense = 2.6131`).

On the other hand, the actual CUDA NanoGPT tuning run still favored the linear
variants:

| variant | final val loss | tokens/s |
| --- | ---: | ---: |
| `linear-frft30` | `3.7809` | `12.7k` |
| `linear-frft15` | `3.8205` | `12.3k` |
| `linear-frft45` | `3.8680` | `12.5k` |
| `linear-fourier` | `3.8768` | `18.2k` |
| `baseline` | `3.9415` | `9.64k` |

So the current state is:

- the branch now works on Linux CUDA and macOS MPS,
- `torch.compile` is active and usable on Linux CUDA,
- but a custom complex-aware Triton kernel is still the next speed step if we
  want the standalone CUDA microbenchmark to close the gap with dense matmuls.

## Next tuning steps

- Sweep `LCTLinear` more finely between `10°` and `40°`.
- Test `inverse_after_multiply=False` for the structured linear layer.
- Run the same ablation at larger widths, where the FFT-backed linear path is
  already faster than `nn.Linear` on CPU.
- Repeat on upstream `karpathy/nanoGPT` with a clean training config and log a
  longer loss curve, not just a 20-step snapshot.
