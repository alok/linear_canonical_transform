# lct-activation

`lct-activation` is a small PyTorch research package for testing Linear Canonical Transform activations in transformer MLPs.

The package keeps two goals in view:

- correctness on finite grids, via a dense reference kernel and explicit tests for special cases
- speed where it matters, via FFT and Bluestein / chirp-z fast paths instead of Python loops

The activation exposed here is genuinely nonlinear. Real channels are packed into complex pairs, transformed by an LCT, passed through a modReLU-style nonlinearity in the transform domain, and unpacked back to the original real shape. That makes it a drop-in candidate for replacing `ReLU` or `GELU` inside NanoGPT feedforward blocks.

## Install

```bash
cd /Users/alokbeniwal/LCT
uv sync --extra dev
```

## Quick use

```python
import torch

from lct_activation import LCTActivation

act = LCTActivation(
    1024,
    a=0.0,
    b=1.0,
    c=0.0,
    dense_threshold=256,
)

x = torch.randn(8, 128, 1024)
y = act(x)
print(y.shape)
```

## Core package

- `src/lct_activation/functional/lct.py`: dense reference kernel, `b ~= 0` branch, Fourier/Laplace special cases, and the finite-dimensional symplectic solve `symplectic_d`
- `src/lct_activation/functional/chirpz.py`: generic `O(N log N)` Bluestein / chirp-z path
- `src/lct_activation/layers.py`: `LCTLayer` plus the real-valued `LCTActivation`

Math notes for the discrete approximation live in [`docs/lct_math.md`](docs/lct_math.md).

## NanoGPT integration

This repo includes a minimal NanoGPT integration under `lct_activation.integrations`:

- `src/lct_activation/integrations/nanogpt.py` provides source-sliced loading for the local `/Users/alokbeniwal/nanogpt` repo, an upstream patch path for `karpathy/nanoGPT`, and model builders for both layouts
- `scripts/bench_nanogpt.py` benchmarks baseline vs LCT activation throughput on random tokens
- `scripts/train_nanogpt_lct.py` runs upstream `train.py` in-process after applying the LCT patch

## Benchmark usage

Smoke-test against the local `/Users/alokbeniwal/nanogpt` repo:

```bash
uv run python scripts/bench_nanogpt.py \
  --device cpu \
  --steps 1 \
  --warmup-steps 1 \
  --batch-size 1 \
  --seq-len 12 \
  --n-layers 1 \
  --n-heads 4 \
  --embed-dim 64
```

Benchmark a different checkout explicitly:

```bash
uv run python scripts/bench_nanogpt.py \
  --repo-dir /path/to/nanoGPT \
  --repo-kind upstream
```

## Training usage

Run upstream NanoGPT with the LCT patch:

```bash
uv run python scripts/train_nanogpt_lct.py \
  --clone-if-missing \
  -- --batch_size=8
```

## Verification

Core verification used in this repo:

```bash
uv run pytest -q tests/test_lct_core.py tests/test_activation.py tests/test_special_cases.py
```
