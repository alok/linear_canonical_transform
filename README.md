# lct-activation

`lct-activation` is a small PyTorch research package for testing Linear Canonical Transform layers inside real models.

The package keeps two goals in view:

- correctness on finite grids, via a dense reference kernel and explicit tests for special cases
- speed where it matters, via FFT and Bluestein / chirp-z fast paths instead of Python loops

The package now exposes two model-facing building blocks:

- `LCTActivation`, a genuinely nonlinear modReLU-style activation in the LCT domain
- `LCTLinear`, a structured `nn.Linear`-style layer that uses fast spectral mixing instead of a dense learned matrix

Real channels are packed into complex pairs, transformed by an LCT, mixed in the transform domain, and unpacked back to real tensors. The default `LCTLinear` initialization is identity-like, so it can slot into an MLP without blowing up activations on step one.

## Install

```bash
cd /Users/alokbeniwal/LCT
uv sync --extra dev
```

Install directly from GitHub with `uv` once this branch is pushed:

```bash
uv add "git+https://github.com/alok/linear_canonical_transform.git@codex/lct-activation-nanogpt"
```

## Quick use

```python
import torch

from lct_activation import LCTActivation, LCTLinear

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

linear = LCTLinear(1024, 2048)
z = linear(torch.randn(8, 1024))
print(z.shape)

dense_equivalent = linear.to_linear()
```

## Core package

- `src/lct_activation/functional/lct.py`: dense reference kernel, `b ~= 0` branch, Fourier/Laplace special cases, and the finite-dimensional symplectic solve `symplectic_d`
- `src/lct_activation/functional/chirpz.py`: generic `O(N log N)` Bluestein / chirp-z path
- `src/lct_activation/layers.py`: `LCTLayer`, `LCTActivation`, and `LCTLinear`

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

Microbenchmark the structured linear layer against `nn.Linear`:

```bash
uv run python scripts/bench_linear.py \
  --device cpu \
  --batch-size 256 \
  --in-features 1024 \
  --out-features 1024
```

On this machine, the current implementation is still slower than `nn.Linear`
for small 512-wide CPU layers, but already faster around 4096 features where
the structured FFT path starts to dominate the dense matmul.

Run the local NanoGPT ablation sweep:

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
uv run pytest -q tests/test_lct_linear.py
```
