# LCT Activation

`lct-activation` is a fresh standalone home for the reusable Linear Canonical
Transform work that had previously been scattered across older local
experiments.

The useful sources I found in `~/` were:

- `/Users/alokbeniwal/fractional_fourier_net`, which was mostly a stub
- `/Users/alokbeniwal/modded-nanogpt`, which had the real LCT research code but
  mixed it with training, papers, and benchmark records

This repo extracts the reusable PyTorch core into a small package with a clean
test suite and a `uv`-first workflow.

## What is here

- `lct_activation.LCTLayer`: a differentiable 1-D discrete LCT module
- `lct_activation.LCTActivation`: a real-valued modReLU-style activation built
  on top of the LCT layer
- `lct_activation.linear_canonical_transform`: functional API
- `lct_activation.symplectic_d`: stable recovery of `d` from `ad - bc = 1`
- Dense reference and Bluestein / Chirp-Z fast paths
- Minimal NanoGPT integration under `src/lct_activation/integrations/`

## Quick start

```bash
uv sync --extra dev
uv run pytest
```

## Example

```python
import torch

from lct_activation import LCTActivation, LCTLayer

x = torch.randn(2, 128, dtype=torch.complex64)
layer = LCTLayer.fractional_fourier(torch.pi / 4)
y = layer(x)
z = layer.inverse(y)

activation = LCTActivation(128)
real_y = activation(torch.randn(2, 128))
```

## NanoGPT integration

The optional NanoGPT hook lives under `src/lct_activation/integrations/`.
There are two convenience scripts:

```bash
uv run python scripts/bench_nanogpt.py --steps 10 --warmup-steps 3
uv run python scripts/train_nanogpt_lct.py --clone-if-missing -- --batch_size=8
```

These are kept as scripts instead of package entry points because they are
workspace-oriented utilities, not part of the core library API.
