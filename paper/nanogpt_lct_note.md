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

## Current conclusion

If this line of work is going to be adopted, the right story is currently:

1. keep `LCTLinear` as the main productized layer,
2. treat `LCTActivation` as exploratory,
3. focus the next NanoGPT tuning wave on the linear variant only.

## Next tuning steps

- Sweep `LCTLinear` over more FrFT angles instead of just Fourier vs `pi/4`.
- Test `inverse_after_multiply=False` for the structured linear layer.
- Run the same ablation at larger widths, where the FFT-backed linear path is
  already faster than `nn.Linear` on CPU.
- Repeat on upstream `karpathy/nanoGPT` with a clean training config and log a
  longer loss curve, not just a 20-step snapshot.
