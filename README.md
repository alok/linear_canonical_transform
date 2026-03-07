# lct-activation

Core package description goes here.

## NanoGPT integration

This repo now includes a minimal NanoGPT integration under `lct_activation.integrations`:

- `src/lct_activation/integrations/nanogpt.py` provides a monkey-patch that replaces NanoGPT MLP activations with a nonlinear LCT-style spectral activation.
- `scripts/train_nanogpt_lct.py` runs upstream `train.py` in-process after applying the patch, and can clone `karpathy/nanoGPT` if the checkout is missing.
- `scripts/bench_nanogpt.py` benchmarks baseline vs LCT throughput on random tokens. It defaults to the local `/Users/alokbeniwal/nanogpt` repo and loads only the pre-training definitions by slicing the source before `gpt = GPT()`.

## Benchmark usage

Local NanoGPT benchmark:

```bash
uv run python scripts/bench_nanogpt.py --steps 10 --warmup-steps 3
```

Benchmark a different checkout explicitly:

```bash
uv run python scripts/bench_nanogpt.py --repo-dir /path/to/nanoGPT --repo-kind upstream
```

## Training usage

Run upstream NanoGPT with the LCT patch:

```bash
uv run python scripts/train_nanogpt_lct.py --clone-if-missing -- --batch_size=8
```
