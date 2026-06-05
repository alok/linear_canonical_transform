# lct-activation

`lct-activation` is a small PyTorch research package for testing Linear
Canonical Transform layers inside real models.

The package keeps two goals in view:

- correctness on finite grids, via a dense reference kernel and explicit tests for special cases
- speed where it matters, via FFT and Bluestein / chirp-z fast paths instead of Python loops

The package now exposes two model-facing building blocks:

- `LCTActivation`, a genuinely nonlinear modReLU-style activation in the LCT domain
- `LCTLinear`, a structured `nn.Linear`-style layer that uses fast spectral mixing instead of a dense learned matrix

Real channels are packed into complex pairs, transformed by an LCT, mixed in the transform domain, and unpacked back to real tensors. The default `LCTLinear` initialization is identity-like, so it can slot into an MLP without blowing up activations on step one.

The finite-dimensional tradeoff is explicit:

- `normalization="unitary"` favors energy preservation and stable optimization
- `normalization="compositional"` favors behavior that tracks matrix
  composition more closely

## Install

For local development:

```bash
cd /Users/alokbeniwal/LCT
uv sync --extra dev
uv run pytest -q
```

Use the package from this checkout in another `uv` project:

```bash
uv add "lct-activation @ file:///Users/alokbeniwal/LCT"
```

Install directly from GitHub with `uv` once the branch or tag you want is
pushed:

```bash
uv add "git+https://github.com/alok/linear_canonical_transform.git@<branch-or-tag>"
```

If you just want one packaged command-line entry point, install `lct` with
`uv tool`:

```bash
uv tool install --from "git+https://github.com/alok/linear_canonical_transform.git@<branch-or-tag>" lct
```

The older direct command names remain available for scripts and CI:

```bash
uv tool install --from "git+https://github.com/alok/linear_canonical_transform.git@<branch-or-tag>" lct-bench-linear
uv tool install --from "git+https://github.com/alok/linear_canonical_transform.git@<branch-or-tag>" lct-bench-nanogpt
uv tool install --from "git+https://github.com/alok/linear_canonical_transform.git@<branch-or-tag>" lct-check-properties
uv tool install --from "git+https://github.com/alok/linear_canonical_transform.git@<branch-or-tag>" lct-doctor
uv tool install --from "git+https://github.com/alok/linear_canonical_transform.git@<branch-or-tag>" lct-summarize-results
uv tool install --from "git+https://github.com/alok/linear_canonical_transform.git@<branch-or-tag>" lct-tune-nanogpt
```

Quick smoke test after installing:

```bash
uv run python - <<'PY'
import torch
from lct_activation import LCTLinear

layer = LCTLinear(16, 16)
x = torch.randn(2, 16)
print(layer(x).shape)
PY
```

Or run the packaged self-check:

```bash
lct doctor
```

Inside this repository, include the checked-in paper evidence artifacts:

```bash
uv run lct doctor --result-dir paper/results --require-results
```

## Quick use

```python
import torch

from lct_activation import LCTActivation, LCTLinear, property_report

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

energy_preserving = LCTLinear(1024, 1024, normalization="unitary")
matrix_like = LCTLinear(1024, 1024, normalization="compositional")

report = property_report(
    16,
    (0.8660254, 0.5, -0.5),
    (0.8660254, -0.5, 0.5),
    normalization="unitary",
    discretization="spectral-frft",
)
print(report.first_unitarity_error, report.composition_error)
```

Compatibility imports under the older repo name also work:

```python
from linear_canonical_transform import LCTLinear
```

Runnable examples live in [`examples/`](examples/):

```bash
uv run python examples/quickstart.py
uv run python examples/property_diagnostics.py
```

## Core package

- `src/lct_activation/functional/lct.py`: dense reference kernel, `b ~= 0` branch, Fourier/Laplace special cases, and the finite-dimensional symplectic solve `symplectic_d`
- `src/lct_activation/functional/chirpz.py`: generic `O(N log N)` Bluestein / chirp-z path
- `src/lct_activation/layers.py`: `LCTLayer`, `LCTActivation`, and `LCTLinear`
- `src/lct_activation/properties.py`: finite-grid diagnostics for determinant,
  unitarity, and composition errors
- `src/lct_activation/doctor.py`: install, smoke-test, and local evidence checks

Math notes for the discrete approximation live in [`docs/lct_math.md`](docs/lct_math.md).
The public API surface is summarized in [`docs/api.md`](docs/api.md).

## Finite-grid property checks

The package includes a small diagnostic CLI for the tradeoff that matters most
in this project: finite LCT kernels can be made very nearly unitary, but that
projection changes how closely finite matrices compose like their continuum
canonical parameters.

```bash
lct-check-properties \
  --length 16 \
  --first-angle-degrees 30 \
  --second-angle-degrees -30 \
  --normalization unitary \
  --unitary-projection
```

The same check is available through the umbrella command:

```bash
lct check-properties \
  --length 16 \
  --first-angle-degrees 30 \
  --second-angle-degrees -30 \
  --discretization spectral-frft
```

The output is JSON with determinant errors, unitarity errors, and composition
error. To compare the unprojected dense kernel:

```bash
lct-check-properties \
  --length 16 \
  --first-angle-degrees 30 \
  --second-angle-degrees -30 \
  --normalization unitary \
  --no-unitary-projection
```

For finite fractional Fourier transforms where exact finite-grid composition is
the priority, use the spectral FrFT discretization:

```bash
lct-check-properties \
  --length 16 \
  --first-angle-degrees 30 \
  --second-angle-degrees -30 \
  --discretization spectral-frft
```

That path constructs a fractional power of the unitary DFT from its four
spectral projectors. It is less a sampled continuum integral kernel and more a
finite-dimensional FrFT algebra: unitary and compositional up to floating-point
error.

The same diagnostics are available from Python:

```python
from lct_activation import composition_error, finite_lct_matrix, unitarity_error

params = (0.8660254, 0.5, -0.5)
matrix = finite_lct_matrix(16, params, normalization="unitary")
print(unitarity_error(matrix))
```

## Result summaries

The checked-in NanoGPT and backend artifacts under `paper/results/` can be
summarized without manual `jq` commands:

```bash
lct-summarize-results --result-dir paper/results
```

To emit JSON for a notebook or plotting script:

```bash
lct-summarize-results --result-dir paper/results --format json
```

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
  --repo-kind upstream \
  --output paper/results/bench_nanogpt_upstream.json
```

Microbenchmark the structured linear layer against `nn.Linear`:

```bash
lct-bench-linear \
  --device cpu \
  --batch-size 256 \
  --in-features 1024 \
  --out-features 1024 \
  --output paper/results/bench_linear_cpu.json
```

On this machine, the current implementation is still slower than `nn.Linear`
for small 512-wide CPU layers, but already faster around 4096 features where
the structured FFT path starts to dominate the dense matmul.

Run the local NanoGPT ablation sweep:

```bash
lct-tune-nanogpt \
  --device cpu \
  --steps 20 \
  --eval-iters 4 \
  --batch-size 8 \
  --seq-len 24 \
  --n-layers 2 \
  --n-heads 4 \
  --embed-dim 64
```

Sweep a few additional FRFT angles for the linear layer:

```bash
lct-tune-nanogpt \
  --device cpu \
  --steps 20 \
  --eval-iters 4 \
  --batch-size 8 \
  --seq-len 24 \
  --n-layers 2 \
  --n-heads 4 \
  --embed-dim 64 \
  --presets baseline linear-fourier \
  --linear-angle-degrees 15 30 45 60 75 \
  --output paper/results/nanogpt_linear_angle_sweep.json
```

Run the packaged branch in a remote Linux container on Modal:

```bash
modal run scripts/modal_linux_smoke.py
```

Run the CUDA benchmark/sweep on Modal:

```bash
modal run scripts/modal_gpu_sweep.py
```

## Training usage

Run upstream NanoGPT with the LCT patch:

```bash
lct-train-nanogpt \
  --clone-if-missing \
  --variant linear \
  -- --batch_size=8
```

## Verification

Core verification used in this repo:

```bash
uv run pytest -q tests/test_lct_core.py tests/test_activation.py tests/test_special_cases.py
uv run pytest -q tests/test_lct_linear.py
uv run pytest -q tests/test_lct_properties.py
```

Before public release, use [`docs/release_checklist.md`](docs/release_checklist.md).
The GitHub Actions workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
runs tests, examples, property diagnostics, result summaries, and package build
checks on Python 3.10 and 3.12.
