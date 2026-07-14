# lct-activation

`lct-activation` is a small PyTorch research package for testing Linear
Canonical Transform layers inside real models.

The package keeps two goals in view:

- correctness on finite grids, via a dense reference kernel and explicit tests for special cases
- structured execution, via FFT and Bluestein / chirp-z paths instead of Python loops

The package now exposes one lead model-facing building block and one
experimental activation:

- `LCTLinear`, a structured `nn.Linear`-style layer that uses fast spectral mixing instead of a dense learned matrix
- `LCTActivation`, a genuinely nonlinear modReLU-style activation in the LCT
  domain for experiments

Both layers are also available as native [MLX](https://github.com/ml-explore/mlx)
modules for Apple silicon via the optional `mlx` extra (see
[MLX backend](#mlx-backend-apple-silicon) below).

Real channels are packed into complex pairs, transformed by an LCT, mixed in the transform domain, and unpacked back to real tensors. The default `LCTLinear` initialization is identity-like, so it can slot into an MLP without blowing up activations on step one.

The current research paper is an interactive HTML instrument under
[`site/`](https://github.com/alok/linear_canonical_transform/tree/main/site): it combines a determinant-one 3D phase-space
explainer, a guided prediction loop, the controlled NanoGPT evidence, and the
exploratory H100 learned-transform result. [`paper/report.md`](https://github.com/alok/linear_canonical_transform/blob/main/paper/report.md)
is retained as the archival fixed-transform report that preceded the learned
symplectic implementation.

The finite-dimensional tradeoff is explicit:

- `normalization="unitary"` favors energy preservation and stable optimization
- `normalization="compositional"` favors behavior that tracks matrix
  composition more closely

## Install

The current public release is
[`v0.1.0`](https://github.com/alok/linear_canonical_transform/releases/tag/v0.1.0).
Add it to a `uv` project directly from the immutable Git tag:

```bash
uv add "lct-activation @ git+https://github.com/alok/linear_canonical_transform.git@v0.1.0"
```

Install the optional MLX backend explicitly when needed:

```bash
uv add "lct-activation[mlx] @ git+https://github.com/alok/linear_canonical_transform.git@v0.1.0"
```

To make the packaged commands available independently of a project:

```bash
uv tool install \
  --from "git+https://github.com/alok/linear_canonical_transform.git@v0.1.0" \
  lct-activation
```

Quick smoke test after installing:

```bash
lct quickstart
```

The same self-contained smoke test can emit JSON:

```bash
lct quickstart --format json
```

Manual Python smoke test in a project that has added the package:

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

## Development

Clone the repository and install its locked development environment:

```bash
git clone https://github.com/alok/linear_canonical_transform.git
cd linear_canonical_transform
uv sync --extra dev
uv run pytest -q
```

To test an unreleased branch or tag in another `uv` project:

```bash
uv add "lct-activation @ git+https://github.com/alok/linear_canonical_transform.git@<branch-or-tag>"
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

## MLX backend (Apple silicon)

The optional MLX backend provides the same two layers as native
`mlx.nn.Module`s. On Apple silicon, `uv sync --extra dev` already pulls in
`mlx`; downstream projects can use the extra explicitly:

```bash
uv add "lct-activation[mlx]"
```

```python
import mlx.core as mx

from lct_activation.mlx import LCTActivation, LCTLinear

act = LCTActivation(1024)
y = act(mx.random.normal((8, 128, 1024)))

linear = LCTLinear(1024, 1024)
z = linear(mx.random.normal((8, 1024)))
```

The MLX backend matches the PyTorch numerics branch by branch and is covered
by parity tests (`tests/test_mlx_backend.py`). Transform parameters `(a, b, c)`
are fixed at construction and compiled into precomputed per-length plans
(chirps, Bluestein tables, dense kernels), because MLX's lazy tracing cannot
branch on traced parameter values; the modReLU bias/gain/residual mix and the
spectral diagonal/bias remain trainable.

Runnable examples live in [`examples/`](https://github.com/alok/linear_canonical_transform/tree/main/examples):

```bash
uv run lct quickstart
uv run python examples/quickstart.py
uv run python examples/property_diagnostics.py
uv run python examples/mlx_quickstart.py  # trains a tiny MLX LCT-MLP
```

## Core package

- `src/lct_activation/functional/lct.py`: dense reference kernel, `b ~= 0` branch, Fourier/Laplace special cases, and the finite-dimensional symplectic solve `symplectic_d`
- `src/lct_activation/functional/chirpz.py`: generic `O(N log N)` Bluestein / chirp-z path
- `src/lct_activation/layers.py`: `LCTLayer`, `LCTActivation`, and `LCTLinear`
- `src/lct_activation/properties.py`: finite-grid diagnostics for determinant,
  unitarity, and composition errors
- `src/lct_activation/doctor.py`: install, smoke-test, and local evidence checks

Math notes for the discrete approximation live in [`docs/lct_math.md`](https://github.com/alok/linear_canonical_transform/blob/main/docs/lct_math.md).
The public API surface is summarized in [`docs/api.md`](https://github.com/alok/linear_canonical_transform/blob/main/docs/api.md).

## License

`lct-activation` is distributed under the Apache License, Version 2.0. See
[`LICENSE`](https://github.com/alok/linear_canonical_transform/blob/main/LICENSE). Copyright 2026 Alok Singh.

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

The spectral FrFT path is intentionally a diagnostics/research API in this
release. It is not wired into `LCTLinear` as a model-facing execution path until
benchmark evidence justifies that promotion.

Use `assert-properties` when you want a CI-friendly pass/fail check with
thresholds:

```bash
lct assert-properties \
  --length 16 \
  --first-angle-degrees 30 \
  --second-angle-degrees -30
```

The assertion command defaults to the spectral FrFT discretization and exits
nonzero if determinant, unitarity, or composition errors exceed the configured
thresholds. Use `check-properties` or `sweep-properties` when comparing
finite-discretization tradeoffs without treating the sampled-kernel path as a
failure.

To generate a compact tradeoff table across lengths and angles:

```bash
lct sweep-properties \
  --length 8 16 32 \
  --angle-pair 30 -30 \
  --angle-pair 45 -45
```

Use JSON when collecting paper artifacts:

```bash
lct sweep-properties \
  --length 8 16 32 \
  --angle-pair 30 -30 \
  --format json \
  --output paper/results/property_sweep.json
```

Saved sweep JSON is understood by `lct-summarize-results`, including unitarity
and composition columns.

The same diagnostics are available from Python:

```python
from lct_activation import composition_error, finite_lct_matrix, property_sweep, unitarity_error

params = (0.8660254, 0.5, -0.5)
matrix = finite_lct_matrix(16, params, normalization="unitary")
print(unitarity_error(matrix))

rows = property_sweep(
    lengths=[8, 16],
    angle_pairs_degrees=[(30.0, -30.0)],
)
print(rows[0].as_dict())
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

- `src/lct_activation/integrations/nanogpt.py` provides source-sliced loading for an explicitly supplied NanoGPT checkout, an upstream patch path for `karpathy/nanoGPT`, and model builders for both layouts
- `scripts/bench_nanogpt.py` benchmarks baseline vs LCT activation throughput on random tokens
- `scripts/train_nanogpt_lct.py` runs upstream `train.py` in-process after applying the LCT patch

## Benchmark usage

Smoke-test against a local NanoGPT checkout:

```bash
uv run python scripts/bench_nanogpt.py \
  --repo-dir /path/to/nanoGPT \
  --repo-kind local \
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

In the checked-in measurements, the current implementation is slower than
`nn.Linear` for small 512-wide CPU layers and faster around 4096 features,
where the structured FFT path starts to dominate the dense matmul.

Benchmark all local Mac backends (torch CPU, torch MPS, MLX) in one run:

```bash
uv run python scripts/bench_mac_local.py \
  --output paper/results/bench_mac_local.json
```

Representative numbers from an Apple-silicon laptop (batch 8, seq 256,
forward pass, median of 30 steps; see
[`paper/results/bench_mac_local.json`](https://github.com/alok/linear_canonical_transform/blob/main/paper/results/bench_mac_local.json)):

| dim  | backend   | `nn.Linear` | `LCTLinear` | GELU    | `LCTActivation` |
|------|-----------|-------------|-------------|---------|-----------------|
| 1024 | torch MPS | 0.54 ms     | 0.39 ms     | 0.29 ms | 1.49 ms         |
| 1024 | MLX       | 0.49 ms     | 0.49 ms     | 0.18 ms | 0.74 ms         |
| 4096 | torch MPS | 5.64 ms     | 1.11 ms     | 0.47 ms | 2.53 ms         |
| 4096 | MLX       | 5.40 ms     | 1.96 ms     | 0.27 ms | 2.59 ms         |

In this sweep, the structured `LCTLinear` overtakes the dense matmul around
1024-2048 features and is about 5x faster at 4096 on MPS. The nonlinear
`LCTActivation` costs roughly 4-10 GELUs in the same measurements; MLX has the
lowest measured time for it up to ~2048 features, with torch MPS edging ahead
at 4096.

Methodology notes: both layers are benchmarked at their default transform
`(a, b, c) = (0, 1, 0)`, i.e. the FFT fast path. The torch numbers include
per-call branch dispatch (several scalar GPU-to-CPU syncs per forward on
MPS), which the MLX backend compiles away into per-length plans during
warmup; that is an honest end-to-end cost of each implementation, not a
like-for-like kernel comparison. Forward+backward timings differentiate
with respect to both the input and the layer parameters on both frameworks.

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

After building a wheel, smoke-test it outside the source project:

```bash
uv build
uv run python scripts/smoke_dist.py
```

Before publishing, run the release verifier against the exact artifacts. It
checks wheel and sdist metadata, the Apache license, public project URLs, local
git origin, the isolated wheel smoke test, and whether the current version is
still uploadable on PyPI:

```bash
uv run python scripts/verify_release.py --check-pypi
```

Before public release, use [`docs/release_checklist.md`](https://github.com/alok/linear_canonical_transform/blob/main/docs/release_checklist.md).
The GitHub Actions workflow in [`.github/workflows/ci.yml`](https://github.com/alok/linear_canonical_transform/blob/main/.github/workflows/ci.yml)
runs tests, examples, property diagnostics, result summaries, package build, and
isolated wheel smoke and release-metadata checks on Python 3.10 and 3.12.
