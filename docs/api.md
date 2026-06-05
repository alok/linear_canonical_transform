# API Guide

This guide summarizes the public surfaces that are stable enough to use from
the current research package. It intentionally separates model-facing APIs from
diagnostic APIs so finite-grid tradeoffs stay visible.

## Model-Facing Layers

Use these from `lct_activation`:

```python
from lct_activation import LCTActivation, LCTLinear
```

`LCTLinear(in_features, out_features, ...)` is the main structured layer. It is
designed as an `nn.Linear`-style module with an identity-like initialization,
fast spectral mixing, and a `to_linear()` helper for materializing a dense
comparison layer.

`LCTActivation(features, ...)` is a nonlinear modReLU-style activation in the
LCT domain. It is useful for experiments, but the release notes should continue
to describe it as exploratory unless benchmark evidence changes.

Compatibility imports are available under the older package name:

```python
from linear_canonical_transform import LCTLinear
```

## Functional Transforms

```python
from lct_activation import (
    chirpz_lct,
    linear_canonical_transform,
    spectral_fractional_fourier_matrix,
    spectral_fractional_fourier_transform,
    symplectic_d,
)
```

`linear_canonical_transform` is the sampled finite-grid LCT implementation used
by the layers and diagnostics.

`chirpz_lct` exposes the Bluestein / chirp-z fast path.

`spectral_fractional_fourier_matrix` and
`spectral_fractional_fourier_transform` expose the finite-dimensional spectral
FrFT construction. This path is algebraic on the finite grid: it is unitary and
composes by angle addition up to floating-point error. It is not the sampled
continuum integral kernel.

## Finite-Grid Diagnostics

```python
from lct_activation import (
    composition_error,
    finite_lct_matrix,
    format_property_sweep_markdown,
    property_report,
    property_sweep,
    relative_frobenius_error,
    unitarity_error,
)
```

`property_report` is the highest-level diagnostic helper. It reports determinant
errors, unitarity errors, and composition error for two canonical transforms.

Use `discretization="lct"` for the sampled-kernel path and
`discretization="spectral-frft"` for finite FrFT parameters where unitary
composition on the finite grid is the priority.

The same checks are available at the command line:

```bash
lct-check-properties --length 16 --first-angle-degrees 30 --second-angle-degrees -30
lct-check-properties --length 16 --first-angle-degrees 30 --second-angle-degrees -30 --discretization spectral-frft
lct sweep-properties --length 8 16 32 --angle-pair 30 -30
```

For paper tables:

```python
rows = property_sweep(
    lengths=[8, 16, 32],
    angle_pairs_degrees=[(30.0, -30.0), (45.0, -45.0)],
)
print(format_property_sweep_markdown(rows))
```

## Install Doctor

For a first run after install:

```bash
lct quickstart
lct quickstart --format json
```

```python
from lct_activation import run_doctor, format_doctor_text

report = run_doctor(result_dir=None)
print(format_doctor_text(report))
```

The matching CLI is:

```bash
lct doctor
```

Inside this repository, include checked-in result artifacts:

```bash
lct doctor --result-dir paper/results --require-results
```

## Result Summaries

```python
from lct_activation.results import collect_result_rows, format_markdown_table
```

For paper or README evidence tables, prefer the CLI:

```bash
lct-summarize-results --result-dir paper/results
```

This keeps checked-in NanoGPT and backend artifacts summarized from source JSON
instead of hand-maintained tables.

All installed commands are also available through the umbrella `lct` command,
for example `lct quickstart`, `lct check-properties ...`, and
`lct summarize-results ...`.

Saved `lct-bench-linear --output ...` JSON files are also summarized directly,
so quick local benchmarks can become paper evidence without manual conversion.
Saved `lct sweep-properties --format json --output ...` artifacts are
summarized with unitarity and composition columns as well.
