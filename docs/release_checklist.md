# Release Checklist

Use this checklist before advertising or publishing `lct-activation`.

## Required Before Public Release

- Choose and add an explicit open-source license file.
- Confirm the package name on PyPI is available or choose the final public name.
  The PyPI JSON API returned `404` for `lct-activation` on 2026-06-05, but
  this must be re-checked immediately before publishing:
  `curl -sS -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/lct-activation/json`.
- Confirm the GitHub repository URL in `pyproject.toml` points to the intended public repo.
- Review `README.md` and `docs/api.md` from a fresh-user perspective:
  install, quickstart, property diagnostics, result summaries, NanoGPT notes.
- Keep claims aligned with checked-in evidence:
  - `LCTLinear` is the lead component.
  - `LCTActivation` remains exploratory.
  - The sampled LCT-kernel FrFT path is diagnostic/tradeoff evidence, not a solved exact-composition claim.
  - The finite spectral FrFT path is the compositional/unitary finite-grid FrFT option.
  - Current KellerJordan/modded-nanogpt evidence is not a record-submission claim.

## Local Verification

Run from the repository root:

```bash
uv sync --extra dev
uv run pytest -q
uv run lct quickstart --format json
uv run lct doctor --result-dir paper/results --require-results
uv run lct-doctor --result-dir paper/results --require-results
uv run lct check-properties --length 8 --first-angle-degrees 30 --second-angle-degrees -30 --discretization spectral-frft
uv run lct assert-properties --length 8 --first-angle-degrees 30 --second-angle-degrees -30
uv run lct sweep-properties --length 8 16 --angle-pair 30 -30 --format json | uv run python -m json.tool >/tmp/lct-property-sweep.json
uv run python examples/quickstart.py
uv run python examples/property_diagnostics.py
uv run lct-check-properties --length 8 --first-angle-degrees 30 --second-angle-degrees -30
uv run lct-check-properties --length 8 --first-angle-degrees 30 --second-angle-degrees -30 --discretization spectral-frft
uv run lct-summarize-results --result-dir paper/results --format json | uv run python -m json.tool >/tmp/lct-summary.json
uv build
uv run python scripts/smoke_dist.py
```

Expected current baseline:

- test suite passes
- install doctor passes core package, layer, compatibility import, spectral FrFT,
  and checked-in result artifact checks
- PEP 561 typed-package markers are present in both import packages
- examples run without external data
- property diagnostics emits valid JSON
- property assertion exits successfully for the spectral-FrFT finite-grid baseline
- property sweep emits valid JSON and compares sampled-kernel and spectral-FrFT rows
- result summary emits valid JSON
- `uv build` creates both an sdist and wheel
- built wheel passes an isolated install smoke test outside the source project

## GitHub Checks

The CI workflow in `.github/workflows/ci.yml` should pass on Python 3.10 and
3.12. It intentionally checks the package from the user-facing surfaces:

- full test suite
- install doctor
- examples
- property diagnostics CLI
- property assertion CLI
- result summary CLI
- package build
- isolated built-wheel smoke test

## Publishing Sketch

Only run this after adding a license and confirming package name/metadata.

```bash
uv build
uv publish
```

If publishing to TestPyPI first, use the relevant `uv publish` options and
install from the TestPyPI index in a clean environment before publishing to
PyPI.

Before publishing, run `uv run python scripts/smoke_dist.py` against the exact
wheel you intend to upload.

## Post-Release Smoke Test

In a temporary directory:

```bash
uv init --bare lct-smoke
cd lct-smoke
uv add lct-activation
uv run python - <<'PY'
import torch
from lct_activation import LCTLinear, property_report

layer = LCTLinear(16, 16)
x = torch.randn(2, 16)
print(layer(x).shape)
print(
    property_report(
        8,
        (0.8660254, 0.5, -0.5),
        (0.8660254, -0.5, 0.5),
        discretization="spectral-frft",
    ).composition_error
)
PY
```
