from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_docs_keep_spectral_frft_diagnostics_only() -> None:
    readme = (ROOT / "README.md").read_text()
    api = (ROOT / "docs" / "api.md").read_text()
    checklist = (ROOT / "docs" / "release_checklist.md").read_text()

    assert "one lead model-facing building block" in readme
    assert "spectral FrFT path is intentionally a diagnostics/research API" in readme
    assert "not model-facing layer backends" in api
    assert "not a model-facing `LCTLinear` execution path" in checklist
