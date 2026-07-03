from __future__ import annotations

import importlib.metadata
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from .layers import LCTLinear
from .properties import property_report
from .results import collect_result_rows

CheckStatus = Literal["pass", "warn", "fail"]

__all__ = [
    "CheckStatus",
    "DoctorCheck",
    "DoctorReport",
    "format_doctor_text",
    "run_doctor",
]


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: CheckStatus
    message: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(check.status == "warn" for check in self.checks)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "has_warnings": self.has_warnings,
            "checks": [check.as_dict() for check in self.checks],
        }


def _package_check() -> DoctorCheck:
    try:
        version = importlib.metadata.version("lct-activation")
    except importlib.metadata.PackageNotFoundError:
        return DoctorCheck(
            "package",
            "warn",
            "lct-activation is importable, but package metadata is not installed",
        )
    return DoctorCheck("package", "pass", f"lct-activation {version} is importable")


def _python_check() -> DoctorCheck:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 10):
        return DoctorCheck("python", "fail", f"Python {version} is too old; Python >=3.10 is required")
    return DoctorCheck("python", "pass", f"Python {version} satisfies >=3.10")


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _torch_check(device_spec: str) -> tuple[DoctorCheck, torch.device | None]:
    try:
        device = _device(device_spec)
        x = torch.ones(2, device=device)
        _ = x + x
    except Exception as exc:
        return DoctorCheck("torch", "fail", f"Torch device check failed: {exc}"), None
    return DoctorCheck("torch", "pass", f"torch {torch.__version__} works on {device}"), device


def _layer_smoke_check(device: torch.device | None) -> DoctorCheck:
    if device is None:
        return DoctorCheck("layer-smoke", "fail", "Skipped because the Torch device check failed")
    try:
        torch.manual_seed(0)
        layer = LCTLinear(16, 16).to(device)
        x = torch.randn(2, 16, device=device)
        y = layer(x)
        if y.shape != (2, 16):
            return DoctorCheck("layer-smoke", "fail", f"LCTLinear returned shape {tuple(y.shape)}")
        if not torch.isfinite(y).all():
            return DoctorCheck("layer-smoke", "fail", "LCTLinear returned non-finite values")
    except Exception as exc:
        return DoctorCheck("layer-smoke", "fail", f"LCTLinear smoke test failed: {exc}")
    return DoctorCheck("layer-smoke", "pass", "LCTLinear forward smoke test passed")


def _spectral_property_check() -> DoctorCheck:
    try:
        report = property_report(
            16,
            (0.8660254, 0.5, -0.5),
            (0.8660254, -0.5, 0.5),
            discretization="spectral-frft",
        )
    except Exception as exc:
        return DoctorCheck("spectral-frft", "fail", f"Spectral FrFT property check failed: {exc}")

    if report.first_unitarity_error > 1e-5 or report.composition_error > 1e-5:
        return DoctorCheck(
            "spectral-frft",
            "fail",
            "Spectral FrFT exceeded tolerance: "
            f"unitarity={report.first_unitarity_error:.3e}, composition={report.composition_error:.3e}",
        )
    return DoctorCheck(
        "spectral-frft",
        "pass",
        "Spectral FrFT preserves unitarity/composition within 1e-5 "
        f"(unitarity={report.first_unitarity_error:.3e}, composition={report.composition_error:.3e})",
    )


def _compat_check() -> DoctorCheck:
    try:
        from linear_canonical_transform import LCTLinear as CompatLCTLinear
    except Exception as exc:
        return DoctorCheck("compat-import", "fail", f"Compatibility import failed: {exc}")
    if CompatLCTLinear is not LCTLinear:
        return DoctorCheck("compat-import", "fail", "Compatibility import returned an unexpected LCTLinear")
    return DoctorCheck("compat-import", "pass", "linear_canonical_transform compatibility import works")


def _results_check(result_dir: Path | None, *, require_results: bool) -> DoctorCheck:
    checked_dir = result_dir
    if checked_dir is None:
        candidate = Path("paper/results")
        checked_dir = candidate if candidate.exists() else None

    if checked_dir is None:
        status: CheckStatus = "fail" if require_results else "warn"
        return DoctorCheck(
            "result-artifacts",
            status,
            "No result artifact directory was checked; pass --result-dir in the repo to verify paper evidence",
        )

    if not checked_dir.exists():
        status = "fail" if require_results else "warn"
        return DoctorCheck("result-artifacts", status, f"{checked_dir} does not exist")

    rows = collect_result_rows(checked_dir.glob("*.json"))
    if not rows:
        status = "fail" if require_results else "warn"
        return DoctorCheck("result-artifacts", status, f"{checked_dir} contains no recognized result rows")

    artifacts = {row.artifact for row in rows}
    return DoctorCheck(
        "result-artifacts",
        "pass",
        f"Found {len(rows)} evidence rows across {len(artifacts)} JSON artifacts in {checked_dir}",
    )


def run_doctor(
    *,
    device: str = "cpu",
    result_dir: Path | None = None,
    require_results: bool = False,
) -> DoctorReport:
    """Run install and finite-grid smoke checks for a fresh LCT environment."""

    checks: list[DoctorCheck] = []
    checks.append(_package_check())
    checks.append(_python_check())
    torch_check, resolved_device = _torch_check(device)
    checks.append(torch_check)
    checks.append(_layer_smoke_check(resolved_device))
    checks.append(_spectral_property_check())
    checks.append(_compat_check())
    checks.append(_results_check(result_dir, require_results=require_results))
    return DoctorReport(tuple(checks))


def format_doctor_text(report: DoctorReport) -> str:
    lines = []
    for check in report.checks:
        prefix = check.status.upper()
        lines.append(f"[{prefix}] {check.name}: {check.message}")
    summary = "ok" if report.ok else "failed"
    if report.ok and report.has_warnings:
        summary = "ok with warnings"
    lines.append(f"summary: {summary}")
    return "\n".join(lines)
