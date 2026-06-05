from __future__ import annotations

from linear_canonical_transform import (
    LCTLinear,
    property_report,
    property_sweep,
    run_doctor,
    spectral_fractional_fourier_matrix,
)


def test_compat_package_reexports_lctlinear() -> None:
    layer = LCTLinear(8, 8)
    assert layer.in_features == 8
    assert layer.out_features == 8


def test_compat_package_reexports_property_helpers() -> None:
    report = property_report(4, (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))
    assert report.length == 4


def test_compat_package_reexports_spectral_frft() -> None:
    matrix = spectral_fractional_fourier_matrix(4, 0.0)
    assert matrix.shape == (4, 4)


def test_compat_package_reexports_doctor() -> None:
    report = run_doctor(result_dir=None)
    assert report.ok


def test_compat_package_reexports_property_sweep() -> None:
    rows = property_sweep(
        lengths=[4],
        angle_pairs_degrees=[(30.0, -30.0)],
        discretizations=("spectral-frft",),
    )
    assert rows[0].discretization == "spectral-frft"
