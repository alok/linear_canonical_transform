from __future__ import annotations

from linear_canonical_transform import LCTLinear, property_report


def test_compat_package_reexports_lctlinear() -> None:
    layer = LCTLinear(8, 8)
    assert layer.in_features == 8
    assert layer.out_features == 8


def test_compat_package_reexports_property_helpers() -> None:
    report = property_report(4, (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))
    assert report.length == 4
