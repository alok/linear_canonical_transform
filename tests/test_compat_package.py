from __future__ import annotations

from linear_canonical_transform import LCTLinear


def test_compat_package_reexports_lctlinear() -> None:
    layer = LCTLinear(8, 8)
    assert layer.in_features == 8
    assert layer.out_features == 8
