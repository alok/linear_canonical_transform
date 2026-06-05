from __future__ import annotations

import math

from lct_activation import property_report


def frft(angle_degrees: float) -> tuple[float, float, float]:
    theta = math.radians(angle_degrees)
    return math.cos(theta), math.sin(theta), -math.sin(theta)


def main() -> None:
    lct_kernel = property_report(
        16,
        frft(30),
        frft(-30),
        normalization="unitary",
        unitary_projection=True,
    )
    spectral = property_report(
        16,
        frft(30),
        frft(-30),
        normalization="unitary",
        discretization="spectral-frft",
    )

    print("lct-kernel unitarity error", lct_kernel.first_unitarity_error)
    print("lct-kernel composition error", lct_kernel.composition_error)
    print("spectral-frft unitarity error", spectral.first_unitarity_error)
    print("spectral-frft composition error", spectral.composition_error)


if __name__ == "__main__":
    main()
