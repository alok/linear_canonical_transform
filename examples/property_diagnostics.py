from __future__ import annotations

import math

from lct_activation import property_report


def frft(angle_degrees: float) -> tuple[float, float, float]:
    theta = math.radians(angle_degrees)
    return math.cos(theta), math.sin(theta), -math.sin(theta)


def main() -> None:
    report = property_report(
        16,
        frft(30),
        frft(-30),
        normalization="unitary",
        unitary_projection=True,
    )

    print("first determinant error", report.first_determinant_error)
    print("first unitarity error", report.first_unitarity_error)
    print("composition error", report.composition_error)


if __name__ == "__main__":
    main()
