from __future__ import annotations

import math

import pytest
import torch

from lct_activation import LCTLayer, linear_canonical_transform


def _fourier_reference(n: int) -> torch.Tensor:
    eye = torch.eye(n, dtype=torch.complex64)
    return torch.fft.fft(eye, norm="ortho")


def _laplace_reference(n: int) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float32)
    omega = (2 * math.pi / n) * k
    t = torch.arange(n, dtype=torch.float32)
    phase = torch.outer(omega.to(torch.complex64), t.to(torch.complex64))
    kernel = torch.exp(-1j * phase)
    return (-1j / math.sqrt(n) * kernel).to(torch.complex64)


@pytest.mark.parametrize(
    ("name", "params", "reference"),
    [
        ("fourier", (0.0, 1.0, 0.0, 0.0), _fourier_reference),
        ("laplace", (0j, 1j, 1j, 0j), _laplace_reference),
    ],
)
def test_special_case_matrix(name: str, params, reference) -> None:
    n = 8
    x = torch.eye(n, dtype=torch.complex64)
    out = linear_canonical_transform(
        x,
        a=params[0],
        b=params[1],
        c=params[2],
        d=params[3],
    )
    expected = reference(n)
    assert torch.allclose(out, expected, atol=1e-4), f"mismatch for {name}"


def test_fractional_fourier_factory_matches_manual_params() -> None:
    angle = math.pi / 3
    layer = LCTLayer.fractional_fourier(angle)
    expected_a = math.cos(angle)
    expected_b = math.sin(angle)
    actual_a, actual_b, actual_c, _ = layer.canonical_matrix
    assert torch.isclose(actual_a.real, torch.tensor(expected_a, dtype=actual_a.real.dtype))
    assert torch.isclose(actual_b.real, torch.tensor(expected_b, dtype=actual_b.real.dtype))
    assert torch.isclose(actual_c.real, torch.tensor(-expected_b, dtype=actual_c.real.dtype))


def test_fresnel_factory() -> None:
    layer = LCTLayer.fresnel(wavelength=1.0, distance=2.0)
    actual_a, actual_b, actual_c, _ = layer.canonical_matrix
    assert torch.isclose(actual_a.real, torch.tensor(1.0, dtype=actual_a.real.dtype))
    assert torch.isclose(actual_b.real, torch.tensor(2.0, dtype=actual_b.real.dtype))
    assert torch.isclose(actual_c.real, torch.tensor(0.0, dtype=actual_c.real.dtype))
