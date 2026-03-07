from __future__ import annotations

import math

import torch

from lct_activation import LCTLayer
from lct_activation.functional import chirpz_lct, linear_canonical_transform, symplectic_d


def _dense_reference(
    x: torch.Tensor,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    centered: bool = True,
) -> torch.Tensor:
    length = x.size(-1)
    sample_index = torch.arange(length, dtype=torch.float64)
    if centered:
        sample_index = sample_index - (length - 1) / 2.0

    input_index = sample_index.view(length, 1)
    output_index = sample_index.view(1, length)

    phase = (
        (a / b) * input_index.square()
        - (2.0 / (b * length)) * (input_index * output_index)
        + (d / b) * output_index.square()
    )
    kernel = torch.exp(1j * math.pi * phase.to(torch.complex128)) / math.sqrt(length)
    return x.to(torch.complex128) @ kernel


def test_symplectic_d_is_stable_near_zero_a() -> None:
    a_t = torch.tensor(1e-10)
    b_t = torch.tensor(0.9)
    c_t = torch.tensor(-1.1)
    d_t = symplectic_d(a_t, b_t, c_t)
    assert torch.isfinite(torch.as_tensor(d_t)).all()


def test_canonical_matrix_stays_symplectic() -> None:
    layer = LCTLayer(a=0.71, b=0.62, c=-0.35)
    a_t, b_t, c_t, d_t = layer.canonical_matrix
    det = a_t * d_t - b_t * c_t
    assert torch.allclose(det, torch.ones_like(det), atol=1e-6, rtol=0.0)


def test_fft_fast_path_matches_dense_reference() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 31, dtype=torch.complex64)
    a_t = 0.0
    b_t = 1.0
    c_t = 0.0
    d_t = 0.0

    actual = linear_canonical_transform(x, a=a_t, b=b_t, c=c_t, d=d_t)
    expected = torch.fft.fft(x, dim=-1, norm="ortho")

    assert torch.allclose(actual, expected.to(actual.dtype), atol=2e-5, rtol=1e-5)


def test_chirpz_matches_dense_reference_for_generic_matrix() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 64, dtype=torch.complex64)
    a_t, b_t, c_t = 0.63, 0.58, -0.21
    d_t = symplectic_d(a_t, b_t, c_t)

    dense = linear_canonical_transform(
        x,
        a=a_t,
        b=b_t,
        c=c_t,
        d=d_t,
        dense_threshold=10_000,
    )
    fast = chirpz_lct(x, a=a_t, b=b_t, c=c_t, d=d_t)

    assert torch.allclose(fast, dense, atol=2e-5, rtol=1e-5)


def test_b_zero_identity_branch_is_exact() -> None:
    torch.manual_seed(2)
    x = torch.randn(4, 19, dtype=torch.complex64)
    y = linear_canonical_transform(x, a=1.0, b=0.0, c=0.0, d=1.0)
    assert torch.equal(x, y)
