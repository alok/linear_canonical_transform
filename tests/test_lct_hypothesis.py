from __future__ import annotations

import math

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from lct_activation import LCTLayer, linear_canonical_transform, symplectic_d


@st.composite
def tensor_shapes(draw):
    batch = draw(st.integers(min_value=1, max_value=4))
    length = draw(st.integers(min_value=2, max_value=64))
    return batch, length


@settings(deadline=None, max_examples=25)
@given(shape=tensor_shapes(), seed=st.integers(min_value=0, max_value=10_000))
def test_identity_special_case_matches_input(shape: tuple[int, int], seed: int) -> None:
    batch, length = shape
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, length, dtype=torch.complex64, generator=generator)

    y = linear_canonical_transform(x, a=1.0, b=0.0, c=0.0, d=1.0)

    assert torch.equal(y, x)


@settings(deadline=None, max_examples=25)
@given(shape=tensor_shapes(), seed=st.integers(min_value=0, max_value=10_000))
def test_fourier_special_case_matches_torch_fft(shape: tuple[int, int], seed: int) -> None:
    batch, length = shape
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, length, dtype=torch.complex64, generator=generator)

    layer = LCTLayer(a=0.0, b=1.0, c=0.0, normalized=True)
    expected = torch.fft.fft(x, dim=-1, norm="ortho")

    assert torch.allclose(layer(x), expected, atol=1e-5, rtol=0.0)


@settings(deadline=None, max_examples=25)
@given(shape=tensor_shapes(), seed=st.integers(min_value=0, max_value=10_000))
def test_inverse_fourier_special_case_matches_torch_ifft(shape: tuple[int, int], seed: int) -> None:
    batch, length = shape
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, length, dtype=torch.complex64, generator=generator)

    y = linear_canonical_transform(x, a=0.0, b=-1.0, c=0.0, d=0.0)
    expected = torch.fft.ifft(x, dim=-1, norm="ortho")

    assert torch.allclose(y, expected, atol=1e-5, rtol=0.0)


@settings(deadline=None, max_examples=20)
@given(
    length=st.integers(min_value=2, max_value=32),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_laplace_special_case_matches_minus_i_fft(length: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(3, length, dtype=torch.complex64, generator=generator)

    y = linear_canonical_transform(x, a=0j, b=1j, c=1j, d=0j)
    expected = -1j * torch.fft.fft(x, dim=-1, norm="ortho")

    assert torch.allclose(y, expected, atol=2e-5, rtol=0.0)


@settings(deadline=None, max_examples=30)
@given(
    a=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    c=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_symplectic_d_satisfies_determinant_constraint(a: float, b: float, c: float) -> None:
    if abs(a) < 1e-3:
        a = 1e-3 if a >= 0 else -1e-3

    d = symplectic_d(a, b, c)
    det = a * d - b * c

    assert math.isfinite(float(det))
    assert abs(det - 1.0) <= 1e-4


@settings(deadline=None, max_examples=30)
@given(angle=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False))
def test_fractional_fourier_factory_matches_trig_definition(angle: float) -> None:
    layer = LCTLayer.fractional_fourier(angle)
    a, b, c, _d = layer.canonical_matrix

    assert torch.isclose(a.real, torch.tensor(math.cos(angle), dtype=a.real.dtype), atol=1e-6)
    assert torch.isclose(b.real, torch.tensor(math.sin(angle), dtype=b.real.dtype), atol=1e-6)
    assert torch.isclose(c.real, torch.tensor(-math.sin(angle), dtype=c.real.dtype), atol=1e-6)
