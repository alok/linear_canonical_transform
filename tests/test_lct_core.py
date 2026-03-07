from __future__ import annotations

import torch

from lct_activation import LCTLayer, linear_canonical_transform, symplectic_d


def test_symplectic_d_is_stable_near_zero_a() -> None:
    a = torch.tensor(1e-10)
    b = torch.tensor(0.9)
    c = torch.tensor(-1.1)
    d = symplectic_d(a, b, c)
    assert torch.isfinite(torch.as_tensor(d)).all()


def test_lctlayer_default_is_identity_for_real_input() -> None:
    layer = LCTLayer()
    x = torch.randn(3, 32, dtype=torch.float32)
    y = layer(x)
    assert torch.allclose(y, x, atol=1e-4, rtol=0.0)


def test_lctlayer_fourier_matches_torch_fft() -> None:
    layer = LCTLayer(a=0.0, b=1.0, c=0.0, normalized=True)
    x = torch.randn(4, 64, dtype=torch.complex64)
    y = layer(x)
    ref = torch.fft.fft(x, dim=-1, norm="ortho")
    assert torch.allclose(y, ref, atol=1e-4, rtol=0.0)


def test_fast_path_matches_dense_reference() -> None:
    x = torch.randn(2, 384, dtype=torch.complex64)
    a, b, c = 0.73, 0.61, -0.42
    d = symplectic_d(a, b, c)

    dense = linear_canonical_transform(
        x,
        a=a,
        b=b,
        c=c,
        d=d,
        normalized=True,
        dense_threshold=4096,
        unitary_projection=False,
    )
    fast = linear_canonical_transform(
        x,
        a=a,
        b=b,
        c=c,
        d=d,
        normalized=True,
        dense_threshold=64,
        unitary_projection=False,
    )
    assert torch.allclose(fast, dense, atol=3e-5, rtol=0.0)

