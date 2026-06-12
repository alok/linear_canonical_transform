"""Parity tests between the PyTorch reference and the MLX backend.

Every test feeds the same NumPy input through both backends and compares the
outputs. Skipped entirely when ``mlx`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

import lct_activation as ref
import lct_activation.mlx as mlx_lct


def _rand(*shape: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def _rand_complex(*shape: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    re = rng.standard_normal(shape)
    im = rng.standard_normal(shape)
    return (re + 1j * im).astype(np.complex64)


def _torch_lct(x: np.ndarray, **params) -> np.ndarray:
    a, b, c = params.pop("a"), params.pop("b"), params.pop("c")
    d = params.pop("d", None)
    if d is None:
        d = ref.symplectic_d(a, b, c)
    y = ref.linear_canonical_transform(
        torch.from_numpy(x), a=a, b=b, c=c, d=d, **params
    )
    return y.numpy()


def _mlx_lct(x: np.ndarray, **params) -> np.ndarray:
    y = mlx_lct.linear_canonical_transform(mx.array(x), **params)
    return np.array(y)


@pytest.mark.parametrize(
    "params,length,atol",
    [
        # FFT special case (the default activation transform).
        ({"a": 0.0, "b": 1.0, "c": 0.0}, 64, 1e-5),
        # Inverse FFT special case.
        ({"a": 0.0, "b": -1.0, "c": 0.0}, 64, 1e-5),
        # |b| = 1 chirp-FFT-chirp path. The torch backend builds its chirp
        # tables in float32 while MLX precomputes them in float64, so the
        # parity floor here is torch's own ~1e-3 phase noise (see
        # test_chirp_path_accuracy_vs_float64 for the ground-truth check).
        ({"a": 0.5, "b": 1.0, "c": -0.5}, 64, 2e-3),
        # Dense kernel path (fractional Fourier, 30 degrees).
        (
            {
                "a": float(np.cos(np.pi / 6)),
                "b": float(np.sin(np.pi / 6)),
                "c": float(-np.sin(np.pi / 6)),
            },
            64,
            1e-4,
        ),
        # Bluestein chirp-z path (length above dense_threshold).
        ({"a": 0.6, "b": 0.7, "c": -0.4}, 512, 5e-4),
        # b ~= 0 resample branch with a non-identity scaling.
        ({"a": 2.0, "b": 0.0, "c": 0.0}, 64, 1e-4),
        # b ~= 0 identity-like branch with a chirp.
        ({"a": 1.0, "b": 0.0, "c": 0.3}, 64, 1e-4),
    ],
)
def test_functional_parity(params: dict, length: int, atol: float) -> None:
    x = _rand_complex(4, length, seed=1)
    expected = _torch_lct(x, **params)
    actual = _mlx_lct(x, **params)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-4)


def test_chirp_path_accuracy_vs_float64() -> None:
    """On the |b|=1 path the MLX backend should be at least as accurate as
    torch against a float64 ground truth (its chirp tables are float64)."""

    a, b, c = 0.5, 1.0, -0.5
    d = float(np.real(np.complex128(ref.symplectic_d(a, b, c))))
    n = 64
    x = _rand_complex(4, n, seed=1)

    idx = np.arange(n) - (n - 1) / 2.0
    chirp_in = np.exp(1j * np.pi * (a / b) * idx**2)
    chirp_out = np.exp(1j * np.pi * (d / b) * idx**2)
    amp = np.exp(-1j * np.pi / 4 * np.sign(b)) / np.sqrt(n)
    ground_truth = (
        amp
        * np.sqrt(n)
        * chirp_out
        * np.fft.fft(x.astype(np.complex128) * chirp_in, axis=-1, norm="ortho")
    )

    torch_err = np.abs(_torch_lct(x, a=a, b=b, c=c) - ground_truth).max()
    mlx_err = np.abs(_mlx_lct(x, a=a, b=b, c=c) - ground_truth).max()
    assert mlx_err <= torch_err + 1e-6


def test_functional_parity_compositional() -> None:
    params = {"a": 0.5, "b": 1.0, "c": -0.5}
    x = _rand_complex(4, 32, seed=2)
    expected = _torch_lct(x, normalization="compositional", **params)
    actual = _mlx_lct(x, normalization="compositional", **params)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_functional_parity_unprojected_dense() -> None:
    params = {
        "a": float(np.cos(1.0)),
        "b": float(np.sin(1.0)),
        "c": float(-np.sin(1.0)),
    }
    x = _rand_complex(4, 48, seed=3)
    expected = _torch_lct(x, unitary_projection=False, **params)
    actual = _mlx_lct(x, unitary_projection=False, **params)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_symplectic_d_matches_reference() -> None:
    for a, b, c in [(1.0, 0.5, 0.25), (0.0, 1.0, -1.0), (0.5, 1.0, -0.5)]:
        assert complex(ref.symplectic_d(a, b, c)) == pytest.approx(
            mlx_lct.symplectic_d(a, b, c)
        )


def test_activation_shape_and_dtype() -> None:
    act = mlx_lct.LCTActivation(15)
    x = mx.array(_rand(2, 4, 15))
    y = act(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_activation_is_nonlinear() -> None:
    act = mlx_lct.LCTActivation(16)
    x = mx.array(_rand(2, 16, seed=4))
    y = mx.array(_rand(2, 16, seed=5))
    lhs = act(x + y)
    rhs = act(x) + act(y)
    assert not np.allclose(np.array(lhs), np.array(rhs), atol=1e-5)


def test_activation_zero_stays_zero() -> None:
    act = mlx_lct.LCTActivation(16)
    y = act(mx.zeros((3, 16)))
    assert np.allclose(np.array(y), 0.0, atol=1e-6)


@pytest.mark.parametrize("channels", [16, 15, 1024])
def test_activation_parity_with_torch(channels: int) -> None:
    x = _rand(2, 8, channels, seed=6)

    torch_act = ref.LCTActivation(channels)
    with torch.no_grad():
        expected = torch_act(torch.from_numpy(x)).numpy()

    mlx_act = mlx_lct.LCTActivation(channels)
    actual = np.array(mlx_act(mx.array(x)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_activation_parity_inverse_and_residual() -> None:
    x = _rand(3, 32, seed=7)

    torch_act = ref.LCTActivation(
        32, inverse_after_nonlinearity=True, residual_mix=0.5
    )
    with torch.no_grad():
        expected = torch_act(torch.from_numpy(x)).numpy()

    mlx_act = mlx_lct.LCTActivation(
        32, inverse_after_nonlinearity=True, residual_mix=0.5
    )
    actual = np.array(mlx_act(mx.array(x)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "in_features,out_features", [(16, 16), (16, 32), (32, 16), (15, 17)]
)
def test_linear_parity_with_torch(in_features: int, out_features: int) -> None:
    x = _rand(4, in_features, seed=8)
    rng = np.random.default_rng(9)
    diag_real = rng.standard_normal(
        max(in_features, out_features) // 2
        + (max(in_features, out_features) % 2)
    ).astype(np.float32)
    diag_imag = rng.standard_normal(diag_real.shape[0]).astype(np.float32)
    bias = rng.standard_normal(out_features).astype(np.float32)

    torch_linear = ref.LCTLinear(in_features, out_features)
    with torch.no_grad():
        torch_linear.spectral_real.copy_(torch.from_numpy(diag_real))
        torch_linear.spectral_imag.copy_(torch.from_numpy(diag_imag))
        torch_linear.bias.copy_(torch.from_numpy(bias))
        expected = torch_linear(torch.from_numpy(x)).numpy()

    mlx_linear = mlx_lct.LCTLinear(in_features, out_features)
    mlx_linear.spectral_real = mx.array(diag_real)
    mlx_linear.spectral_imag = mx.array(diag_imag)
    mlx_linear.bias = mx.array(bias)
    actual = np.array(mlx_linear(mx.array(x)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_linear_materialized_weight_matches_forward() -> None:
    layer = mlx_lct.LCTLinear(16, 16)
    layer.spectral_imag = mx.array(
        np.random.default_rng(10).standard_normal(8).astype(np.float32) * 0.1
    )
    x = mx.array(_rand(4, 16, seed=11))
    dense = layer.to_linear()
    np.testing.assert_allclose(
        np.array(layer(x)), np.array(dense(x)), atol=1e-5, rtol=1e-5
    )


def test_layer_matrix_is_unitary_for_fourier() -> None:
    layer = mlx_lct.LCTLayer(a=0.0, b=1.0, c=0.0)
    m = np.array(layer.matrix(32))
    np.testing.assert_allclose(m @ m.conj().T, np.eye(32), atol=1e-5)


def test_activation_gradients_are_finite() -> None:
    import mlx.nn as mlx_nn

    act = mlx_lct.LCTActivation(64)
    x = mx.array(_rand(8, 64, seed=12))

    def loss_fn(model):
        return mx.mean(mx.square(model(x)))

    loss, grads = mlx_nn.value_and_grad(act, loss_fn)(act)
    mx.eval(loss, grads)
    assert np.isfinite(loss.item())
    bias_grad = np.array(grads["bias"])
    assert np.isfinite(bias_grad).all()
    assert np.abs(bias_grad).sum() > 0


def test_linear_gradients_are_finite() -> None:
    import mlx.nn as mlx_nn

    layer = mlx_lct.LCTLinear(64, 64)
    x = mx.array(_rand(8, 64, seed=13))
    target = mx.array(_rand(8, 64, seed=14))

    def loss_fn(model):
        return mx.mean(mx.square(model(x) - target))

    loss, grads = mlx_nn.value_and_grad(layer, loss_fn)(layer)
    mx.eval(loss, grads)
    assert np.isfinite(loss.item())
    for key in ("spectral_real", "spectral_imag", "bias"):
        grad = np.array(grads[key])
        assert np.isfinite(grad).all(), key
        assert np.abs(grad).sum() > 0, key
