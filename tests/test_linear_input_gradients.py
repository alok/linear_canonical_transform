"""Input-gradient correctness for LCTLinear's custom autograd path.

LCTLinear is a *linear* map (plus bias), so for loss ``sum(layer(x) * G)``
the input gradient must equal ``G @ W`` where ``W`` is the materialized
weight. This catches adjoint bugs in the manual backward (the tile-expansion
reduce used the wrong index mapping until it was checked this way).
"""

from __future__ import annotations

import pytest
import torch

import lct_activation.triton_ops as triton_ops
from lct_activation import LCTLinear
from lct_activation.triton_ops import pack_real_pairs, reduce_unpacked_grad


class _CudaLikeTensor:
    is_cuda = True

    def __init__(self, *, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


def test_raw_triton_dispatch_defers_to_autograd() -> None:
    original_has_triton = triton_ops.HAS_TRITON
    triton_ops.HAS_TRITON = True
    try:
        trainable = _CudaLikeTensor(requires_grad=True)
        frozen = _CudaLikeTensor(requires_grad=False)
        with torch.enable_grad():
            assert not triton_ops._raw_triton_enabled(True, trainable)
            assert triton_ops._raw_triton_enabled(True, frozen)
        with torch.no_grad():
            assert triton_ops._raw_triton_enabled(True, trainable)
    finally:
        triton_ops.HAS_TRITON = original_has_triton


@pytest.mark.parametrize(
    "in_features,out_features",
    [(16, 16), (16, 32), (32, 16), (15, 17), (8, 24)],
)
def test_lct_linear_input_gradient_matches_materialized_weight(
    in_features: int, out_features: int
) -> None:
    torch.manual_seed(0)
    layer = LCTLinear(in_features, out_features)
    with torch.no_grad():
        layer.spectral_real.normal_()
        layer.spectral_imag.normal_()

    x = torch.randn(4, in_features, requires_grad=True)
    g = torch.randn(4, out_features)

    (layer(x) * g).sum().backward()

    weight = layer.materialize_weight()
    expected = g @ weight
    assert x.grad is not None
    torch.testing.assert_close(x.grad, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("original,expanded", [(8, 16), (5, 16), (6, 20)])
def test_tile_reduce_is_adjoint_of_tile_expansion(original: int, expanded: int) -> None:
    torch.manual_seed(1)
    x = torch.randn(3, original)
    grad = torch.randn(3, expanded)

    repeats = (expanded + original - 1) // original
    expanded_x = x.repeat(1, repeats)[..., :expanded]
    reduced = reduce_unpacked_grad(
        grad, original_channels=original, expanded_channels=expanded, mode="tile"
    )

    lhs = (expanded_x * grad).sum()
    rhs = (x * reduced).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_pack_real_pairs_tile_autograd_matches_manual_reduce() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 6, requires_grad=True)
    packed = pack_real_pairs(x, 16, mode="tile")
    grad_complex = torch.randn(2, 8, dtype=torch.complex64)
    (packed * grad_complex.conj()).real.sum().backward()

    grad_unpacked = torch.view_as_real(grad_complex).reshape(2, 16)
    manual = reduce_unpacked_grad(
        grad_unpacked, original_channels=6, expanded_channels=16, mode="tile"
    )
    assert x.grad is not None
    torch.testing.assert_close(x.grad, manual, atol=1e-5, rtol=1e-5)
