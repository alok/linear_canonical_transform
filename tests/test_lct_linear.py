from __future__ import annotations

import torch
import torch.nn.functional as F

from lct_activation import LCTLinear


def test_lct_linear_preserves_expected_shape() -> None:
    layer = LCTLinear(15, 11)
    x = torch.randn(4, 7, 15, dtype=torch.float32)
    y = layer(x)
    assert y.shape == (4, 7, 11)
    assert y.dtype == x.dtype


def test_lct_linear_is_identity_like_at_init() -> None:
    layer = LCTLinear(16, 16, bias=False)
    x = torch.randn(3, 16, dtype=torch.float32)
    y = layer(x)
    assert torch.allclose(y, x, atol=1e-4, rtol=0.0)


def test_materialized_weight_matches_forward() -> None:
    layer = LCTLinear(12, 9)
    x = torch.randn(5, 12, dtype=torch.float32)
    weight = layer.materialize_weight()
    y_direct = layer(x)
    y_dense = F.linear(x, weight, layer.bias)
    assert torch.allclose(y_direct, y_dense, atol=1e-4, rtol=0.0)


def test_lct_linear_is_linear_without_bias() -> None:
    layer = LCTLinear(10, 10, bias=False)
    x = torch.randn(2, 10, dtype=torch.float32)
    y = torch.randn(2, 10, dtype=torch.float32)
    lhs = layer(x + y)
    rhs = layer(x) + layer(y)
    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=0.0)


def test_to_linear_round_trips_dense_equivalent() -> None:
    layer = LCTLinear(14, 9)
    x = torch.randn(6, 14, dtype=torch.float32)
    dense = layer.to_linear()
    assert torch.allclose(layer(x), dense(x), atol=1e-4, rtol=0.0)
