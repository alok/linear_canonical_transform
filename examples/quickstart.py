from __future__ import annotations

import torch

from lct_activation import LCTLinear


def main() -> None:
    torch.manual_seed(0)
    layer = LCTLinear(16, 16)
    x = torch.randn(2, 16)
    y = layer(x)
    dense = layer.to_linear()

    print("input", tuple(x.shape))
    print("output", tuple(y.shape))
    print("matches dense materialization", torch.allclose(y, dense(x), atol=1e-4, rtol=0.0))


if __name__ == "__main__":
    main()
