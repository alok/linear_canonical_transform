"""MLX quickstart: build a tiny LCT-based MLP and train it for a few steps.

Requires the optional ``mlx`` extra (pulled automatically by
``uv sync --extra dev`` on Apple silicon).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from lct_activation.mlx import LCTActivation, LCTLinear


class TinyLCTMLP(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.up = LCTLinear(dim, hidden)
        self.act = LCTActivation(hidden)
        self.down = LCTLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(self.act(self.up(x)))


def main() -> None:
    mx.random.seed(0)
    dim, hidden = 32, 64
    model = TinyLCTMLP(dim, hidden)

    # Fit a fixed random nonlinear target.
    x = mx.random.normal((256, dim))
    target = mx.tanh(x @ mx.random.normal((dim, dim))) * 0.5

    def loss_fn(m: TinyLCTMLP) -> mx.array:
        return mx.mean(mx.square(m(x) - target))

    value_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=1e-2)

    first_loss = final_loss = 0.0
    for step in range(50):
        loss, grads = value_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        final_loss = loss.item()
        if step == 0:
            first_loss = final_loss
        if step % 10 == 0:
            print(f"step {step:3d} loss {final_loss:.5f}")

    print(f"first {first_loss:.5f} -> final {final_loss:.5f}")
    print("loss decreased", final_loss < first_loss)


if __name__ == "__main__":
    main()
