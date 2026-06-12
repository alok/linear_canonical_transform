# Building the MLX Backend by Hand: A Developer's Guide

This is the long-form companion to the MLX backend change
(`src/lct_activation/mlx.py`, `tests/test_mlx_backend.py`,
`scripts/bench_mac_local.py`). It explains the surrounding system, every design
decision, and enough background that you could reimplement the change yourself
without reading the diff.

## 1. What this package is, in one screen

The package implements **finite-dimensional Linear Canonical Transforms
(LCTs)** as PyTorch layers. An LCT is the two-parameter-family generalization
of the Fourier transform: it is parameterized by a 2x2 matrix

```
[ a  b ]
[ c  d ]      with  ad - bc = 1
```

acting on time-frequency space. Special cases: `(0, 1, 0, ...)` is the Fourier
transform, `(cos θ, sin θ, -sin θ)` is the fractional Fourier transform (FrFT),
`(1, λz, 0)` is Fresnel propagation, and `b = 0` cases are chirp-multiplied
rescalings. Because `ad - bc = 1`, three parameters `(a, b, c)` determine the
fourth: `d = (1 + bc) / a` (with a regularized fallback near `a = 0` —
see `symplectic_d`).

On a finite grid you cannot have everything at once: a discrete LCT cannot
simultaneously be unitary (energy preserving), compose like the underlying
matrices, and sample the continuum kernel. The package makes the tradeoff
explicit with `normalization="unitary"` vs `"compositional"`, and (for the
sampled dense kernel) an optional QR projection to the nearest-ish unitary.

Two model-facing modules are built on top of the transform:

- **`LCTModReLU` / `LCTActivation`** — pack real channels into complex pairs,
  transform, apply modReLU (`relu(|z| + bias) * z / (|z| + eps)`), optionally
  inverse-transform, unpack. A genuinely nonlinear activation that acts in the
  LCT domain. Learnables: per-complex-channel `bias`, scalar `output_gain`,
  scalar `residual_mix`, and (in torch) the transform parameters.
- **`LCTLinear`** — pack, transform (FFT in the default `(0, 1, 0)` case),
  multiply by a learnable complex diagonal, inverse transform, unpack, crop.
  This is an `O(N log N)` structured replacement for a dense `nn.Linear`,
  identity-like at init.

The PyTorch forward (`functional/lct.py`) dispatches across five numerical
paths, in this order:

1. `|b| <= b_eps`: no Fourier component; the kernel degenerates to a
   chirp-multiplied resampling. Implemented with `grid_sample`.
2. FFT special case (`a≈0, b≈±1, c≈0`): plain `torch.fft.fft/ifft`.
3. A `(0, i, i)` Laplace-like special case via a small dense kernel.
4. `|b| = 1`: decompose as chirp-multiply → FFT → chirp-multiply.
5. Otherwise: dense `N x N` kernel for `N <= dense_threshold`, else
   **Bluestein / chirp-z**, which evaluates a chirp transform of any length
   as one FFT convolution (`O(N log N)` for any `b`).

Keep this dispatch in your head; the whole MLX port is "reproduce these
branches exactly".

## 2. Why an MLX backend at all

The research goal of the moment: *run the LCT activation inside real models on
a Mac, fast*. Two backends matter on Apple silicon:

- **torch MPS** — works (complex FFT support landed in recent torch), but
  per-op overhead is high for small tensors, and complex tensor support still
  has gaps (e.g. no complex `linalg.qr` on MPS).
- **MLX** — Apple's array framework, native to unified memory, lazy, with
  complex FFTs and autodiff. Typically the fastest path for FFT-ish workloads
  at transformer widths on M-series chips.

The benchmark added in this change confirms both intuitions: at dim 1024 the
MLX LCT activation forward is ~2.4x faster than torch MPS; at dim 4096 the
structured `LCTLinear` is ~5x faster than dense `nn.Linear` on MPS and ~2.6x
on MLX.

## 3. The one architectural decision that shapes everything

**MLX is lazy.** `mx.array` operations build a graph; nothing computes until
`mx.eval` (or an implicit force like `.item()`). Function transforms like
`mx.value_and_grad` *trace* your Python function with abstract arrays.

The PyTorch forward branches on the *values* of the transform parameters every
call (`float(torch.abs(b_c))`, etc.). That's fine in an eager framework. Under
MLX tracing it is not: calling `.item()` on a traced array either forces an
eval mid-trace or errors, and a Python `if` on a traced value cannot be
recorded into the graph.

So the MLX port makes the transform parameters `(a, b, c)` **fixed Python
complex numbers, set at construction**. This buys three things:

1. **Branch selection happens once**, at plan-compile time, on concrete
   values — exactly mirroring the torch dispatch order.
2. **All tables are precomputable**: chirps, Bluestein convolution kernels
   (including its FFT!), dense kernels, resampling weights. The dense kernels
   and chirp tables are taken *verbatim from the torch reference* at
   plan-compile time (see the parity traps below for why bit-sharing beats
   "equally accurate"); the Bluestein tables are NumPy `complex128`, matching
   the torch CPU work precision. Everything is cast to `complex64` MLX
   constants. The runtime forward is nothing but FFTs, elementwise complex
   multiplies, and matmuls. This is *faster* than the torch implementation,
   which rebuilds chirp tables and re-checks branches every forward because
   its parameters are (in principle) learnable.
3. **Cross-backend reproducibility becomes checkable**: with shared tables,
   torch-vs-MLX parity is ~1e-6 on every branch at every size, so "train in
   one backend, run in the other" is a tested property rather than a hope.
   An earlier draft built the tables in float64 instead — more accurate in
   isolation (torch's float32 chirp phases carry ~1e-1 rad of noise at
   `N=2048`), but it silently diverged from the reference at scale.
   `test_chirp_path_accuracy_vs_float64` still pins the accuracy ordering
   against a float64 ground truth.

What you give up: learnable `(a, b, c)` in the MLX backend. The genuinely
learnable parts of both layers — modReLU `bias`/`output_gain`/`residual_mix`,
and `LCTLinear`'s `spectral_real`/`spectral_imag`/`bias` — do not affect
branch selection and remain ordinary trainable MLX parameters. For the
activation use case (`a=0, b=1, c=0` → FFT path) this matches how the layer is
actually used; in practice even the torch version is normally run with
`learnable_transform=False` in `LCTLinear`.

### The "plan" pattern

The implementation centers on one function:

```python
_compile_plan(length, a, b, c, d, mode, centered,
              dense_threshold, b_eps, unitary_projection) -> Plan
# Plan = Callable[[mx.array], mx.array], operating on the last axis
```

It reproduces the torch dispatch in order, and each branch returns a closure
over precomputed `mx.array` constants. `LCTLayer` memoizes plans per feature
length in a private dict (`self._plans`), so a layer used at one width
compiles exactly once. MLX `nn.Module` ignores attributes whose names start
with `_` when collecting parameters, which is what keeps the plan cache and
the cached constants out of `trainable_parameters()`.

## 4. The numerical parity traps (where the bugs actually were)

If you reimplement this, these four traps are where you will lose hours.

### Trap 1: FFT normalization conventions

torch exposes `norm=` on `fft/ifft`: `"backward"` (unscaled fft, ifft
divides by N), `"ortho"` (both scale by `1/sqrt(N)`), `"forward"` (fft divides
by N, ifft unscaled). MLX has **no** `norm` argument: `mx.fft.fft` is
unscaled, `mx.fft.ifft` divides by N. So every torch `norm=` site must map to
an explicit scale:

| torch                  | MLX                      |
|------------------------|--------------------------|
| `fft(x, norm="ortho")` | `mx.fft.fft(x) / sqrt(N)`|
| `ifft(x, norm="ortho")`| `mx.fft.ifft(x) * sqrt(N)`|
| `ifft(x, norm="forward")`| `mx.fft.ifft(x) * N`   |

The unitary/compositional `NormMode` chooses between these per branch
(`_fft_norm` / `_ifft_norm` in the torch code; `_fft` / `_ifft` helpers in the
MLX module).

### Trap 2: torch rounds parameters through complex64 *before* promoting

The torch implementation does `a_c = torch.as_tensor(a, dtype=torch.complex64)`
at the top, and the chirp-z path then promotes those *rounded* values to
complex128. That float32 rounding (~1e-7 relative) looks harmless until it
multiplies `n² ≈ 2.6·10⁵` inside a phase: at `N=512` it produces ~0.07 rad of
phase error — a 7% output discrepancy if the MLX side uses the exact float64
parameter. The fix that makes the backends agree: round every parameter
through `complex(np.complex64(...))` at plan-compile, and also round the
*division results* `a/b`, `d/b`, `c*d` the same way, because torch computes
those in complex64 too. (This was found empirically: the chirp-z parity test
failed at 6.6e-2 relative until the rounding matched.)

### Trap 3: complex QR is only unique up to column phases — and the kernel is rank-deficient

The dense path with `normalization="unitary"` and `unitary_projection=True`
projects the kernel with QR. Two compounding problems:

1. NumPy's and torch's LAPACK paths return Q factors that differ by a
   per-column phase diagonal — both valid QRs, wildly different matrices
   elementwise (observed: relative error 1.23 between backends).
2. Worse: even using *torch's own* `torch.linalg.qr` on a kernel built with
   NumPy is not enough. The centered dense LCT kernel is numerically
   **rank-deficient** at moderate sizes (condition number ~1e9 at `N=64`),
   so QR amplifies 1-ulp build differences (`np.exp` vs `torch.exp` disagree
   in the last bits) into O(1) Q-factor differences. Each backend's kernel is
   individually unitary — they are just *different* unitaries. An angle scan
   of `LCTLayer.fractional_fourier` found 8 of 15 angles broken beyond 1e-3
   (worst: maxabs 3.3) while the original test suite passed at one lucky
   angle.

The robust fix: don't rebuild the kernel at all. Apply the torch
implementation to an identity matrix at plan-compile time
(`torch_lct(eye)` returns exactly the matrix of the linear map, QR
projection included) and ship those bits as the MLX constant. The same
rank-deficiency argument forces the layer's `d = (1+bc)/a` to be computed
with torch's float32 arithmetic rather than in Python float64 — a ~1e-7
difference in `d` alone reproduced O(1) layer divergence. MLX cannot QR
on-device regardless: `mx.linalg.qr` rejects complex inputs as of 0.31.

### Trap 4: `grid_sample` semantics in the `b ≈ 0` branch

torch implements the `b ≈ 0` resampling with
`grid_sample(mode="bilinear", padding_mode="zeros", align_corners=True)` over
a width-N "image". With `align_corners=True`, a normalized grid coordinate
`g ∈ [-1, 1]` maps to position `p = (g + 1)/2 · (N-1)`. Work through the
torch code's two cases and you get clean closed forms:

- centered: `p_k = Re(d)·(k - (N-1)/2) + (N-1)/2`
- uncentered: `p_k = Re(d)·k`

The MLX plan reproduces bilinear-with-zero-padding by hand: `i_lo = floor(p)`,
`i_hi = i_lo + 1`, weights `w_hi = p - i_lo`, `w_lo = 1 - w_hi`, **zero the
weight** of any tap whose true index is outside `[0, N-1]`, clamp indices for
the gather, then `mx.take(x, idx, axis=-1)` twice and add. The positions
depend only on `k`, never on the batch, so the index/weight vectors are
precomputed constants.

Also note the early-out: when `|d - 1| ≤ 1e-6` and `|c·d| ≤ 1e-6`, the branch
is the identity and the plan is literally `lambda x: x` (this mirrors a
fast-path that earlier NanoGPT benchmarking showed matters a lot).

### Bonus traps, briefly

- **Bluestein wrap-around**: the chirp kernel `q` lives on a power-of-two
  circle of length `1 << (2N-1).bit_length()`; entries `q[conv_len-(N-1):]`
  are the *reversed* tail `q_first[1:]`. Off-by-one here corrupts only the
  last few output samples — write the parity test before the code.
- **torch's chirp-z ignores `centered`** (`del centered` in `chirpz.py`).
  Mirror the quirk; don't "fix" it on one side only.
- **`mx.eye(n, dtype=mx.complex64)` fails on GPU** ("scatter does not support
  complex64"). Build identity matrices in NumPy and convert.
- **Complex gathers cannot backprop on Metal**: the vjp of `mx.take` is a
  scatter, and MLX has no complex64 scatter on GPU — so a complex gather in
  the `b ≈ 0` plan makes the whole branch raise at *training* time while
  every forward-only parity test passes. Gather the float32 real/imag planes
  separately and recombine. Always write at least one gradient test per
  branch, not just per layer.
- **Complex pair packing order**: `z[k] = x[2k] + i·x[2k+1]` (torch does this
  with `view_as_complex` on a `(..., N/2, 2)` reshape). Unpack by stacking
  `(Re, Im)` on a new last axis and flattening — then crop, since odd channel
  counts are padded up.

## 5. The module layer (MLX `nn.Module` mechanics)

Three classes mirror the torch ones:

- `LCTLayer`: holds `(a, b, c, d)` as Python complex attributes, plan caches
  in `_plans`/`_inverse_plans`. `inverse()` compiles a plan with parameters
  `(d, -b, -c, a)` — the inverse of a unit-determinant matrix. Real input is
  cast to complex64, transformed, then `mx.real(...)` cast back.
- `LCTModReLU` (aliased `LCTActivation`): identical math to torch, with
  `mx.maximum(radius + bias, 0)` for the relu. Watch dtype promotion: the
  gain is real, the spectrum complex; cast the gain to complex64 explicitly.
- `LCTLinear`: pack → (FFT | transform) → learnable complex diagonal
  multiply → (iFFT | inverse transform) → unpack → crop → bias. The complex
  diagonal is stored as two real arrays (`spectral_real`, `spectral_imag`)
  exactly like torch, which keeps every optimizer happy and lets parity tests
  copy parameters across backends directly.

MLX module conventions used (worth knowing if MLX is new to you):

- Any `mx.array` attribute is automatically a parameter; underscore-prefixed
  attributes are excluded. There is no `requires_grad`; you differentiate
  with `mlx.nn.value_and_grad(module, loss_fn)` (params) or
  `mx.value_and_grad(fn, argnums=...)` (explicit arguments).
- There is no `register_buffer`; precomputed constants are just underscore
  attributes.
- `mx.stop_gradient` is the `torch.no_grad()` analog used in
  `materialize_weight` / `to_linear`.

## 6. The parity test strategy

`tests/test_mlx_backend.py` runs the same NumPy input through both backends
(`pytest.importorskip("mlx.core")` keeps it green on machines without MLX):

1. **Functional parity, one case per branch** — FFT, iFFT, chirp, dense
   (projected and not), Bluestein, both `b≈0` sub-branches, compositional
   mode. Tolerances are 1e-4-ish except the chirp path at 2e-3, where torch's
   float32 tables are the floor.
2. **Accuracy-ordering test** — computes the chirp path in float64 NumPy as
   ground truth and asserts the MLX error is ≤ the torch error. This converts
   the loose tolerance above from "shrug" into a verified claim about *which
   side* is inaccurate.
3. **Layer parity** — activation (even/odd/large widths, inverse+residual
   variants) and linear (square/expand/shrink/odd shapes) with parameters
   copied across backends.
4. **Gradient smoke tests** — `value_and_grad` produces finite, nonzero
   grads for every trainable parameter.
5. **Behavioral tests** mirroring the torch suite: nonlinearity, zero
   preservation, shape/dtype, unitarity of the Fourier plan matrix.

The methodology that made this tractable: when a parity test fails, *don't
stare at the code* — compute the same case in float64 NumPy and compare all
three. Whichever backend is far from float64 owns the bug (or owns the
documented precision floor).

## 7. The benchmark

`scripts/bench_mac_local.py` measures `LCTActivation` vs `GELU` and
`LCTLinear` vs `nn.Linear` on torch-CPU, torch-MPS, and MLX, forward and
forward+backward, median of N steps after warmup.

Honest-timing rules it follows; copy them if you extend it:

- **torch MPS**: `torch.mps.synchronize()` inside the timed region (after the
  op), otherwise you measure kernel-launch time.
- **MLX**: `mx.eval(out)` inside the timed region, otherwise you measure
  graph construction.
- **fwd+bwd comparability**: torch differentiates w.r.t. input *and* params
  (`x.requires_grad_(True)` + module params). The MLX side does the same via
  `mx.value_and_grad(loss_fn, argnums=(0, 1))` over `(params, x)`, falling
  back to input-only for parameterless baselines like GELU.
- Output JSON carries a `results: [{name, tokens_per_second, ...}]` array, the
  generic shape `lct summarize-results` and `lct doctor --require-results`
  already understand, so the artifact slots into the existing evidence
  tooling without code changes.

Checked-in numbers live in `paper/results/bench_mac_local.json`; headline
findings are in the README table (LCTLinear crossover vs dense at ~1-2k
features; ~5x at 4096 on MPS; MLX fastest for the activation at 768-2048).

## 8. Packaging

- `pyproject.toml` gains `[project.optional-dependencies].mlx = ["mlx>=0.31"]`
  for downstream users, and the `dev` extra gains
  `mlx>=0.31; sys_platform == 'darwin' and platform_machine == 'arm64'` so a
  Mac dev setup gets it automatically while Linux CI resolves the same lock
  without MLX at all.
- The module is `lct_activation.mlx` — a submodule named like the upstream
  package. This is safe (absolute imports inside it resolve `mlx.core` to the
  top-level package; `sys.modules` keys `"mlx"` and `"lct_activation.mlx"`
  never collide) but worth knowing about when reading tracebacks.
- Nothing imports `lct_activation.mlx` unconditionally: not the package
  `__init__`, not the CLI. Importing it without MLX installed raises a clear
  `ImportError` with the install hint.

## 9. What adversarial review caught (and why parity probes earn their keep)

The change was reviewed by independent agents instructed to *refute* each
other's findings with numeric probes. Beyond the traps above, two findings
are worth internalizing:

- **The parity probe found a pre-existing torch bug.** Comparing input
  gradients between backends (with finite differences as referee) exposed
  that `reduce_unpacked_grad`'s tile-mode fallback used `i // repeats`
  (the adjoint of `repeat_interleave`) where the forward uses `repeat`
  (`i % original`). Net effect: `LCTLinear` computed wrong *input* gradients
  whenever `out_features > in_features` on the default path — losses and
  parameter grads looked fine, so nothing else caught it. The MLX side,
  which gets its backward from autodiff rather than a hand-written adjoint,
  was correct. A second implementation of the same math is one of the best
  bug detectors there is.
- **Benchmark harnesses accumulate subtle asymmetries.** The torch fwd+bwd
  loop never cleared `x.grad`, so every timed iteration after the first paid
  an `AccumulateGrad` add (~33 MB at dim 4096) the functional MLX side never
  performs. Fix: clear the input grad inside the timed closure. The fwd-only
  numbers were unaffected.

The linearity identity used for the gradient regression test is worth
remembering: for any linear layer, `d/dx sum(layer(x)·G) = G @ W` with `W`
the materialized weight — an exact oracle that needs no finite differences.

## 10. What's deliberately *not* in this change

- No MLX spectral-FrFT diagnostics (the torch ones remain the only
  implementation; they are research APIs, not layer backends).
- No learnable `(a, b, c)` under MLX (see section 3; revisit only if a real
  experiment needs it — the path would be: fix the branch at compile time,
  recompute chirps from traced parameters inside the plan, accept that branch
  boundaries are frozen).
- No `mx.compile` integration. Worth trying for another speedup, but it
  interacts with plan caching and wasn't needed to beat torch-MPS.
- No NanoGPT-on-MLX integration; the torch NanoGPT path stays the training
  evidence vehicle.
