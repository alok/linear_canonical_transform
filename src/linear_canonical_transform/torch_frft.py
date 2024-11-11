import torch
import torch.nn as nn

from math import ceil

import torch


def idfrft(x: torch.Tensor, a: float | torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return dfrft(x, -a, dim=dim)


def dfrft(x: torch.Tensor, a: float | torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    dfrft_matrix = dfrftmtx(x.size(dim), a, device=x.device)
    dtype = torch.promote_types(dfrft_matrix.dtype, x.dtype)
    return torch.einsum(
        _get_dfrft_einsum_str(len(x.shape), dim),
        dfrft_matrix.type(dtype),
        x.type(dtype),
    )


def _get_dfrft_einsum_str(dim_count: int, req_dim: int) -> str:
    if req_dim < -dim_count or req_dim >= dim_count:
        raise ValueError("Dimension size error.")
    dim = torch.remainder(req_dim, torch.tensor(dim_count))
    diff = dim_count - dim
    remaining_str = "".join([chr(num) for num in range(98, 98 + diff)])
    return f"ab,...{remaining_str}->...{remaining_str.replace('b', 'a', 1)}"


def idfrftmtx(
    N: int,
    a: float | torch.Tensor,
    *,
    approx_order: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    return dfrftmtx(N, -a, approx_order=approx_order, device=device)


def dfrftmtx(
    N: int,
    a: float | torch.Tensor,
    *,
    approx_order: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError(
            "Necessary conditions for integers: N > 1 and approx_order >= 2."
        )
    evecs = _get_dfrft_evecs(N, approx_order=approx_order, device=device).type(
        torch.complex64
    )
    idx = _dfrft_index(N, device=device)
    evals = torch.exp(-1j * a * (torch.pi / 2) * idx).type(torch.complex64)
    dfrft_matrix = torch.einsum("ij,j,kj->ik", evecs, evals, evecs)
    return dfrft_matrix


def _get_dfrft_evecs(
    N: int,
    *,
    approx_order: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError(
            "Necessary conditions for integers: N > 1 and approx_order >= 2."
        )
    S = _create_hamiltonian(N, approx_order=approx_order, device=device)
    P = _create_odd_even_decomp_matrix(N, device=device)

    CS = torch.einsum("ij,jk,lk->il", P, S, P)
    C2 = CS[: N // 2 + 1, : N // 2 + 1]
    S2 = CS[N // 2 + 1 :, N // 2 + 1 :]

    _, VC = torch.linalg.eigh(C2)  # ascending order
    _, VS = torch.linalg.eigh(S2)  # ascending order

    N0, N1 = ceil(N / 2 - 1), N // 2 + 1
    qvc = torch.cat((VC, torch.zeros((N0, N1), device=device)))
    qvs = torch.cat((torch.zeros((N1, N0), device=device), VS))

    SC2 = torch.matmul(P, qvc).flip(-1)  # descending order
    SS2 = torch.matmul(P, qvs).flip(-1)  # descending order

    if N % 2 == 0:
        evecs = torch.zeros(N, N + 1, device=device)
        SS2_new = torch.hstack(
            (SS2, torch.zeros((SS2.size(0), 1), dtype=SS2.dtype, device=SS2.device))
        )
        evecs[:, : N + 1 : 2] = SC2
        evecs[:, 1:N:2] = SS2_new
        evecs = torch.hstack((evecs[:, : N - 1], evecs[:, -1].unsqueeze(-1)))
    else:
        evecs = torch.zeros(N, N, device=device)
        evecs[:, : N + 1 : 2] = SC2
        evecs[:, 1:N:2] = SS2
    return evecs


def _dfrft_index(N: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if N < 1:
        raise ValueError("N must be positive integer.")
    shift = 1 - (N % 2)  # 1 if N is even, 0 if N is odd
    last_entry = torch.tensor(N - 1 + shift, device=device)
    return torch.cat(
        (
            torch.arange(0, N - 1, dtype=torch.float32, device=device),
            last_entry.unsqueeze(-1),
        )
    )


def _circulant(vector: torch.Tensor) -> torch.Tensor:
    """
    Generate a circulant matrix based on the input vector.

    Parameters:
        vector (torch.Tensor): 1-dimensional PyTorch tensor representing
        the first row of the circulant matrix.

    Returns:
        torch.Tensor: The resulting circulant matrix.

    Example:
        >>> circulant(torch.tensor([1, 2, 3]))
        tensor([[1, 3, 2],
                [2, 1, 3],
                [3, 2, 1]])
    """
    vector = vector.flatten()
    size = vector.size(-1)
    idx = torch.arange(size, device=vector.device)
    indices = torch.remainder(idx[:, None] - idx, size)
    return vector[indices]


def _conv1d_full(vector: torch.Tensor, kernel1d: torch.Tensor) -> torch.Tensor:
    """
    Perform full 1-dimensional convolution on 1-dimensional input tensor and kernel.

    Parameters:
        input (torch.Tensor): Input 1-dimensional tensor.
        kernel (torch.Tensor): Convolution kernel (also 1-dimensional).

    Returns:
        torch.Tensor: Resulting 1-dimensional convolution with full padding.

    Example:
        >>> conv1d_full(torch.tensor([1, 2, 3, 4]), torch.tensor([1, -1, 2]))
        tensor([1, 1, 3, 5, 2, 8])
    """
    padding_size = kernel1d.size(0) - 1
    padded_input = torch.nn.functional.pad(
        vector, (padding_size, padding_size), mode="constant", value=0
    )
    conv_output = torch.conv1d(
        padded_input.view(1, 1, -1), kernel1d.view(1, 1, -1).flip(-1)
    )
    return conv_output.reshape(-1)


def _create_hamiltonian(
    N: int,
    *,
    approx_order: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError(
            "Necessary conditions for integers: N > 1 and approx_order >= 2."
        )

    order = approx_order // 2
    dum0 = torch.tensor([1.0, -2.0, 1.0], device=device)
    dum = dum0.clone()
    s = torch.zeros(1, device=device)

    for k in range(1, order + 1):
        coefficient = (
            2
            * (-1) ** (k - 1)
            * torch.prod(torch.arange(1, k, device=device)) ** 2
            / torch.prod(torch.arange(1, 2 * k + 1, device=device))
        )
        s = (
            coefficient
            * torch.cat(
                (
                    torch.zeros(1, device=device),
                    dum[k + 1 : 2 * k + 1],
                    torch.zeros(N - 1 - 2 * k, device=device),
                    dum[:k],
                )
            )
            + s
        )
        dum = _conv1d_full(dum, dum0)

    return _circulant(s) + torch.diag(torch.real(torch.fft.fft(s)))


def _create_odd_even_decomp_matrix(
    N: int, *, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    if N < 1:
        raise ValueError("N must be positive integer.")

    x1 = torch.ones(1 + N // 2, dtype=torch.float32, device=device)
    x2 = -torch.ones(N - N // 2 - 1, dtype=torch.float32, device=device)
    diagonal = torch.diag(torch.cat((x1, x2)))
    anti = torch.diag(torch.ones(N - 1, device=device), -1).rot90()
    P = (diagonal + anti) / torch.sqrt(torch.tensor(2.0))

    P[0, 0] = 1
    if N % 2 == 0:
        P[N // 2, N // 2] = 1
    return P


import torch
from torch.fft import fft, fftshift, ifft


def frft_shifted(
    fc: torch.Tensor, a_param: float | torch.Tensor, *, dim: int = -1
) -> torch.Tensor:
    """
    Shift the input tensor before applying FrFT to match NumPy's FrFT instead of MATLAB's FrFT.
    """
    return fftshift(frft(fftshift(fc, dim=dim), a_param, dim=dim), dim=dim)


def ifrft(
    fc: torch.Tensor, a_param: float | torch.Tensor, *, dim: int = -1
) -> torch.Tensor:
    return frft(fc, -a_param, dim=dim)


def frft(
    fc: torch.Tensor, a_param: float | torch.Tensor, *, dim: int = -1
) -> torch.Tensor:
    N = fc.size(dim)
    if N % 2 == 1:
        raise ValueError("signal size must be even")

    # 4-modulation and shifting to [-2, 2] interval
    if not isinstance(a_param, torch.Tensor):
        a_param = torch.tensor(a_param)
    a = torch.fmod(a_param, 4)
    if a > 2:
        a -= 4
    elif a < -2:
        a += 4

    # special integer cases with zero gradient, hence the a * zeros
    if a == 0.0:
        return fc + a * torch.zeros_like(fc, device=fc.device)
    elif a == 2.0 or a == -2.0:
        return _dflip(fc, dim=dim) + a * torch.zeros_like(fc, device=fc.device)

    biz = _bizinter(fc, dim=dim)
    zeros = torch.zeros_like(biz, device=fc.device).index_select(
        dim, torch.arange(0, N, device=fc.device)
    )
    fc = torch.cat([zeros, biz, zeros], dim=dim)

    res = fc
    if (0 < a < 0.5) or (1.5 < a < 2):
        res = _corefrmod2(fc, torch.tensor(1.0), dim=dim)
        a -= 1

    if (-0.5 < a < 0) or (-2 < a < -1.5):
        res = _corefrmod2(fc, torch.tensor(-1.0), dim=dim)
        a += 1

    res = _corefrmod2(res, a, dim=dim)
    res = torch.index_select(
        res, dim=dim, index=torch.arange(N, 3 * N, device=fc.device)
    )
    res = _bizdec(res, dim=dim)

    # Double the first entry of the vector in the given dimension,
    # res[0] *= 2 in n-dimensional case, i.e., Hadamard product with
    # [2, 1, 1, ..., 1] along n-th axis.
    first_entry_doubler_vec = torch.ones(res.size(dim), device=fc.device)
    first_entry_doubler_vec[0] = 2
    res = _vecmul_ndim(res, first_entry_doubler_vec, dim=dim)
    return res


def _dflip(tensor: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    first, remaining = torch.tensor_split(tensor, (1,), dim=dim)
    return torch.concat((first, remaining.flip(dim)), dim=dim)


def _bizdec(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    k = torch.arange(0, x.size(dim), 2, device=x.device)
    return x.index_select(dim, k)


def _bizinter(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    if x.is_complex():
        return _bizinter_real(x.real, dim=dim) + 1j * _bizinter_real(x.imag, dim=dim)
    else:
        return _bizinter_real(x, dim=dim)


def _bizinter_real(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    N = x.size(dim)
    N1 = N // 2 + (N % 2)
    N2 = 2 * N - (N // 2)

    upsampled = _upsample2(x, dim=dim)
    xf = fft(upsampled, dim=dim)
    xf = torch.index_fill(xf, dim, torch.arange(N1, N2, device=x.device), 0)
    return 2 * torch.real(ifft(xf, dim=dim))


def _upsample2(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    upsampled = x.repeat_interleave(2, dim=dim)
    idx = torch.arange(1, upsampled.size(dim), 2, device=x.device)
    return torch.index_fill(upsampled, dim, idx, 0)


def _corefrmod2(
    signal: torch.Tensor, a: torch.Tensor, *, dim: int = -1
) -> torch.Tensor:
    # constants
    N = signal.size(dim)
    Nend = N // 2
    Nstart = -(N % 2 + Nend)
    deltax = torch.sqrt(torch.tensor(N, device=signal.device))

    phi = a * torch.pi / 2
    alpha = -1j * torch.pi * torch.tan(phi / 2)
    beta = 1j * torch.pi / torch.sin(phi)

    Aphi_num = torch.exp(-1j * (torch.pi * torch.sign(torch.sin(phi)) / 4 - phi / 2))
    Aphi_denum = torch.sqrt(torch.abs(torch.sin(phi)))
    Aphi = Aphi_num / Aphi_denum

    # Chirp Multiplication
    x = torch.arange(Nstart, Nend, device=signal.device) / deltax
    chirp = torch.exp(alpha * x**2)
    multip = _vecmul_ndim(signal, chirp, dim=dim)

    # Chirp Convolution
    t = torch.arange(-N + 1, N, device=signal.device) / deltax
    hlptc = torch.exp(beta * t**2)

    N2 = hlptc.size(0)
    next_power_two = 2 ** torch.ceil(torch.log2(torch.tensor(N2 + N - 1))).int()
    Hc = ifft(
        _vecmul_ndim(
            fft(multip, n=next_power_two, dim=dim),
            fft(hlptc, n=next_power_two),
            dim=dim,
        ),
        dim=dim,
    )
    Hc = torch.index_select(
        Hc, dim, torch.arange(N - 1, 2 * N - 1, device=signal.device)
    )

    # Chirp Multiplication
    result = _vecmul_ndim(Hc, Aphi * chirp, dim=dim) / deltax

    # Adjustment
    if N % 2 == 1:
        return torch.roll(result, -1, dims=(dim,))
    else:
        return result


def _vecmul_ndim(
    tensor: torch.Tensor,
    vector: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """
    Multiply two tensors (`torch.mul()`) along a given dimension.
    """
    return torch.einsum(_get_mul_dim_einstr(len(tensor.shape), dim), tensor, vector)


def _get_mul_dim_einstr(dim_count: int, req_dim: int) -> str:
    if req_dim < -dim_count or req_dim >= dim_count:
        raise ValueError("Dimension size error.")
    dim = torch.remainder(req_dim, torch.tensor(dim_count))
    diff = dim_count - dim
    remaining_str = "".join([chr(num) for num in range(97, 97 + diff)])
    return f"...{remaining_str},a->...{remaining_str}"


class FrFTLayer(nn.Module):
    def __init__(
        self, order: float = 1.0, *, dim: int = -1, trainable: bool = True
    ) -> None:
        super().__init__()
        self.order = nn.Parameter(
            torch.tensor(order, dtype=torch.float32),
            requires_grad=trainable,
        )
        self.dim = dim

    def __repr__(self) -> str:
        return f"FrFTLayer(order={self.order.item()}, dim={self.dim})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return frft(x, self.order, dim=self.dim)


class DFrFTLayer(nn.Module):
    def __init__(
        self, order: float = 1.0, *, dim: int = -1, trainable: bool = True
    ) -> None:
        super().__init__()
        self.order = nn.Parameter(
            torch.tensor(order, dtype=torch.float32),
            requires_grad=trainable,
        )
        self.dim = dim

    def __repr__(self) -> str:
        return f"DFrFTLayer(order={self.order.item()}, dim={self.dim})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dfrft(x, self.order, dim=self.dim)


def test_frft_matches_fft() -> None:
    """Test that FrFT with order=1 matches regular FFT.

    Based on torch-frft package's implementation which matches MATLAB's fracF.m.
    We need to use fftshift for proper comparison since FrFT is defined on
    centered grid [-N/2, (N-1)/2] while FFT uses [0, N-1].
    """
    # Create random complex input tensor
    N = 32  # Must be even for frft
    x = torch.randn(N, dtype=torch.complex64) + 1j * torch.randn(
        N, dtype=torch.complex64
    )

    # FrFT with order=1 should match FFT after proper shifting
    frft_result = fftshift(frft(fftshift(x), a_param=1.0))
    fft_result = fft(x, norm="ortho")  # Match MATLAB normalization
    # need inner fftshift else every other value is negated
    # Test that 4 transforms is identity (a=1 four times)
    x_recovered = x
    # Test additivity: FrFT(a) + FrFT(b) = FrFT(a+b)
    # additivity fails, numerical errors?
    frft_ab = frft_shifted(frft_shifted(x, a_param=.5), a_param=.7)
    frft_sum = frft_shifted(x, a_param=.5 + .7)
    
    if not torch.allclose(frft_ab, frft_sum, rtol=1e-5, atol=1e-5):
        print("First 5 values of FrFT(a+b):", frft_sum[:5])
        print("First 5 values of FrFT(b)(FrFT(a)):", frft_ab[:5])
        print("Max difference:", torch.max(torch.abs(frft_sum - frft_ab)))
        raise AssertionError("FrFT should be additive in its parameter")
    for _ in range(12):
        x_recovered = frft_shifted(x_recovered, a_param=1.0)
    # Check results match within numerical tolerance, its about 2e-6 to 5e-6
    if not torch.allclose(frft_result, fft_result, rtol=1e-5, atol=1e-5):
        print("First 5 values of shifted frft_result:", frft_result[:5])
        print("First 5 values of fft_result:", fft_result[:5])
        print("Ratio of norms:", torch.norm(frft_result) / torch.norm(fft_result))
        print(
            "Ratio to sqrt(N):",
            torch.norm(frft_result)
            / torch.norm(fft_result)
            / torch.sqrt(torch.tensor(N)),
        )
        raise AssertionError("FrFT with order=1 should match FFT after shifting")
    
    print("Max absolute difference:", torch.max(torch.abs(x - x_recovered)))
    print("Mean absolute difference:", torch.mean(torch.abs(x - x_recovered)))
    if not torch.allclose(x, x_recovered, rtol=1e-5, atol=1e-5):
        print("First 5 values of original x:", x[:5])
        print("First 5 values of recovered x:", x_recovered[:5])
        print("Ratio of norms:", torch.norm(x_recovered) / torch.norm(x))
        print(
            "Ratio to sqrt(N):",
            torch.norm(x_recovered) / torch.norm(x) / torch.sqrt(torch.tensor(N)),
        )
        raise AssertionError("Inverse FrFT should recover original signal")


test_frft_matches_fft()
