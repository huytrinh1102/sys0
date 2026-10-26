"""
Depthwise causal convolution with the interface of `causal_conv1d_fn` from
https://github.com/Dao-AILab/causal-conv1d:

    out = causal_conv1d_fft(x, weight, bias=None, activation=None)
    x: (batch, dim, seqlen), weight: (dim, width), bias: (dim,), out: (batch, dim, seqlen)

with the same semantics as `F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]`,
i.e. `weight[:, -1]` multiplies the current input and `weight[:, -1 - k]` the input k steps in the past.

The CUDA kernel of causal-conv1d only supports width <= 4. LinOSS needs width == seqlen, so long filters
are computed with FFT (O(L log L)); short filters use cuDNN through `F.conv1d`.
"""

import torch
import torch.nn.functional as F

# Filters up to this width use the direct convolution.
DIRECT_CONV_MAX_WIDTH = 64


def next_fast_len(n):
    """Smallest m >= n whose prime factors are all in {2, 3, 5, 7} (fast sizes for cuFFT)."""
    m = n
    while True:
        r = m
        for p in (2, 3, 5, 7):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 1


def fft_causal_conv_lag(x, kernel, dim=-1):
    """Causal convolution y[t] = sum_k kernel[..., k] x[t - k] along `dim`, computed with FFT in fp32.

    Args:
        x: input, convolved along `dim`.
        kernel: (..., W) kernel in lag order (kernel[..., 0] multiplies the current input). Its leading
            dims must broadcast against x with `dim` moved last.
        dim: time dimension of x.
    Returns:
        y with the shape of x, in fp32 (fp64 if x is fp64).
    """
    L = x.shape[dim]
    W = kernel.shape[-1]
    n = next_fast_len(L + W - 1)
    # Half-precision cuFFT only supports power-of-2 sizes and loses accuracy; float64 is kept for tests.
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    xf = torch.fft.rfft(x.to(dtype), n=n, dim=dim)
    kf = torch.fft.rfft(kernel.to(dtype), n=n, dim=-1)
    if dim not in (-1, x.dim() - 1):
        kf = kf.movedim(-1, dim)
    y = torch.fft.irfft(xf * kf, n=n, dim=dim)
    return y.narrow(dim, 0, L)


def causal_conv1d_fft(x, weight, bias=None, activation=None):
    """Drop-in replacement for `causal_conv1d_fn` that supports any filter width.

    Args:
        x: (batch, dim, seqlen).
        weight: (dim, width).
        bias: (dim,) or None.
        activation: None, "silu" or "swish".
    Returns:
        (batch, dim, seqlen) in the dtype of x.
    """
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    dim, width = weight.shape
    seqlen = x.shape[-1]
    if width <= DIRECT_CONV_MAX_WIDTH:
        out = F.conv1d(
            F.pad(x, (width - 1, 0)), weight.unsqueeze(1).to(x.dtype), None, groups=dim
        )
    else:
        out = fft_causal_conv_lag(x, weight.flip(-1)).to(x.dtype)
    if bias is not None:
        out = out + bias.to(out.dtype).unsqueeze(-1)
    if activation is not None:
        out = F.silu(out)
    assert out.shape[-1] == seqlen
    return out
