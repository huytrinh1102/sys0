"""
Custom GPU kernels for LinOSS.

`linoss_scan(x, M, f)` runs the recurrence with Triton on CUDA tensors and falls back to the FFT
convolution from `torch_linoss.ops` elsewhere (CPU, or when Triton is not installed). `linoss_step`
is the single-step update for streaming inference.
"""

import torch

from torch_linoss.ops import fft_causal_conv_lag, linoss_kernel, linoss_step

try:
    from torch_linoss.cuda_ops.linoss_scan import linoss_scan as _triton_linoss_scan

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def linoss_scan(x, M, f, h0=None, return_final_state=False):
    """See `torch_linoss.cuda_ops.linoss_scan.linoss_scan`."""
    if HAS_TRITON and x.is_cuda:
        return _triton_linoss_scan(x, M, f, h0=h0, return_final_state=return_final_state)
    if h0 is not None or return_final_state:
        raise NotImplementedError("initial/final states need the Triton kernel (CUDA)")
    y = fft_causal_conv_lag(x, linoss_kernel(M, f, x.shape[-1]))
    return y.to(x.dtype)


__all__ = ["HAS_TRITON", "linoss_scan", "linoss_step"]
