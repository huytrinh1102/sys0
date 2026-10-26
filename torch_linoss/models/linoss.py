"""
Reference LinOSS model built on the pure-PyTorch ops (FFT convolution). Used for training and as the
numerical reference for the custom-kernel model in `linoss_cuda.py`.
"""

import torch

from torch_linoss.models.common import LinOSSLayerBase, LinOSSNet
from torch_linoss.ops import fft_causal_conv_lag, linoss_kernel, next_fast_len

PATHS = ("auto", "depthwise", "freq")


class LinOSSLayer(LinOSSLayerBase):
    """LinOSS layer computed as a long causal convolution.

    path:
        "depthwise": project to 2P channels, convolve each with its length-L kernel via FFT, project back.
        "freq": FFT the H input channels once and apply B, the kernel and C per frequency (complex
            matmuls), then inverse-FFT the H output channels. Fewer FFT channels when H < 2P.
        "auto": "freq" if H < 2P else "depthwise".
    """

    def __init__(self, ssm_size, H, discretization, A_eps=0.0, path="auto"):
        super().__init__(ssm_size, H, discretization, A_eps)
        if path not in PATHS:
            raise ValueError(f"Unknown path {path!r}, expected one of {PATHS}")
        self.path = path

    def resolved_path(self):
        if self.path == "auto":
            return "freq" if self.H < 2 * self.ssm_size else "depthwise"
        return self.path

    def mix(self, x):
        M, f = self.recurrence()
        kernel = linoss_kernel(M, f, x.shape[-1])
        return fft_causal_conv_lag(x, kernel.repeat(2, 1))

    def forward(self, u):
        if self.resolved_path() == "depthwise":
            return super().forward(u)
        return self._forward_freq(u)

    def _forward_freq(self, u):
        L = u.shape[1]
        M, f = self.recurrence()
        kernel = linoss_kernel(M, f, L)
        n = next_fast_len(2 * L - 1)
        real = torch.float64 if u.dtype == torch.float64 else torch.float32
        cplx = torch.complex128 if real == torch.float64 else torch.complex64
        with torch.autocast(device_type=u.device.type, enabled=False):
            uf = torch.fft.rfft(u.to(real), n=n, dim=1)  # (batch, F, H)
            kf = torch.fft.rfft(kernel.to(real), n=n, dim=-1).T  # (F, P)
            xf = uf @ self.in_proj_weight().T.to(cplx)  # (batch, F, 2P)
            xf = xf * kf.repeat(1, 2)
            yf = xf @ self.out_proj_weight().T.to(cplx)  # (batch, F, H)
            # Contiguous so downstream elementwise ops (and dropout masks) see the same layout as "depthwise".
            y = torch.fft.irfft(yf, n=n, dim=1)[:, :L].contiguous()
        return y.to(u.dtype) + self.D * u


class LinOSS(LinOSSNet):
    layer_cls = LinOSSLayer

    def set_path(self, path):
        """Select the convolution path ("auto", "depthwise", "freq") of every layer."""
        if path not in PATHS:
            raise ValueError(f"Unknown path {path!r}, expected one of {PATHS}")
        for block in self.blocks:
            block.ssm.path = path
        return self
