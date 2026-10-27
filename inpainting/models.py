"""
LinOSS inpainting network: per-step MLP encoder → LinOSS blocks → probabilistic head.

The network predicts a correction to the linear interpolation of the keyframes:
    mean = lin + mu,  variance = exp(logvar)    (normalized units; mu = 0 at keyframes)

direction="causal": forward LinOSS layers. With the linear-interpolation input this means a 10 ms lookahead.
direction="bi":     every layer runs forward and backward (two independent oscillator banks) and sums them.

All LinOSS layers use the timescale initialization (oscillator frequencies log-uniform over a band in Hz).
`cond` (B, L, cond_dim) is concatenated to the encoder input: the stage-2 hook for SigLIP / tactile signals.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_linoss.models.common import LinOSSBlock  # noqa: E402
from torch_linoss.models.linoss_cuda import LinOSSCudaLayer  # noqa: E402

from data import C, MASK_IDX, N_IN  # noqa: E402

LOGVAR_MIN, LOGVAR_MAX = -10.0, 5.0


class BiLinOSSLayer(nn.Module):
    """Forward and backward LinOSS layers over the same sequence, summed."""

    def __init__(self, ssm_size, H, discretization, A_eps=0.0, init=None):
        super().__init__()
        self.fwd = LinOSSCudaLayer(ssm_size, H, discretization, A_eps=A_eps, init=init)
        self.bwd = LinOSSCudaLayer(ssm_size, H, discretization, A_eps=A_eps, init=init)

    def forward(self, u):
        return self.fwd(u) + self.bwd(u.flip(1)).flip(1)


class InpaintNet(nn.Module):
    def __init__(self, direction="bi", head="nll", H=128, depth=4, P=128, cond_dim=0,
                 f_min=0.5, f_max=100.0, fs=1000.0, drop_rate=0.05, A_eps=1e-3):
        super().__init__()
        assert direction in ("bi", "causal") and head in ("nll", "mse")
        self.config = dict(direction=direction, head=head, H=H, depth=depth, P=P, cond_dim=cond_dim,
                           f_min=f_min, f_max=f_max, fs=fs, drop_rate=drop_rate, A_eps=A_eps)
        self.head_type = head
        init = dict(fs=fs, f_min=f_min, f_max=f_max)
        layer_cls = BiLinOSSLayer if direction == "bi" else LinOSSCudaLayer
        self.encoder = nn.Sequential(nn.Linear(N_IN + cond_dim, 256), nn.GELU(), nn.Linear(256, H))
        self.blocks = nn.ModuleList(
            [LinOSSBlock(layer_cls, P, H, "IM", drop_rate, "batch", A_eps, init) for _ in range(depth)]
        )
        self.head = nn.Linear(H, C * (2 if head == "nll" else 1))

    def forward(self, x, cond=None):
        """x (B, L, N_IN) → (mu (B, L, 54), logvar (B, L, 54) or None). mu is the correction to x[..., :54]."""
        if cond is not None:
            x = torch.cat([x, cond], dim=-1)
        h = self.encoder(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        missing = (x[..., MASK_IDX:MASK_IDX + 1] < 0.5).to(out.dtype)
        mu = out[..., :C] * missing
        logvar = out[..., C:].clamp(LOGVAR_MIN, LOGVAR_MAX) if self.head_type == "nll" else None
        return mu, logvar


def gaussian_nll(target, mu, logvar):
    """Elementwise 0.5 * ((target - mu)^2 * exp(-logvar) + logvar)."""
    return 0.5 * ((target - mu) ** 2 * torch.exp(-logvar) + logvar)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
