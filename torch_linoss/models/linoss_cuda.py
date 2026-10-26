"""
LinOSS model built on the custom Triton kernels (`torch_linoss.cuda_ops`). This is the model that is
exported for deployment (`torch_linoss/export.py`). It has the same parameters as `LinOSS`, so weights move
between the two with `load_state_dict` or `LinOSSCuda.from_reference`.
"""

import torch

from torch_linoss.cuda_ops import linoss_scan
from torch_linoss.models.common import LinOSSLayerBase, LinOSSNet


class LinOSSCudaLayer(LinOSSLayerBase):
    """LinOSS layer that runs the recurrence with the `linoss::scan` kernel (no length-L kernel, no FFT).

    After `freeze()`, the discretized recurrence and the real projection matrices are cached as buffers,
    so inference does not recompute them. `unfreeze()` (or `train()` on the model) drops the cache.
    """

    def __init__(self, ssm_size, H, discretization, A_eps=0.0):
        super().__init__(ssm_size, H, discretization, A_eps)
        self.frozen = False
        self.register_buffer("_M", None, persistent=False)
        self.register_buffer("_f", None, persistent=False)
        self.register_buffer("_w_in", None, persistent=False)
        self.register_buffer("_w_out", None, persistent=False)

    def channel_recurrence(self):
        """(M (2P, 2, 2), f (2P, 2)): the per-state recurrence, shared by the real and imaginary channels."""
        if self.frozen:
            return self._M, self._f
        M, f = self.recurrence()
        return M.repeat(2, 1, 1), f.repeat(2, 1)

    def in_proj_weight(self):
        return self._w_in if self.frozen else super().in_proj_weight()

    def out_proj_weight(self):
        return self._w_out if self.frozen else super().out_proj_weight()

    def mix(self, x):
        M, f = self.channel_recurrence()
        return linoss_scan(x, M, f)

    @torch.no_grad()
    def freeze(self):
        self.frozen = False
        M, f = self.channel_recurrence()
        self._M, self._f = M.float().contiguous(), f.float().contiguous()
        self._w_in = super().in_proj_weight().contiguous()
        self._w_out = super().out_proj_weight().contiguous()
        self.frozen = True

    def unfreeze(self):
        self.frozen = False
        self._M = self._f = self._w_in = self._w_out = None


class LinOSSCuda(LinOSSNet):
    layer_cls = LinOSSCudaLayer

    @classmethod
    def from_reference(cls, model):
        """Build a LinOSSCuda with the configuration and weights of a `LinOSS` (or any LinOSSNet)."""
        new = cls(**model.config)
        new.load_state_dict(model.state_dict())
        p = next(model.parameters())
        return new.to(device=p.device, dtype=p.dtype)

    def freeze(self):
        """Switch to eval mode and cache the per-layer recurrence for inference/export."""
        self.eval()
        for block in self.blocks:
            block.ssm.freeze()
        return self

    def train(self, mode=True):
        if mode:
            for block in self.blocks:
                block.ssm.unfreeze()
        return super().train(mode)
