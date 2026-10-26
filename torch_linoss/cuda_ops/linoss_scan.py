"""
`linoss::scan`: the LinOSS recurrence as a custom PyTorch op backed by Triton kernels.

The op is registered with `torch.library.triton_op`, so `torch.compile`, `torch.export` and AOTInductor can
trace it without graph breaks, and the compiled Triton kernels are packaged into exported `.pt2` files.
"""

from typing import Optional

import torch
import triton
from torch.library import triton_op, wrap_triton

from torch_linoss.cuda_ops.linoss_scan_triton import (
    _linoss_scan_bwd_kernel,
    _linoss_scan_fwd_kernel,
)

BLOCK_L = 64


def _block_d(batch, D):
    # Each program walks the whole sequence, so parallelism is batch * D / BLOCK_D programs. Keep enough
    # programs to fill the GPU while giving each one a reasonably wide tile. A symbolic batch (dynamic-shape
    # export) gets a fixed width, so the exported program does not specialize on the batch size.
    if not isinstance(batch, int) or not isinstance(D, int):
        return 8
    for block_d in (32, 16, 8, 4):
        if batch * triton.cdiv(D, block_d) >= 256:
            return block_d
    return 4


def _num_chunks(L):
    return triton.cdiv(L, BLOCK_L)


def _pack_mf(M, f):
    return torch.cat([M.reshape(-1, 4), f], dim=-1).float().contiguous()


@triton_op("linoss::scan_fwd", mutates_args={})
def scan_fwd(
    x: torch.Tensor, mf: torch.Tensor, h0: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: (B, D, L), mf: (D, 6) fp32, h0: (B, D, 2) fp32 or None.

    Returns y (B, D, L) in x.dtype, chunk-start states (B, NC, D, 2), final state (B, D, 2).
    """
    batch, D, L = x.shape
    nc = _num_chunks(L)
    y = torch.empty_like(x)
    hs = x.new_empty((batch, nc, D, 2), dtype=torch.float32)
    ht = x.new_empty((batch, D, 2), dtype=torch.float32)
    block_d = _block_d(batch, D)
    grid = (batch, triton.cdiv(D, block_d))
    wrap_triton(_linoss_scan_fwd_kernel)[grid](
        x, y, mf, h0 if h0 is not None else ht, hs, ht,
        L, D, nc,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        HAS_H0=h0 is not None,
        BLOCK_L=BLOCK_L,
        BLOCK_D=block_d,
    )
    return y, hs, ht


@triton_op("linoss::scan_bwd", mutates_args={})
def scan_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    mf: torch.Tensor,
    hs: torch.Tensor,
    dht: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns dx (B, D, L), per-batch partial gradients of mf (B, D, 6), and d h0 (B, D, 2)."""
    batch, D, L = x.shape
    nc = hs.shape[1]
    dx = torch.empty_like(x)
    dmf = x.new_empty((batch, D, 6), dtype=torch.float32)
    dh0 = x.new_empty((batch, D, 2), dtype=torch.float32)
    block_d = _block_d(batch, D)
    grid = (batch, triton.cdiv(D, block_d))
    wrap_triton(_linoss_scan_bwd_kernel)[grid](
        x, dy, mf, hs, dht if dht is not None else dh0, dx, dmf, dh0,
        L, D, nc,
        x.stride(0), x.stride(1), x.stride(2),
        dy.stride(0), dy.stride(1), dy.stride(2),
        dx.stride(0), dx.stride(1), dx.stride(2),
        HAS_DHT=dht is not None,
        BLOCK_L=BLOCK_L,
        BLOCK_D=block_d,
    )
    return dx, dmf, dh0


def _setup_context(ctx, inputs, output):
    x, mf, h0 = inputs
    _, hs, _ = output
    ctx.save_for_backward(x, mf, hs)
    ctx.has_h0 = h0 is not None
    ctx.mark_non_differentiable(hs)
    ctx.set_materialize_grads(False)


def _backward(ctx, dy, dhs, dht):
    x, mf, hs = ctx.saved_tensors
    if dy is None:
        dy = torch.zeros_like(x)
    if dht is not None:
        dht = dht.float().contiguous()
    dx, dmf, dh0 = scan_bwd(x, dy, mf, hs, dht)
    return dx, dmf.sum(0), dh0 if ctx.has_h0 else None


scan_fwd.register_autograd(_backward, setup_context=_setup_context)


def linoss_scan(x, M, f, h0=None, return_final_state=False):
    """Run h_t = M h_{t-1} + f x_t, y_t = h_t[1] for every (batch, channel).

    Args:
        x: (batch, D, L), any float dtype and strides (e.g. a transposed view of (batch, L, D)).
        M: (D, 2, 2) transition matrices.
        f: (D, 2) input vectors.
        h0: optional (batch, D, 2) initial state.
        return_final_state: also return the (batch, D, 2) state after the last step.
    Returns:
        y (batch, D, L) in x.dtype, and the final state if requested.
    """
    if h0 is not None:
        h0 = h0.float().contiguous()
    y, _, ht = scan_fwd(x, _pack_mf(M, f), h0)
    if return_final_state:
        return y, ht
    return y
