"""
Convolution kernel of the LinOSS recurrence.

Unrolling x_n = M x_{n-1} + f u_n from x_{-1} = 0 gives the position read-out

    y_n = sum_k kappa[k] u_{n-k},   kappa[k] = e_2^T M^k f,

so each state dimension is a causal convolution with a real scalar kernel of length L.
"""

import torch


def linoss_kernel(M, f, L):
    """Compute kappa[p, k] = (M_p^k f_p)[1] for k < L.

    Uses repeated doubling: V[:, :, n:2n] = M^n V[:, :, :n], then M^{2n} = (M^n)^2.
    This takes ceil(log2 L) steps of batched 2x2 products, is differentiable, and does not need an
    eigendecomposition (M is a non-diagonalizable Jordan block when relu(A) == 0).

    Args:
        M: (P, 2, 2) transition matrices.
        f: (P, 2) input vectors.
        L: kernel length.
    Returns:
        (P, L) kernel.
    """
    V = f.unsqueeze(-1)  # (P, 2, 1): M^0 f
    Mp = M
    while V.shape[-1] < L:
        V = torch.cat([V, Mp @ V], dim=-1)
        if V.shape[-1] < L:
            Mp = Mp @ Mp
    return V[:, 1, :L]
