"""
Naive time-stepping implementations. They are slow and only used as a test oracle and for
step-by-step (streaming) inference.
"""

import torch


def linoss_step(state, x_t, M, f):
    """One step of x_n = M x_{n-1} + f * u_n for every channel.

    Args:
        state: (batch, D, 2) previous state [z; y].
        x_t: (batch, D) input at this step.
        M: (D, 2, 2), f: (D, 2).
    Returns:
        (y_t (batch, D), new state (batch, D, 2)).
    """
    state = torch.einsum("dij,bdj->bdi", M, state) + f * x_t.unsqueeze(-1)
    return state[..., 1], state


def linoss_recurrence_reference(x, M, f, state=None):
    """Run the recurrence over a whole sequence with a Python loop.

    Args:
        x: (batch, D, L).
        M: (D, 2, 2), f: (D, 2).
        state: optional initial state (batch, D, 2); zeros if None.
    Returns:
        (y (batch, D, L), final state (batch, D, 2)).
    """
    batch, D, L = x.shape
    if state is None:
        state = x.new_zeros(batch, D, 2)
    ys = []
    for t in range(L):
        y_t, state = linoss_step(state, x[..., t], M, f)
        ys.append(y_t)
    return torch.stack(ys, -1), state


def linoss_layer_reference(u, A_diag, steps, B, C, D, discretization):
    """Full LinOSS layer written like the JAX code: complex B and C, a time loop over the 2-vector states.

    Args:
        u: (batch, L, H) real input.
        A_diag, steps: (P,) raw parameters.
        B: (P, H, 2), C: (H, P, 2) real/imag parts.
        D: (H,).
    Returns:
        (batch, L, H).
    """
    from torch_linoss.ops.linoss_params import discretize

    M, f = discretize(A_diag, steps, discretization)
    Bc = torch.complex(B[..., 0], B[..., 1])
    Cc = torch.complex(C[..., 0], C[..., 1])
    Bu = torch.einsum("ph,blh->bpl", Bc, u.to(Bc.dtype))  # (batch, P, L), complex
    Mc, fc = M.to(Bc.dtype), f.to(Bc.dtype)
    ys, _ = linoss_recurrence_reference(Bu, Mc, fc)
    out = torch.einsum("hp,bpl->blh", Cc, ys).real
    return out + D * u
