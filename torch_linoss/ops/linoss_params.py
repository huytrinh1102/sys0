"""
Discretization of the LinOSS oscillator ODE.

For every state dimension p the second-order ODE y'' = -A y + B u is written as a first-order system in
x = [z; y] (velocity, position) and discretized with step size dt. Both schemes give a time-invariant
recurrence

    x_n = M x_{n-1} + f * (Bu)_n

with a 2x2 transition matrix M and an input vector f (per state dimension):

- IM (implicit):   S = 1 / (1 + dt^2 A)
                   M = [[S, -dt A S], [dt S, S]],      f = [dt S, dt^2 S]
- IMEX (symplectic implicit-explicit):
                   M = [[1, -dt A], [dt, 1 - dt^2 A]], f = [dt, dt^2]

These match `apply_linoss_im` / `apply_linoss_imex` in `models/LinOSS.py` (JAX).
"""

import torch
import torch.nn.functional as F

DISCRETIZATIONS = ("IM", "IMEX")


def check_discretization(discretization):
    if discretization not in DISCRETIZATIONS:
        raise ValueError(
            f"Unknown LinOSS discretization {discretization!r}, expected one of {DISCRETIZATIONS}"
        )


def discretize(A_diag, steps, discretization, A_eps=0.0):
    """Map raw parameters to the recurrence (M, f).

    Args:
        A_diag: raw diagonal state matrix (P,), passed through relu.
        steps: raw step sizes (P,), passed through sigmoid.
        discretization: "IM" or "IMEX".
        A_eps: floor added after the relu, A = relu(A_diag) + A_eps. The original model uses 0; a small
            positive value keeps oscillators from collapsing to a double integrator (A = 0).
    Returns:
        M: (P, 2, 2) transition matrices.
        f: (P, 2) input vectors.
    """
    check_discretization(discretization)
    A = F.relu(A_diag) + A_eps
    dt = torch.sigmoid(steps)
    if discretization == "IM":
        S = 1.0 / (1.0 + dt**2 * A)
        M = torch.stack(
            [torch.stack([S, -dt * A * S], -1), torch.stack([dt * S, S], -1)], -2
        )
        f = torch.stack([dt * S, dt**2 * S], -1)
    else:
        one = torch.ones_like(dt)
        M = torch.stack(
            [torch.stack([one, -dt * A], -1), torch.stack([dt, 1.0 - dt**2 * A], -1)],
            -2,
        )
        f = torch.stack([dt, dt**2], -1)
    return M, f
