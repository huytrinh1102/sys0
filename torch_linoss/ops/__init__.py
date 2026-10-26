"""Pure-PyTorch LinOSS operations shared by every model and backend."""

from torch_linoss.ops.causal_conv import (
    causal_conv1d_fft,
    fft_causal_conv_lag,
    next_fast_len,
)
from torch_linoss.ops.kernel import linoss_kernel
from torch_linoss.ops.linoss_params import (
    DISCRETIZATIONS,
    check_discretization,
    discretize,
)
from torch_linoss.ops.reference import (
    linoss_layer_reference,
    linoss_recurrence_reference,
    linoss_step,
)

__all__ = [
    "DISCRETIZATIONS",
    "causal_conv1d_fft",
    "check_discretization",
    "discretize",
    "fft_causal_conv_lag",
    "linoss_kernel",
    "linoss_layer_reference",
    "linoss_recurrence_reference",
    "linoss_step",
    "next_fast_len",
]
