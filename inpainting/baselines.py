"""
Classical 100 Hz → 1 kHz upsampling baselines, computed per channel from the keyframes only.

- zoh:    zero-order hold of the last keyframe (causal, no lookahead)
- linear: linear interpolation between neighbouring keyframes (needs the next keyframe: 10 ms lookahead)
- cubic:  natural cubic spline through all keyframes (offline: uses more than one future keyframe)
"""

import numpy as np
from scipy.interpolate import CubicSpline

from data import STRIDE, linear_interp


def zoh(y):
    key = y[::STRIDE]
    return key[np.minimum(np.arange(len(y)) // STRIDE, len(key) - 1)]


def linear(y):
    return linear_interp(y)


def cubic(y):
    t_key = np.arange(0, len(y), STRIDE)
    return CubicSpline(t_key, y[::STRIDE], axis=0, bc_type="natural")(np.arange(len(y)))


BASELINES = {"zoh": zoh, "linear": linear, "cubic": cubic}
