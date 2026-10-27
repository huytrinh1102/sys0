import json
import os

import numpy as np
import pytest
import torch

from baselines import BASELINES
from data import MASK_IDX, N_IN, OUT, STRIDE, episodes, keyframe_diffs, linear_interp, model_inputs, \
    quat_to_rotmat, rot6d_to_rotmat, rotmat_to_rot6d, splits
from models import LOGVAR_MAX, LOGVAR_MIN, InpaintNet, gaussian_nll

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_rot6d_roundtrip():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(100, 4))
    R = quat_to_rotmat(q)
    assert np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-10)
    assert np.allclose(rot6d_to_rotmat(rotmat_to_rot6d(R)), R, atol=1e-10)


def test_baselines_keep_keyframes_and_linear_ramp():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(STRIDE * 12 + 1, 5))
    for fn in BASELINES.values():
        assert np.allclose(fn(y)[::STRIDE], y[::STRIDE])
    ramp = np.linspace(0, 1, STRIDE * 6 + 1)[:, None] * np.array([[1.0, -2.0]])
    assert np.allclose(linear_interp(ramp), ramp)


def test_model_inputs():
    y = np.arange(STRIDE * 3 + 1, dtype=np.float32)[:, None].repeat(54, 1)
    x, lin, mask = model_inputs(y)
    assert x.shape == (len(y), N_IN) and mask[::STRIDE].all() and mask.sum() == 4
    assert np.allclose(x[:, MASK_IDX], mask)
    assert np.allclose(x[:, N_IN - 1], (np.arange(len(y)) % STRIDE) / STRIDE)


def test_keyframe_diffs_use_only_neighbouring_keyframes():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(STRIDE * 8 + 1, 3))
    f1, f2 = keyframe_diffs(y)
    y2 = y.copy(); y2[STRIDE * 5:] += 7.0          # change keyframe 5 and later
    g1, g2 = keyframe_diffs(y2)
    first = STRIDE * 4                              # gap 4 is the first that may see keyframe 5
    assert np.allclose(f1[:first], g1[:first]) and np.allclose(f2[:first], g2[:first])
    assert not np.allclose(f1[first:first + STRIDE], g1[first:first + STRIDE])


def test_splits_disjoint_and_complete():
    names = episodes()
    s = splits(names)
    allv = sum(s.values(), [])
    assert len(allv) == len(set(allv)) == len(names) == 268


def test_gaussian_nll_closed_form():
    t, mu, lv = torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([float(np.log(0.25))])
    expected = 0.5 * ((0.5 ** 2) / 0.25 + np.log(0.25))
    assert torch.allclose(gaussian_nll(t, mu, lv), torch.tensor([expected], dtype=torch.float32))


def inputs(L=801):
    y = torch.randn(L, 54).numpy()
    x, _, _ = model_inputs(y)
    return torch.as_tensor(x, device="cuda")[None]


@cuda
def test_keyframes_exact_and_logvar_clamped():
    m = InpaintNet("bi").cuda().eval()
    x = inputs()
    x[..., :MASK_IDX] *= 50  # extreme values, mask channel untouched
    with torch.no_grad():
        mu, lv = m(x)
    assert mu[:, ::STRIDE].abs().max() == 0
    assert lv.min() >= LOGVAR_MIN and lv.max() <= LOGVAR_MAX


@cuda
def test_causal_sees_at_most_one_gap_ahead():
    torch.manual_seed(0)
    m = InpaintNet("causal").cuda().eval()
    y = np.random.default_rng(1).normal(size=(801, 54))
    K = 400  # perturb the keyframe at step 400 (and everything after it)
    y2 = y.copy(); y2[K:] += 3.0
    x1, x2 = (torch.as_tensor(model_inputs(v)[0], device="cuda")[None] for v in (y, y2))
    with torch.no_grad():
        a, _ = m(x1); b, _ = m(x2)
    # the interpolation of the gap before K uses keyframe K, so outputs up to K - STRIDE must not change
    assert (a[:, :K - STRIDE + 1] - b[:, :K - STRIDE + 1]).abs().max() < 1e-5
    assert (a[:, K - STRIDE + 1:] - b[:, K - STRIDE + 1:]).abs().max() > 1e-3


@cuda
def test_bidirectional_sees_both_sides():
    torch.manual_seed(0)
    m = InpaintNet("bi").cuda().eval()
    x = inputs()
    x2 = x.clone(); x2[:, 600:, :MASK_IDX] += 3.0
    with torch.no_grad():
        a, _ = m(x); b, _ = m(x2)
    assert (a[:, 100:500] - b[:, 100:500]).abs().max() > 1e-3  # earlier outputs react to later inputs
