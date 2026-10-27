"""
End-effector trajectories for 100 Hz → 1 kHz inpainting (pick-up-mug dataset).

Streams: the right wrist and the 5 right fingertips. Features per stream: position (m) and orientation as a
continuous 6D rotation (first two columns of R) → 6 × 9 = 54 channels at 1 kHz.

Keyframes are steps 0, 10, 20, … (100 Hz). Every episode is cropped to end on a keyframe (dropping at most 9
final samples), so every missing step lies between two keyframes.

Model input per 1 kHz step (3·54 + 2 = 164 channels):
- the linear interpolation between the surrounding keyframes (54),
- the keyframe first difference over the current gap (54) and the keyframe second difference (curvature) at
  the gap's start keyframe (54), each scaled to O(1) by its training spread,
- a keyframe mask (1) and the position within the gap k/10 (1).
The model predicts a correction to the interpolation. The correction is ~1e-3 of a channel's spread, so it
is not recoverable from the interpolated positions at float precision; the difference features carry that
information at O(1). Everything at a step in gap i comes from keyframes i-1, i, i+1, so a causal model sees
at most 10 steps (10 ms) ahead.
"""

import glob
import json
import os

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA_DIR = "/home/azureuser/huy/dataset/pi2_limit/pickup_mug_augmented"
STREAMS = [("wrist", "RIGHT_ARM_END_EFFECTOR/end_effector")] + \
          [(f"finger_{i}", f"RIGHT_HAND_END_EFFECTORS/finger_{i}_ee") for i in range(1, 6)]
STRIDE = 10  # 1 kHz → 100 Hz keyframes
N_STREAMS = len(STREAMS)
C = 9 * N_STREAMS  # per stream: pos 3 + rot6d 6
N_IN = 3 * C + 2
MASK_IDX = 3 * C  # keyframe-mask input channel
POS_IDX = np.array([9 * s + i for s in range(N_STREAMS) for i in range(3)])
ROT_IDX = np.array([9 * s + 3 + i for s in range(N_STREAMS) for i in range(6)])


# ---------------- rotations
def quat_to_rotmat(q):
    """q (..., 4) as w, x, y, z → R (..., 3, 3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rotmat_to_rot6d(R):
    """First two columns of R, flattened as [c0, c1] → (..., 6)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_rotmat(r6):
    """Gram–Schmidt on numpy (..., 6) → (..., 3, 3)."""
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(1e-9)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(1e-9)
    return np.stack([b1, b2, np.cross(b1, b2)], -1)


def rotation_angle_deg(R1, R2):
    """Geodesic angle between rotation matrices (..., 3, 3), in degrees."""
    tr = np.einsum("...ij,...ij->...", R1, R2)  # trace(R1^T R2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0)))


# ---------------- loading and caching
def episodes():
    return sorted(os.path.basename(p.rstrip("/")) for p in glob.glob(os.path.join(DATA_DIR, "*/")))


def load_episode(name):
    """(features (N, 54) float32, utime (N,) int64 µs) for one episode, cached in out/cache/."""
    cache = os.path.join(OUT, "cache", f"ee_{name}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["x"], z["utime"]
    feats, utime = [], None
    with h5py.File(os.path.join(DATA_DIR, name, "data.h5"), "r") as f:
        for _, path in STREAMS:
            a = f[f"data/observation/end_effector/{path}/pose_timeseries"][()]
            if utime is None:
                utime = a[:, 0].astype(np.int64)
            n = min(len(utime), len(a))
            utime, a = utime[:n], a[:n]
            feats = [x[:n] for x in feats]
            R = quat_to_rotmat(a[:, 7:11])
            feats.append(np.concatenate([a[:, 1:4], rotmat_to_rot6d(R)], axis=1))
    x = np.concatenate(feats, axis=1).astype(np.float32)
    keep = np.concatenate([[True], np.diff(utime) > 0])  # each episode repeats its first sample
    x, utime = x[keep], utime[keep]
    n = STRIDE * ((len(x) - 1) // STRIDE) + 1  # end on a keyframe
    x, utime = x[:n], utime[:n]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, x=x, utime=utime)
    return x, utime


def splits(names, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(names))
    n_tr, n_va = int(round(0.8 * len(names))), int(round(0.1 * len(names)))
    out = {"train": sorted(names[i] for i in perm[:n_tr]),
           "val": sorted(names[i] for i in perm[n_tr:n_tr + n_va]),
           "test": sorted(names[i] for i in perm[n_tr + n_va:])}
    return out


# ---------------- inputs
def linear_interp(y):
    """Linear interpolation between keyframes (every STRIDE steps) of y (L, C) → (L, C). L = STRIDE*k + 1."""
    key = y[::STRIDE]
    k = np.arange(len(y)) % STRIDE
    i = np.arange(len(y)) // STRIDE
    lo = key[np.minimum(i, len(key) - 1)]
    hi = key[np.minimum(i + 1, len(key) - 1)]
    w = (k / STRIDE)[:, None]
    return (1 - w) * lo + w * hi


def keyframe_diffs(y):
    """Per 1 kHz step: first difference of the current gap and second difference at the gap's start keyframe.

    In gap i (steps 10i … 10i+9) these use keyframes i-1, i, i+1 only; in the first gap the curvature is 0.
    """
    key = y[::STRIDE]
    K = len(key)
    d1 = np.diff(key, axis=0)                              # (K-1,): gap i
    d2 = key[2:] - 2 * key[1:-1] + key[:-2]                # (K-2,): curvature at keyframe i+1
    gap = np.minimum(np.arange(len(y)) // STRIDE, K - 2)
    f1 = d1[gap]
    f2 = d2[np.clip(gap - 1, 0, K - 3)]
    f2[gap == 0] = 0.0
    return f1, f2


def model_inputs(y_norm, d1_std=1.0, d2_std=1.0):
    """y_norm (L, C) → (x (L, N_IN), lin (L, C), keyframe mask (L,))."""
    L = len(y_norm)
    lin = linear_interp(y_norm)
    f1, f2 = keyframe_diffs(y_norm)
    k = np.arange(L) % STRIDE
    mask = (k == 0).astype(np.float32)
    x = np.concatenate([lin, f1 / d1_std, f2 / d2_std, mask[:, None], (k / STRIDE)[:, None]], axis=1)
    return x.astype(np.float32), lin.astype(np.float32), mask


def load_conditions(name):
    """Stage-2 hook: SigLIP camera embeddings and 6D tactile signals on the 1 kHz grid. Empty for now."""
    return None


class InpaintData:
    def __init__(self, device="cuda", seed=0):
        self.device = device
        names = episodes()
        self.split = splits(names, seed)
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "splits.json"), "w") as f:
            json.dump(self.split, f, indent=1)
        self.raw = {}
        bad_dt = []
        for n in names:
            x, ut = load_episode(n)
            dt = np.diff(ut)
            if np.any(np.abs(dt - 1000) > 1):  # ±1 µs is timestamp rounding
                bad_dt.append((n, int((np.abs(dt - 1000) > 1).sum()), int(dt.max())))
            self.raw[n] = x
        self.bad_dt = bad_dt
        tr = np.concatenate([self.raw[n] for n in self.split["train"]])
        self.mean = tr.mean(0)
        self.std = tr.std(0) + 1e-6
        # The correction to linear interpolation is ~1e-3 of a channel's spread, so it gets its own scale:
        # targets are r = (y - lin) / res_std, which is O(1) (linear interpolation scores r-RMSE ≈ 1).
        res = []
        for n in self.split["train"]:
            y = (self.raw[n] - self.mean) / self.std
            res.append((y - linear_interp(y))[np.arange(len(y)) % STRIDE != 0])
        self.res_std = np.concatenate(res).std(0) + 1e-9
        f = [keyframe_diffs((self.raw[n] - self.mean) / self.std) for n in self.split["train"]]
        self.d1_std = np.concatenate([a for a, _ in f]).std(0) + 1e-9
        self.d2_std = np.concatenate([b for _, b in f]).std(0) + 1e-9
        self.gpu = {}
        for n, x in self.raw.items():
            y = (x - self.mean) / self.std
            xin, lin, mask = model_inputs(y, self.d1_std, self.d2_std)
            r = ((y - lin) / self.res_std).astype(np.float32)
            self.gpu[n] = {k: torch.as_tensor(v, device=device) for k, v in
                           (("x", xin), ("r", r), ("lin", lin), ("mask", mask))}
        self.res_std_t = torch.as_tensor(self.res_std, device=device, dtype=torch.float32)

    def denorm(self, y):
        return y * self.std + self.mean

    def prediction(self, lin, mu):
        """Normalized prediction from the interpolation and a correction in residual units."""
        return lin + mu * self.res_std_t

    def sigma(self, logvar):
        """Predicted standard deviation in original units (metres / rot6d units)."""
        return self.std * self.res_std * np.exp(0.5 * logvar)

    def window(self, name, start, length):
        g = self.gpu[name]
        sl = slice(start, start + length)
        return g["x"][sl], g["r"][sl], g["lin"][sl], g["mask"][sl]

    def batch(self, size, length, rng):
        """Random keyframe-aligned windows of `length` steps (length = STRIDE*k + 1) from training episodes."""
        names = self.split["train"]
        usable = np.array([max(len(self.raw[n]) - length, -1) for n in names])
        p = np.clip(usable, 0, None) + 1e-9
        p = p / p.sum()
        out = [[], [], [], []]
        for _ in range(size):
            n = names[rng.choice(len(names), p=p)]
            n_starts = (len(self.raw[n]) - length) // STRIDE + 1
            start = STRIDE * int(rng.integers(0, n_starts))
            for lst, v in zip(out, self.window(n, start, length)):
                lst.append(v)
        return tuple(torch.stack(v) for v in out)

    def full(self, name):
        return self.window(name, 0, len(self.raw[name]))


if __name__ == "__main__":
    d = InpaintData()
    print({k: len(v) for k, v in d.split.items()}, "episodes with steps off 1 ms by >1 µs:", d.bad_dt)
    L = [len(x) for x in d.raw.values()]
    print("lengths", min(L), "-", max(L), "total steps", sum(L))
    print("channel std (pos mm):", (1000 * d.std[POS_IDX]).round(1))
