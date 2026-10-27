"""
1 kHz comparison: what each method puts between the 100 Hz keyframes, against ground truth.

    python plot_1khz.py --seed 2345 --ms 60 --where fast   # or --where hardest

Writes out/figures/1khz_<stream>_<where>.png. Columns: bidirectional methods | causal methods. Rows:
1. position at 1 kHz (every ms marked) with the 100 Hz keyframes
2. 1 kHz detail = signal − linear interpolation of the keyframes (µm): the part the methods must invent
3. error = prediction − ground truth (µm)
NLL head solid, MSE head dashed; colour follows the direction as in plot.py.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import BASELINES
from data import STRIDE, linear_interp
from plot import FIG, GRID, INK, INK_2, METHODS, preds, style

COLS = {
    "Bidirectional (sees the whole trajectory)": [
        ("linear", None, "Linear interpolation", METHODS["linear"][1], "-"),
        ("cubic", None, "Cubic spline", METHODS["cubic"][1], "-"),
        ("bi", "nll", "LinOSS bi, β-NLL head", METHODS["bi"][1], "-"),
        ("bi", "mse", "LinOSS bi, MSE head", METHODS["bi"][1], "--")],
    "Causal (10 ms lookahead)": [
        ("linear", None, "Linear interpolation", METHODS["linear"][1], "-"),
        ("causal", "nll", "LinOSS causal, β-NLL head", METHODS["causal"][1], "-"),
        ("causal", "mse", "LinOSS causal, MSE head", METHODS["causal"][1], "--")],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2345)
    p.add_argument("--ms", type=int, default=60, help="window length in ms (multiple of 10)")
    p.add_argument("--where", choices=("fast", "hardest"), default="fast",
                   help="fast: fastest motion away from the episode start; hardest: most 1 kHz detail anywhere")
    args = p.parse_args()
    os.makedirs(FIG, exist_ok=True)
    P = {(d, h): preds(d, args.seed, h) for d in ("bi", "causal") for h in ("nll", "mse")}
    ep = sorted(P[("bi", "nll")])[0]
    true = P[("bi", "nll")][ep]["true"].astype(np.float64)
    pred = {("linear", None): BASELINES["linear"](true), ("cubic", None): BASELINES["cubic"](true)}
    pred.update({k: v[ep]["pred"].astype(np.float64) for k, v in P.items()})
    lin = linear_interp(true)
    n = args.ms + 1

    for s_idx, stream in ((0, "wrist"), (1, "finger_1")):
        pos = true[:, 9 * s_idx:9 * s_idx + 3]
        if args.where == "hardest":  # most 1 kHz detail (deviation from linear interpolation), e.g. the start-up transient
            score = np.linalg.norm(pos - lin[:, 9 * s_idx:9 * s_idx + 3], axis=1) ** 2
        else:  # fastest motion, skipping the first 200 ms of start-up
            score = np.r_[np.linalg.norm(np.diff(pos, axis=0), axis=1), 0.0]
            score[:200] = 0
        c0 = int(np.argmax(np.convolve(score, np.ones(n), "valid")))
        c0 -= c0 % STRIDE
        sl = slice(c0, c0 + n)
        ax_i = int(np.argmax(np.abs(true[sl, 9 * s_idx:9 * s_idx + 3] - lin[sl, 9 * s_idx:9 * s_idx + 3]).max(0)))
        c = 9 * s_idx + ax_i
        t = np.arange(n)
        key = t % STRIDE == 0
        miss = ~key
        gt = true[sl, c]
        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, sharey="row")
        for j, (title, methods) in enumerate(COLS.items()):
            a0, a1, a2 = axes[:, j]
            a0.plot(t, 1000 * (gt - gt[0]), color=INK, lw=1.4, marker="o", ms=2.6, label="Ground truth (1 kHz)")
            a1.plot(t, 1e6 * (gt - lin[sl, c]), color=INK, lw=2.2, marker="o", ms=3, label="Ground truth", zorder=5)
            for m, h, label, color, ls in methods:
                y = pred[(m, h)][sl, c]
                e = 1e6 * (y - gt)
                rms = np.sqrt(np.mean(e[miss] ** 2))
                if m != "linear":
                    a0.plot(t, 1000 * (y - gt[0]), color=color, lw=1.2, ls=ls, label=label)
                a1.plot(t, 1e6 * (y - lin[sl, c]), color=color, lw=1.6, ls=ls, marker=".", ms=3, label=label)
                a2.plot(t, e, color=color, lw=1.6, ls=ls, marker=".", ms=3, label=f"{label} ({rms:.1f} µm RMS)")
            for a in (a0, a1, a2):
                for k in t[key]:
                    a.axvline(k, color=GRID, lw=0.8, zorder=0)
            a0.scatter(t[key], 1000 * (gt[key] - gt[0]), s=60, facecolor="white", edgecolor=INK, lw=1.4, zorder=6,
                       label="100 Hz keyframe (model input)")
            a1.scatter(t[key], np.zeros(key.sum()), s=40, facecolor="white", edgecolor=INK, lw=1.2, zorder=6)
            a2.scatter(t[key], np.zeros(key.sum()), s=40, facecolor="white", edgecolor=INK, lw=1.2, zorder=6)
            a1.axhline(0, color=INK_2, lw=0.6); a2.axhline(0, color=INK_2, lw=0.6)
            a0.set_title(title, loc="left", fontsize=11.5, color=INK)
            for a in (a0, a1, a2):
                style(a)
            a0.legend(frameon=False, fontsize=8, labelcolor=INK, loc="best")
            a2.legend(frameon=False, fontsize=8, labelcolor=INK, loc="best")
            a2.set_xlabel("time (ms at 1 kHz; grey lines = 100 Hz keyframes)", color=INK_2)
        axes[0, 0].set_ylabel(f"{stream} {'xyz'[ax_i]} position (mm, rel.)", color=INK_2)
        axes[1, 0].set_ylabel("1 kHz detail: signal − linear interp (µm)", color=INK_2)
        axes[2, 0].set_ylabel("error: prediction − truth (µm)", color=INK_2)
        fig.suptitle(f"{stream} {'xyz'[ax_i]} at 1 kHz from 100 Hz keyframes, {args.ms} ms of test episode {ep.split('_')[0]} "
                     f"(steps {c0}–{c0 + args.ms}, seed {args.seed})", x=0.01, ha="left", fontsize=12.5, color=INK)
        fig.tight_layout()
        out = os.path.join(FIG, f"1khz_{stream}_{args.where}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
