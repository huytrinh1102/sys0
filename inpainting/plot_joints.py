"""
Per-joint 1 kHz trajectory comparison: position x/y/z and all 9 rotation-matrix entries.

    python plot_joints.py --seed 2345 --ms 60

Writes out/figures/joints/<stream>.png, one per end effector. Every panel shows ground truth and every method over
the same window (fastest wrist motion of the first test episode, skipping the start-up transient), with a strip of
prediction − truth below. Rotation matrices come from the 6D outputs via Gram–Schmidt (R = [b1 b2 b1×b2]).
NLL head solid, MSE head dashed; colour follows the method as in plot.py.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from baselines import BASELINES
from data import N_STREAMS, STRIDE, rot6d_to_rotmat, rotation_angle_deg
from plot import FIG, GRID, INK, INK_2, METHODS, NAMES, preds, style

LINES = [  # (key, label, colour, linestyle)
    (("linear", None), "Linear interpolation", METHODS["linear"][1], "-"),
    (("cubic", None), "Cubic spline", METHODS["cubic"][1], "-"),
    (("causal", "nll"), "LinOSS causal, β-NLL", METHODS["causal"][1], "-"),
    (("causal", "mse"), "LinOSS causal, MSE", METHODS["causal"][1], "--"),
    (("bi", "nll"), "LinOSS bi, β-NLL", METHODS["bi"][1], "-"),
    (("bi", "mse"), "LinOSS bi, MSE", METHODS["bi"][1], "--"),
]


def channels(y, s):
    """(L, 54) → position (L, 3) in m and rotation matrix (L, 3, 3) of stream s."""
    return y[:, 9 * s:9 * s + 3], rot6d_to_rotmat(y[:, 9 * s + 3:9 * s + 9])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2345)
    p.add_argument("--ms", type=int, default=60)
    args = p.parse_args()
    out_dir = os.path.join(FIG, "joints")
    os.makedirs(out_dir, exist_ok=True)
    P = {(d, h): preds(d, args.seed, h) for d in ("bi", "causal") for h in ("nll", "mse")}
    ep = sorted(P[("bi", "nll")])[0]
    true = P[("bi", "nll")][ep]["true"].astype(np.float64)
    pred = {("linear", None): BASELINES["linear"](true), ("cubic", None): BASELINES["cubic"](true)}
    pred.update({k: v[ep]["pred"].astype(np.float64) for k, v in P.items()})

    n = args.ms + 1
    speed = np.r_[np.linalg.norm(np.diff(true[:, :3], axis=0), axis=1), 0.0]
    speed[:200] = 0
    c0 = int(np.argmax(np.convolve(speed, np.ones(n), "valid")))
    c0 -= c0 % STRIDE
    sl = slice(c0, c0 + n)
    t = np.arange(n)
    key = t % STRIDE == 0
    miss = ~key

    for s in range(N_STREAMS):
        gp, gR = channels(true[sl], s)
        M = {k: channels(v[sl], s) for k, v in pred.items()}
        fig = plt.figure(figsize=(16, 19))
        gs = GridSpec(12, 3, figure=fig, height_ratios=[3, 1.25, 0.42] * 4, hspace=0.1, wspace=0.18)
        panels = [(0, j, f"position {'xyz'[j]}", lambda pr, j=j: pr[0][:, j], gp[:, j], 1000, "mm (rel. to window start)", 1e6, "µm")
                  for j in range(3)]
        panels += [(1 + i, j, f"R[{i},{j}]", lambda pr, i=i, j=j: pr[1][:, i, j], gR[:, i, j], 1, "", 1e4, "×10⁻⁴")
                   for i in range(3) for j in range(3)]
        for row, col, name, get, g, scale, unit, escale, eunit in panels:
            a = fig.add_subplot(gs[3 * row, col])
            e_ax = fig.add_subplot(gs[3 * row + 1, col], sharex=a)
            off = g[0] if scale != 1 else 0.0
            for k in t[key]:
                a.axvline(k, color=GRID, lw=0.8, zorder=0); e_ax.axvline(k, color=GRID, lw=0.8, zorder=0)
            a.plot(t, scale * (g - off), color=INK, lw=2.0, marker="o", ms=2.4, label="Ground truth (1 kHz)", zorder=5)
            for k, label, color, ls in LINES:
                y = get(M[k])
                a.plot(t, scale * (y - off), color=color, lw=1.2, ls=ls, label=label)
                e_ax.plot(t, escale * (y - g), color=color, lw=1.2, ls=ls)
            a.scatter(t[key], scale * (g[key] - off), s=34, facecolor="white", edgecolor=INK, lw=1.2, zorder=6,
                      label="100 Hz keyframe (input)")
            e_ax.axhline(0, color=INK_2, lw=0.6)
            a.set_title(name, loc="left", fontsize=10.5, color=INK, pad=3)
            if unit:
                a.set_ylabel(unit, color=INK_2, fontsize=8.5)
            e_ax.set_ylabel(f"error ({eunit})", color=INK_2, fontsize=8)
            style(a); style(e_ax)
            plt.setp(a.get_xticklabels(), visible=False)
            if row == 3:
                e_ax.set_xlabel("time (ms at 1 kHz)", color=INK_2)
            if row == 0 and col == 0:
                handles, labels = a.get_legend_handles_labels()

        # whole-episode summary for this joint, missing steps only
        Lfull = np.arange(len(true)) % STRIDE != 0
        tp, tR = channels(true, s)
        summ = []
        for k, label, _, _ in LINES:
            pp, pR = channels(pred[k], s)
            pos = 1e6 * np.sqrt(np.mean((pp - tp)[Lfull] ** 2))
            ang = np.sqrt(np.mean(rotation_angle_deg(pR, tR)[Lfull] ** 2))
            summ.append(f"{label}: {pos:.1f} µm, {ang:.4f}°")
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.975), ncol=8, frameon=False, fontsize=9.5,
                   labelcolor=INK, columnspacing=1.4, handlelength=2.2)
        fig.suptitle(f"{NAMES[s]}: predicted vs ground-truth trajectory at 1 kHz, steps {c0}–{c0 + args.ms} of test episode "
                     f"{ep.split('_')[0]} (seed {args.seed})", x=0.01, y=0.99, ha="left", fontsize=13, color=INK)
        fig.text(0.01, 0.951, "Whole test episode, missing steps (position RMSE, rotation RMS angle):\n" +
                 "  ·  ".join(summ[:3]) + "\n" + "  ·  ".join(summ[3:]), fontsize=9, color=INK_2, va="top", linespacing=1.5)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.03)
        out = os.path.join(out_dir, f"{NAMES[s]}.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
