"""
Demo plots in the style of ACMPolicy/serving/demo.py: per test episode, full-episode and per-chunk
GT-vs-prediction figures for the right arm EE and the 5 right fingertips, in both representations.

    python demo.py --run bi_nll_seed_2345 --num_episodes 3
    python demo.py --run causal_mse_seed_2345 --episode_ids 260810120617_... --chunk_ms 200

Writes out/demo/<run>/<episode>/{lie,quaternion}/:
  full_arm.png, full_hand_right.png      whole episode at 1 kHz
  chunks/chunk_XXX_{arm,hand_right}.png  zoomed windows (default 4 × 200 ms over the fastest motion)
Lie = position xyz + rotation matrix entries r00..r22 (12 dims); quaternion = xyz + qw qx qy qz (7 dims).
GT blue, prediction orange, 100 Hz keyframes (the model's only input) green. Only the right side is modelled
(the left arm moves ~1 cm in this task), so the arm figure has one column.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import N_STREAMS, OUT, STRIDE, rot6d_to_rotmat
from plot import preds

QUAT_LABELS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
LIE_LABELS = ["x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22"]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]  # finger_1..5


def rotmat_to_quat(R):
    """(..., 3, 3) → (..., 4) wxyz, Shepperd's method."""
    m = R.reshape(-1, 3, 3)
    q = np.empty((len(m), 4))
    tr = np.trace(m, axis1=1, axis2=2)
    for i, (r, t) in enumerate(zip(m, tr)):
        if t > 0:
            s = 2 * np.sqrt(t + 1)
            q[i] = [s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        else:
            k = int(np.argmax(np.diag(r)))
            if k == 0:
                s = 2 * np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
                q[i] = [(r[2, 1] - r[1, 2]) / s, s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
            elif k == 1:
                s = 2 * np.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
                q[i] = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s]
            else:
                s = 2 * np.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
                q[i] = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4]
    return q.reshape(R.shape[:-2] + (4,))


def to_reps(y):
    """(L, 54) → {"lie": (L, 6, 12), "quaternion": (L, 6, 7)} per stream (0 = wrist, 1..5 = fingers)."""
    L = len(y)
    y = y.reshape(L, N_STREAMS, 9)
    R = rot6d_to_rotmat(y[..., 3:])
    return {"lie": np.concatenate([y[..., :3], R.reshape(L, N_STREAMS, 9)], -1),
            "quaternion": np.concatenate([y[..., :3], rotmat_to_quat(R)], -1)}


def match_hemisphere(pred_q, gt_q):
    """Flip predicted quaternions onto the GT hemisphere (q and −q are the same rotation)."""
    out = pred_q.copy()
    flip = (out[..., 3:] * gt_q[..., 3:]).sum(-1) < 0
    out[..., 3:][flip] *= -1
    return out


def mse_missing(pred, gt, t):
    miss = t % STRIDE != 0
    return float(np.mean((pred[miss] - gt[miss]) ** 2))


def plot_arm(gt, pred, t, labels, title, out_path, spans=(), dots=True, lw=0.8):
    D = len(labels)
    fig, axes = plt.subplots(D, 1, figsize=(12, 2 * D), sharex=True)
    fig.suptitle(f"{title}  MSE(missing steps)={mse_missing(pred, gt, t):.3e}", fontsize=12, y=0.998)
    key = t % STRIDE == 0
    for d, ax in enumerate(axes):
        ax.plot(t, gt[:, d], linewidth=lw, color="tab:blue", label="GT")
        ax.plot(t, pred[:, d], linewidth=lw, color="tab:orange", alpha=0.85, label="Pred")
        if dots:
            ax.plot(t[key], gt[key, d], "o", markersize=3, color="tab:green", zorder=5, label="Obs (100 Hz)")
        for ci, (ws, we) in enumerate(spans):
            ax.axvspan(ws, we, alpha=0.08, color=plt.cm.tab10.colors[ci % 10])
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)
        ax.set_ylabel(labels[d], fontsize=7, fontweight="bold")
        if d == 0:
            ax.set_title("Right Arm", fontsize=10, fontweight="bold")
            ax.legend(fontsize=6, loc="upper right")
    axes[-1].set_xlabel("1 kHz step", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_hand(gt, pred, t, labels, title, out_path, spans=(), dots=True, lw=0.7):
    """gt/pred: (T, 5, D)."""
    D = len(labels)
    fig, axes = plt.subplots(D, 5, figsize=(26, 2 * D), sharex=True)
    fig.suptitle(f"{title}  MSE(missing steps)={mse_missing(pred, gt, t):.3e}", fontsize=12, y=0.998)
    key = t % STRIDE == 0
    for f in range(5):
        for d in range(D):
            ax = axes[d, f]
            ax.plot(t, gt[:, f, d], linewidth=lw, color="tab:blue")
            ax.plot(t, pred[:, f, d], linewidth=lw, color="tab:orange", alpha=0.85)
            if dots:
                ax.plot(t[key], gt[key, f, d], "o", markersize=2.5, color="tab:green", zorder=5)
            for ci, (ws, we) in enumerate(spans):
                ax.axvspan(ws, we, alpha=0.08, color=plt.cm.tab10.colors[ci % 10])
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=5)
            if d == 0:
                ax.set_title(FINGER_NAMES[f], fontsize=9, fontweight="bold")
            if f == 0:
                ax.set_ylabel(labels[d], fontsize=6)
    axes[0, 4].plot([], [], color="tab:blue", label="GT")
    axes[0, 4].plot([], [], color="tab:orange", label="Pred")
    axes[0, 4].plot([], [], marker="o", color="tab:green", linewidth=0, label="Obs (100 Hz)")
    axes[0, 4].legend(fontsize=6, loc="upper right")
    for c in range(5):
        axes[-1, c].set_xlabel("1 kHz step", fontsize=6)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def chunk_starts(true, n_chunks, length):
    """Non-overlapping windows over the fastest wrist motion, starting on keyframes, skipping start-up."""
    speed = np.r_[np.linalg.norm(np.diff(true[:, :3], axis=0), axis=1), 0.0]
    score = np.convolve(speed, np.ones(length), "valid")
    score[:200] = -1
    starts = []
    for _ in range(n_chunks):
        s = int(np.argmax(score))
        if score[s] < 0:
            break
        s -= s % STRIDE
        starts.append(s)
        score[max(0, s - length):s + length] = -1
    return sorted(starts)


def run_episode(run, ep, P, out_dir, n_chunks, chunk_ms):
    true = P[ep]["true"].astype(np.float64)
    pred = P[ep]["pred"].astype(np.float64)
    G, Q = to_reps(true), to_reps(pred)
    Q["quaternion"] = match_hemisphere(Q["quaternion"], G["quaternion"])
    t = np.arange(len(true))
    starts = chunk_starts(true, n_chunks, chunk_ms + 1)
    spans = [(s, s + chunk_ms) for s in starts]
    res = {}
    for rep, labels in (("lie", LIE_LABELS), ("quaternion", QUAT_LABELS)):
        d = os.path.join(out_dir, rep)
        os.makedirs(os.path.join(d, "chunks"), exist_ok=True)
        g, p = G[rep], Q[rep]
        name = f"({rep.capitalize()}) Full Episode — {ep} — {run}"
        plot_arm(g[:, 0], p[:, 0], t, labels, f"Arm EE {name}", os.path.join(d, "full_arm.png"),
                 spans=spans, dots=False, lw=0.7)
        plot_hand(g[:, 1:], p[:, 1:], t, labels, f"Right Hand EE {name}", os.path.join(d, "full_hand_right.png"),
                  spans=spans, dots=False, lw=0.6)
        for ci, s in enumerate(starts):
            sl = slice(s, s + chunk_ms + 1)
            cname = f"({rep.capitalize()}) — Chunk {ci} (steps {s}–{s + chunk_ms}) — {run}"
            plot_arm(g[sl, 0], p[sl, 0], t[sl], labels, f"Arm EE {cname}",
                     os.path.join(d, "chunks", f"chunk_{ci:03d}_arm.png"))
            plot_hand(g[sl, 1:], p[sl, 1:], t[sl], labels, f"Right Hand EE {cname}",
                      os.path.join(d, "chunks", f"chunk_{ci:03d}_hand_right.png"))
        res[f"arm_{rep}"] = mse_missing(p[:, 0], g[:, 0], t)
        res[f"hand_{rep}"] = mse_missing(p[:, 1:], g[:, 1:], t)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="bi_nll_seed_2345", help="folder in out/runs/")
    p.add_argument("--num_episodes", type=int, default=3)
    p.add_argument("--episode_ids", nargs="+", default=None)
    p.add_argument("--n_chunks", type=int, default=4)
    p.add_argument("--chunk_ms", type=int, default=200)
    p.add_argument("--out_dir", default=os.path.join(OUT, "demo"))
    args = p.parse_args()
    direction, head, _, seed = args.run.split("_")
    P = preds(direction, int(seed), head)
    eps = sorted(P)
    if args.episode_ids:
        eps = [e for e in eps if any(e.startswith(i) for i in args.episode_ids)]
    else:
        eps = eps[:args.num_episodes]
    keys = ["arm_lie", "hand_lie", "arm_quaternion", "hand_quaternion"]
    summary = []
    for i, ep in enumerate(eps):
        print(f"[{i + 1}/{len(eps)}] {ep}")
        res = run_episode(args.run, ep, P, os.path.join(args.out_dir, args.run, ep), args.n_chunks, args.chunk_ms)
        print("  " + "  ".join(f"{k}={res[k]:.3e}" for k in keys))
        summary.append((ep, res))
    lines = [f"Demo results ({len(summary)} test episodes), run {args.run}, MSE on missing 1 kHz steps", ""]
    lines += [f"{k:>16s} MSE: {np.mean([s[1][k] for s in summary]):.3e} +/- {np.std([s[1][k] for s in summary]):.3e}"
              for k in keys]
    lines += ["", "Per-episode:"] + [f"  {e}: " + " ".join(f"{k}={r[k]:.3e}" for k in keys) for e, r in summary]
    os.makedirs(os.path.join(args.out_dir, args.run), exist_ok=True)
    open(os.path.join(args.out_dir, args.run, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    print(f"Results saved to: {os.path.join(args.out_dir, args.run)}")


if __name__ == "__main__":
    main()
