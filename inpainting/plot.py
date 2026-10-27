"""
Figures and summary for 100 Hz → 1 kHz end-effector inpainting.

    python plot.py --seed 2345

Writes to out/figures/:
- closeup_<stream>.png     prediction error (µm) over 200 ms of x/y/z for every method, with keyframes and ±2σ
- error_by_gap_step.png    position RMSE vs. step within the 10 ms gap (1–9 ms)
- per_stream.png           position and rotation RMSE per end effector
- calibration.png          predicted σ vs. actual |error| (NLL head)
- results.md               summary table (mean ± std over seeds) incl. zero-order hold
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import BASELINES
from data import OUT, POS_IDX, STREAMS, STRIDE

FIG = os.path.join(OUT, "figures")
INK, INK_2, GRID, MUTED = "#2b2b29", "#5f5e57", "#e4e4df", "#9a9a92"
METHODS = {  # key: (label, color); colors follow the method in every chart
    "linear": ("Linear interpolation", "#eda100"),
    "cubic": ("Cubic spline", "#1baf7a"),
    "causal": ("LinOSS causal (10 ms lookahead)", "#eb6834"),
    "bi": ("LinOSS bidirectional", "#2a78d6"),
}
NAMES = [n for n, _ in STREAMS]


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)


def runs(direction, head="nll"):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "runs", f"{direction}_{head}_seed_*", "test_metrics.json")))]


def preds(direction, seed, head="nll"):
    z = np.load(os.path.join(OUT, "runs", f"{direction}_{head}_seed_{seed}", "test_pred.npz"))
    eps = sorted({k.split("/")[0] for k in z.files})
    return {e: {k: z[f"{e}/{k}"] for k in ("pred", "true", "sigma") if f"{e}/{k}" in z.files} for e in eps}


def closeups(seed):
    """Error (prediction − ground truth, µm) over 200 ms of the fastest motion; keyframes sit at 0 error."""
    P = {d: preds(d, seed) for d in ("causal", "bi")}
    ep = sorted(P["bi"])[0]
    true = P["bi"][ep]["true"]
    pred = {"linear": BASELINES["linear"](true.astype(np.float64)), "cubic": BASELINES["cubic"](true.astype(np.float64)),
            "causal": P["causal"][ep]["pred"], "bi": P["bi"][ep]["pred"]}
    sig = {"causal": P["causal"][ep].get("sigma"), "bi": P["bi"][ep].get("sigma")}
    speed = np.linalg.norm(np.diff(true[:, POS_IDX[:3]], axis=0), axis=1)
    c0 = int(np.clip(np.argmax(np.convolve(speed, np.ones(200), "valid")), 0, len(true) - 201))
    c0 -= c0 % STRIDE
    sl = slice(c0, c0 + 201)
    t = np.arange(len(true))[sl] - c0
    key = (np.arange(len(true)) % STRIDE == 0)[sl]
    for s_idx, stream in ((0, "wrist"), (1, "finger_1")):
        fig, axes = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True)
        for j, ax in enumerate(axes):
            c = 9 * s_idx + j
            if sig["bi"] is not None:
                b = 2e6 * sig["bi"][sl, c]
                ax.fill_between(t, -b, b, color=METHODS["bi"][1], alpha=0.12, lw=0, label="LinOSS bidirectional ±2σ")
            for m, (label, color) in METHODS.items():
                e = 1e6 * (pred[m][sl, c] - true[sl, c])
                rms = np.sqrt(np.mean(e[~key] ** 2))
                ax.plot(t, e, color=color, lw=1.6, label=f"{label} ({rms:.1f} µm)")
            ax.scatter(t[key], np.zeros(key.sum()), s=14, facecolor="white", edgecolor=INK, zorder=4,
                       label="100 Hz keyframe (exact)")
            ax.axhline(0, color=INK, lw=0.8)
            ax.set_ylabel(f"{stream} {'xyz'[j]} error (µm)", color=INK_2, fontsize=9.5)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.32))
            style(ax)
        axes[-1].set_xlabel("time in window (ms at 1 kHz)", color=INK_2)
        fig.suptitle(f"{stream}: prediction error over 200 ms of the fastest motion (test episode {ep.split('_')[0]}, seed {seed})",
                     x=0.01, ha="left", fontsize=12, color=INK)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"closeup_{stream}.png"), dpi=150); plt.close(fig)


def error_by_gap(base):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ks = list(range(1, STRIDE))
    for m, (label, color) in METHODS.items():
        if m in base:
            v = [base[m]["pos_rmse_mm_by_gap_step"][str(k)] for k in ks]
            lo = hi = None
        else:
            rs = runs(m)
            if not rs:
                continue
            arr = np.array([[r["pos_rmse_mm_by_gap_step"][str(k)] for k in ks] for r in rs])
            v, lo, hi = arr.mean(0), arr.min(0), arr.max(0)
            if len(rs) > 1:
                ax.fill_between(ks, 1000 * lo, 1000 * hi, color=color, alpha=0.15, lw=0)
        ax.plot(ks, 1000 * np.asarray(v), color=color, marker="o", lw=2, markeredgecolor="white", label=label)
    ax.set_xticks(ks); ax.set_xlabel("Step within the 10 ms gap (ms after the keyframe)", color=INK_2)
    ax.set_ylabel("Position RMSE (µm)", color=INK_2); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title("Error inside each gap (test episodes; band = seed range)", loc="left", fontsize=12, color=INK)
    style(ax); fig.tight_layout(); fig.savefig(os.path.join(FIG, "error_by_gap_step.png"), dpi=150); plt.close(fig)


def per_stream(base):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    w = 0.2
    x = np.arange(len(NAMES))
    for ax, key, unit, scale in ((axes[0], "pos_rmse_mm", "Position RMSE (µm)", 1000), (axes[1], "rot_rmse_deg", "Rotation RMSE (°)", 1)):
        for i, (m, (label, color)) in enumerate(METHODS.items()):
            if m in base:
                v = np.array([base[m]["per_stream"][n][key] for n in NAMES]) * scale; e = None
            else:
                rs = runs(m)
                if not rs:
                    continue
                a = np.array([[r["per_stream"][n][key] for n in NAMES] for r in rs]) * scale
                v, e = a.mean(0), a.std(0)
            ax.bar(x + (i - 1.5) * w, v, width=w * 0.92, color=color, label=label, yerr=e,
                   error_kw=dict(ecolor=INK_2, lw=1, capsize=2))
        ax.set_xticks(x, NAMES); ax.set_ylabel(unit, color=INK_2); style(ax)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.suptitle("Error per end effector on missing steps (test episodes; LinOSS = mean ± std over seeds)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "per_stream.png"), dpi=150); plt.close(fig)


def calibration(seed):
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    for d in ("causal", "bi"):
        P = preds(d, seed)
        sig, err = [], []
        for v in P.values():
            if "sigma" not in v:
                continue
            miss = np.arange(len(v["true"])) % STRIDE != 0
            sig.append(v["sigma"][miss][:, POS_IDX].ravel()); err.append(np.abs(v["pred"] - v["true"])[miss][:, POS_IDX].ravel())
        if not sig:
            continue
        sig, err = 1e6 * np.concatenate(sig), 1e6 * np.concatenate(err)  # metres → µm
        edges = np.quantile(sig, np.linspace(0, 1, 16))
        idx = np.clip(np.digitize(sig, edges) - 1, 0, 14)
        xs = [sig[idx == i].mean() for i in range(15)]
        ys = [np.sqrt((err[idx == i] ** 2).mean()) for i in range(15)]
        ax.plot(xs, ys, color=METHODS[d][1], marker="o", lw=2, markeredgecolor="white", label=METHODS[d][0])
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], color=INK, lw=1, label="Perfect calibration")
    ax.set_xlabel("Predicted σ (µm, binned)", color=INK_2); ax.set_ylabel("Actual RMS error in bin (µm)", color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title(f"Uncertainty calibration, position channels (seed {seed})", loc="left", fontsize=11.5, color=INK)
    style(ax); fig.tight_layout(); fig.savefig(os.path.join(FIG, "calibration.png"), dpi=150); plt.close(fig)


def table(base):
    def ms(v, f="{:.2f}"):
        return f"{f.format(np.mean(v))} ± {f.format(np.std(v))}" if len(v) > 1 else f.format(v[0])
    rows = ["| Method | Residual RMSE (1 ≈ linear) | Position RMSE (µm) | Rotation RMSE (°) | Velocity RMSE (mm/s) | ±1σ / ±2σ coverage | Params | Seeds |",
            "|---|---|---|---|---|---|---|---|"]
    for m in ("zoh", "linear", "cubic"):
        b = base[m]
        rows.append(f"| {'Zero-order hold' if m == 'zoh' else METHODS[m][0]} | {b['rmse_res']:.3f} | {1000 * b['pos_rmse_mm']:.1f} | "
                    f"{b['rot_rmse_deg']:.4f} | {b['vel_rmse_mm_s']:.2f} | — | — | — |")
    for d in ("causal", "bi"):
        for head in ("nll", "mse"):
            rs = runs(d, head)
            if not rs:
                continue
            g = lambda k: [r[k] for r in rs]
            cov = f"{ms([100 * x for x in g('coverage_1sigma')], '{:.0f}')}% / {ms([100 * x for x in g('coverage_2sigma')], '{:.0f}')}%" \
                if "coverage_1sigma" in rs[0] else "—"
            rows.append(f"| {METHODS[d][0]} ({head.upper()}) | {ms(g('rmse_res'), '{:.3f}')} | {ms([1000 * x for x in g('pos_rmse_mm')], '{:.1f}')} | "
                        f"{ms(g('rot_rmse_deg'), '{:.4f}')} | {ms(g('vel_rmse_mm_s'))} | {cov} | {rs[0]['params']:,} | {len(rs)} |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2345)
    args = p.parse_args()
    os.makedirs(FIG, exist_ok=True)
    base = json.load(open(os.path.join(OUT, "baselines.json")))
    closeups(args.seed)
    error_by_gap(base)
    per_stream(base)
    calibration(args.seed)
    t = table(base)
    open(os.path.join(FIG, "results.md"), "w").write("# 100 Hz → 1 kHz end-effector inpainting (test episodes, missing steps)\n\n" + t + "\n")
    print(t)


if __name__ == "__main__":
    main()
