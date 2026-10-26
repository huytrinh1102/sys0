"""
Comparison charts for the PyTorch port vs. the original JAX LinOSS. Writes PNGs to visual_output/:

    speed_inference.png      inference time vs. sequence length (JAX, PyTorch FFT, Triton, Triton + compile)
    speed_training.png       training-step time vs. sequence length
    ppg_mse_by_seed.png      PPG test MSE per seed: paper, JAX rerun, PyTorch
    ppg_validation_curves.png  PPG validation MSE during training: paper vs. PyTorch model sizes
    accuracy_by_seed.png     Heartbeat and EigenWorms test accuracy per seed

    python -m torch_linoss.plot_results
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "visual_output"
INK, INK_2, GRID = "#2b2b29", "#5f5e57", "#e4e4df"
# Categorical slots 1-5 in fixed order. Slots 3-5 are below 3:1 on white, so every series also gets a
# marker shape and a direct label.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
SEEDS = [2345, 3456, 4567, 5678, 6789]
# Each implementation keeps its color and marker in every chart.
IMPL_SLOT = {"jax": 0, "torch_fft": 1, "torch_triton": 2, "torch_triton_compiled": 3}


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def title(ax, text, sub=None):
    ax.set_title(text, loc="left", fontsize=12, color=INK, pad=22 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.plot([], [], color=color)  # keeps the color cycle untouched


# ---------------------------------------------------------------- speed


def load_bench():
    results = []
    for name in ("bench_torch.json", "bench_jax.json"):
        with open(os.path.join("outputs_torch", "benchmarks", name)) as f:
            results += json.load(f)["results"]
    return results


def speed_lines(ax, results, suite, batch, impls):
    for impl, label in impls:
        i = IMPL_SLOT[impl]
        rows = sorted((r["L"], r["ms"]) for r in results
                      if r["suite"] == suite and r["impl"] == impl and r.get("batch") == batch and r["ms"])
        L, ms = zip(*rows)
        ax.plot(L, ms, color=C[i], linewidth=2, marker=MARKERS[i], markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, label=label)
        end_label(ax, L[-1], ms[-1], f"{label}  {ms[-1]:.0f} ms", C[i])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(L, [f"{l // 1024}K" for l in L])
    ax.minorticks_off()
    ax.set_xlabel("Sequence length", color=INK_2)
    ax.set_ylabel("Time (ms, log scale)", color=INK_2)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    ax.set_xlim(right=ax.get_xlim()[1] * 4)  # room for end labels
    style(ax)


def plot_speed():
    results = load_bench()
    infer = [("jax", "JAX (original)"), ("torch_fft", "PyTorch FFT"),
             ("torch_triton", "PyTorch Triton"), ("torch_triton_compiled", "Triton + compile")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, batch in zip(axes, (4, 32)):
        speed_lines(ax, results, "sweep", batch, infer)
        title(ax, f"Inference time, batch {batch}", "EigenWorms model (2 blocks, H=128, P=64), A10 GPU")
    save(fig, "speed_inference.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    speed_lines(ax, results, "train", 4, infer[:1] + infer[2:3])
    title(ax, "Training step time (forward + backward + Adam)", "batch 4, same model")
    save(fig, "speed_training.png")


# ---------------------------------------------------------------- PPG


def seed_of(path):
    return int(re.search(r"seed_(\d+)", path).group(1))


def test_by_seed(pattern, scale=1.0):
    return {seed_of(r): scale * float(np.load(os.path.join(r, "test_metric.npy"))) for r in glob.glob(pattern)}


def plot_ppg_seeds():
    series = [
        ("Paper (JAX, shipped)", test_by_seed("outputs/LinOSS_IM/ppg/*", 100)),
        ("JAX rerun here", {2345: 100 * float(np.load("outputs_torch/jax_rerun_logs/jax_ppg_seed2345_pred.npz")["mse"])}),
        ("PyTorch, per-sample dropout", test_by_seed("outputs_torch/LinOSS_IM/ppg/*hidden_dim_16_*dropout_sample*", 100)),
        ("PyTorch, batch dropout + ε", test_by_seed("outputs_torch/LinOSS_IM/ppg/*hidden_dim_16_*dropout_batch_Aeps*", 100)),
    ]
    cap = 11.0  # diverged runs (~54) are drawn at the cap
    fig, ax = plt.subplots(figsize=(11, 5.2))
    offsets = np.linspace(-0.27, 0.27, len(series))
    for i, (label, vals) in enumerate(series):
        xs = [SEEDS.index(s) + offsets[i] for s in vals]
        ys = [min(v, cap) for v in vals.values()]
        ax.scatter(xs, ys, s=70, color=C[i], marker=MARKERS[i], edgecolors="white", linewidths=1.5,
                   label=label, zorder=3)
        for x, v in zip(xs, vals.values()):
            if v > cap:
                ax.annotate(f"diverged\n({v:.0f})", (x, cap), xytext=(0, -26), textcoords="offset points",
                            ha="center", fontsize=8, color=INK_2)
    paper = list(series[0][1].values())
    ax.axhline(np.mean(paper), color=C[0], linewidth=1, alpha=0.5)
    ax.annotate(f"paper mean {np.mean(paper):.2f}", (-0.45, np.mean(paper)), xytext=(0, -11),
                textcoords="offset points", ha="left", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(SEEDS)), [f"seed {s}" for s in SEEDS])
    ax.set_ylim(5.5, cap + 0.4)
    ax.set_ylabel("Test MSE (×10⁻², lower is better)", color=INK_2)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2, labelcolor=INK)
    style(ax)
    title(ax, "PPG-DaLiA test MSE by seed", "same splits and initial weights; diverged runs drawn at the top edge")
    save(fig, "ppg_mse_by_seed.png")


def val_curve(run_dir):
    steps = np.load(os.path.join(run_dir, "steps.npy"))
    vals = np.load(os.path.join(run_dir, "all_val_metric.npy"))[1:]  # drop the placeholder initial value
    steps = steps[steps > 0] if len(steps) > len(vals) else steps[1:]
    n = min(len(steps), len(vals))
    return steps[:n], 100 * vals[:n]


def plot_ppg_curves():
    runs = [
        ("Paper (JAX), 6×16×64", glob.glob("outputs/LinOSS_IM/ppg/*seed_2345")),
        ("PyTorch 6×16×64", glob.glob("outputs_torch/LinOSS_IM/ppg/*hidden_dim_16_*dropout_batch_Aeps*seed_2345")),
        ("PyTorch 4×64×128", glob.glob("outputs_torch/LinOSS_IM/ppg/*num_blocks_4_hidden_dim_64_*seed_2345")),
        ("PyTorch 4×128×128, lr 3e-4", glob.glob("outputs_torch/LinOSS_IM/ppg/*lr_0.0003_num_blocks_4_hidden_dim_128_*seed_2345")),
        ("PyTorch 4×224×224, lr 1e-4", glob.glob("outputs_torch/LinOSS_IM/ppg/*lr_0.0001_num_blocks_4_hidden_dim_224_*seed_2345")),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (label, dirs) in enumerate(runs):
        if not dirs:
            continue
        steps, vals = val_curve(dirs[0])
        best = np.argmin(vals)
        ax.plot(steps / 1000, vals, color=C[i], linewidth=2, marker=MARKERS[i], markevery=[best],
                markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                label=f"{label}   best {vals[best]:.2f}")
        peak = np.argmax(vals)
        if vals[peak] > 13:  # off the chart: say how far
            ax.annotate(f"blows up to {vals[peak]:.0f} at step {steps[peak] // 1000}k",
                        (steps[peak] / 1000, 13), xytext=(6, -10), textcoords="offset points",
                        fontsize=8, color=INK)
    ax.set_xlabel("Training step (thousands)", color=INK_2)
    ax.set_ylabel("Validation MSE (×10⁻²)", color=INK_2)
    ax.set_ylim(5.5, 13)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    style(ax)
    title(ax, "PPG-DaLiA validation MSE during training (seed 2345)",
          "blocks × width H × oscillators P; marker = best checkpoint (used for the test score)")
    save(fig, "ppg_validation_curves.png")


# ---------------------------------------------------------------- classification


def jax_rerun_heartbeat():
    out = {}
    for f in glob.glob("outputs_torch/jax_rerun_logs/jax_heartbeat_*.log"):
        hits = re.findall(r"Test metric: ([0-9.]+)", open(f).read())
        if hits:
            out[seed_of(f.replace("heartbeat_", "seed_"))] = 100 * float(hits[-1])
    return out


def plot_accuracy():
    datasets = {
        "Heartbeat": [
            ("Paper (JAX, shipped)", test_by_seed("outputs/LinOSS_IM/Heartbeat/*", 100)),
            ("JAX rerun here", jax_rerun_heartbeat()),
            ("PyTorch, per-sample dropout", test_by_seed("outputs_torch/LinOSS_IM/Heartbeat/*dropout_sample*", 100)),
            ("PyTorch, batch dropout", test_by_seed("outputs_torch/LinOSS_IM/Heartbeat/*dropout_batch*", 100)),
        ],
        "EigenWorms": [
            ("Paper (JAX, shipped)", test_by_seed("outputs/LinOSS_IM/EigenWorms/*", 100)),
            ("JAX rerun here", {}),
            ("PyTorch, per-sample dropout", {}),
            ("PyTorch, batch dropout", {k: v for k, v in test_by_seed(
                "outputs_torch/LinOSS_IM/EigenWorms/*dropout_batch*", 100).items() if k in (2345, 3456)}),
        ],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, (name, series) in zip(axes, datasets.items()):
        for i, (label, vals) in enumerate(series):
            if not vals:
                ax.scatter([], [], color=C[i], marker=MARKERS[i], label=label)
                continue
            xs = [i + 0.12 * (j - 2) for j, s in enumerate(SEEDS) if s in vals]
            ys = [vals[s] for s in SEEDS if s in vals]
            ax.scatter(xs, ys, s=55, color=C[i], marker=MARKERS[i], edgecolors="white", linewidths=1.5,
                       label=label, zorder=3)
            m = np.mean(ys)
            ax.plot([i - 0.32, i + 0.32], [m, m], color=INK, linewidth=1.5, zorder=4)
            ax.annotate(f"{m:.1f}" + (f" (n={len(ys)})" if len(ys) < 5 else ""), (i + 0.34, m),
                        va="center", fontsize=8, color=INK)
        ax.set_xticks(range(4), ["Paper", "JAX\nrerun", "PyTorch\nper-sample", "PyTorch\nbatch"])
        ax.set_ylabel("Test accuracy (%)", color=INK_2)
        style(ax)
        title(ax, name, "dots = seeds, bar = mean")
    for x in (1, 2):
        axes[1].text(x, 0.5, "not run", ha="center", fontsize=8, color=INK_2,
                     transform=axes[1].get_xaxis_transform())
    axes[1].text(3, 0.04, "seed 4567 diverged;\n5678, 6789 not run", ha="center", fontsize=8, color=INK_2,
                 transform=axes[1].get_xaxis_transform())
    save(fig, "accuracy_by_seed.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    plot_speed()
    plot_ppg_seeds()
    plot_ppg_curves()
    plot_accuracy()


if __name__ == "__main__":
    main()
