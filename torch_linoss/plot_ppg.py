"""
Plot PPG-DaLiA heart-rate predictions of the JAX and PyTorch LinOSS models against the ground truth:
one figure per test window, with one stacked subplot per model against the ground truth.

    python -m torch_linoss.plot_ppg --jax_pred jax_ppg_seed2345_pred.npz \
        --torch_checkpoint outputs_torch/.../best.pt --out_dir visual_output

Several checkpoints (one subplot each, plus a predicted-vs-truth scatter):

    python -m torch_linoss.plot_ppg --jax_pred ... --torch_checkpoint a/best.pt b/best.pt \
        --torch_labels "PyTorch 6x16x64" "PyTorch 4x64x128" --suffix _all_models

`--jax_pred` is an .npz with the JAX model's test predictions (`pred`, shape (n_test, 390)); the original
training code does not save weights, so these come from the run that trained the model.
Targets are heart-rate labels every 2 s, min-max normalized to [-1, 1] per subject by process_ppg.py.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_linoss.data import create_dataset, epoch_batches
from torch_linoss.models import LinOSSCuda

SECONDS_PER_LABEL = 2.0
COLORS = {"truth": "#2b2b29"}
# Categorical slots in fixed order: JAX first, then the PyTorch checkpoints in the order given.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRID = "#e4e4df"
TEXT_SECONDARY = "#5f5e57"


@torch.inference_mode()
def torch_predictions(checkpoint, dataset):
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model = LinOSSCuda(**ckpt["config"]).cuda()
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()
    preds = [model(X)[:, :, 0].float().cpu() for X, _ in epoch_batches(dataset.test, 4)]
    return torch.cat(preds).numpy()


def mse(pred, y):
    return float(((pred - y) ** 2).mean())


def style_axis(ax):
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def legend_above(ax):
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9,
              labelcolor=COLORS["truth"], borderaxespad=0.2)


def plot_window(idx, y, models, out_path):
    """models: list of (label, predictions (n_test, 390), color)."""
    t = np.arange(y.shape[1]) * SECONDS_PER_LABEL
    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.2 + 2.5 * n), sharex=True, sharey=True, squeeze=False)
    for ax, (label, pred, color) in zip(axes[:, 0], models):
        ax.plot(t, y[idx], color=COLORS["truth"], linewidth=2.0, label="Ground truth")
        ax.plot(t, pred[idx], color=color, linewidth=2.0, label=label)
        ax.set_title(f"{label}   window MSE {mse(pred[idx], y[idx]):.4f}   (test MSE {mse(pred, y):.4f})",
                     loc="left", fontsize=11, color=COLORS["truth"])
        ax.set_ylabel("Heart rate\n(normalized)", color=TEXT_SECONDARY, fontsize=10)
        legend_above(ax)
        style_axis(ax)
    axes[-1, 0].set_xlabel("Time (s)", color=TEXT_SECONDARY, fontsize=10)
    fig.suptitle(f"PPG-DaLiA test window {idx}: LinOSS heart-rate prediction", x=0.01, ha="left",
                 fontsize=13, color=COLORS["truth"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter(y, models, out_path):
    """Predicted vs. ground-truth heart rate over every test label, one panel per model."""
    n = len(models)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.4 * rows + 0.6), squeeze=False)
    lims = (-1.05, 1.05)
    for ax, (label, pred, color) in zip(axes.flat, models):
        ax.scatter(y.ravel(), pred.ravel(), s=2, color=color, alpha=0.12, linewidths=0, rasterized=True)
        ax.plot(lims, lims, color=COLORS["truth"], linewidth=1)
        ax.text(0.03, 0.97, f"test MSE {mse(pred, y):.4f}", transform=ax.transAxes, va="top",
                fontsize=9, color=COLORS["truth"])
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_title(label, loc="left", fontsize=11, color=COLORS["truth"])
        ax.set_xlabel("Ground truth", color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel("Prediction", color=TEXT_SECONDARY, fontsize=9)
        style_axis(ax)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("PPG-DaLiA test set: predicted vs. ground-truth heart rate (normalized); line = perfect prediction",
                 x=0.01, ha="left", fontsize=12, color=COLORS["truth"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def default_windows(y, preds, k):
    """The k windows with the most heart-rate variation, plus the median-error window."""
    dynamic = [int(i) for i in np.argsort(-y.std(axis=1))[:k]]
    err = sum(((p - y) ** 2).mean(1) for p in preds)
    median = int(np.argsort(err)[len(err) // 2])
    return dynamic + ([median] if median not in dynamic else [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax_pred", required=True)
    parser.add_argument("--torch_checkpoint", nargs="+", required=True)
    parser.add_argument("--torch_labels", nargs="*", help="one label per checkpoint")
    parser.add_argument("--seed", type=int, default=2345)
    parser.add_argument("--windows", type=int, nargs="*", help="test window indices (default: automatic)")
    parser.add_argument("--num_dynamic", type=int, default=2)
    parser.add_argument("--suffix", default="", help="appended to the output file names")
    parser.add_argument("--out_dir", default="visual_output")
    args = parser.parse_args()
    labels = args.torch_labels or (["PyTorch (Triton port)"] if len(args.torch_checkpoint) == 1 else
                                   [f"PyTorch #{i + 1}" for i in range(len(args.torch_checkpoint))])
    assert len(labels) == len(args.torch_checkpoint), "one --torch_labels entry per checkpoint"
    assert len(labels) + 1 <= len(SERIES), f"at most {len(SERIES) - 1} PyTorch checkpoints"

    dataset = create_dataset("data_dir", "ppg", include_time=True, T=1.0, use_presplit=True, seed=args.seed)
    y = dataset.test.y.cpu().numpy()
    jax = np.load(args.jax_pred)
    assert np.allclose(jax["y"], y), "JAX predictions belong to a different test set"
    models = [("JAX (original)", jax["pred"], SERIES[0])]
    for i, (label, ckpt) in enumerate(zip(labels, args.torch_checkpoint)):
        models.append((label, torch_predictions(ckpt, dataset), SERIES[i + 1]))

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, f"ppg_test_predictions{args.suffix}.npz"), y=y,
             **{label: pred for label, pred, _ in models})
    windows = args.windows or default_windows(y, [p for _, p, _ in models], args.num_dynamic)
    for idx in windows:
        out = os.path.join(args.out_dir, f"ppg_window_{idx:03d}{args.suffix}.png")
        plot_window(idx, y, models, out)
        print(f"window {idx}: " + ", ".join(f"{l} {mse(p[idx], y[idx]):.4f}" for l, p, _ in models) + f" -> {out}")
    if len(models) > 2:
        out = os.path.join(args.out_dir, f"ppg_pred_vs_gt_scatter{args.suffix}.png")
        plot_scatter(y, models, out)
        print("->", out)
    print("test MSE: " + ", ".join(f"{l} {mse(p, y):.4f}" for l, p, _ in models))


if __name__ == "__main__":
    main()
