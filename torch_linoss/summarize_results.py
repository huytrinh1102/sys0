"""
Compare PyTorch results (outputs_torch/) with the JAX results shipped with the repo (outputs/).

    python -m torch_linoss.summarize_results
"""

import glob
import json
import os

import numpy as np

DATASETS = [
    "EigenWorms",
    "EthanolConcentration",
    "Heartbeat",
    "MotorImagery",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "ppg",
]


def collect(pattern):
    metrics, ms_per_step, steps = [], [], []
    for run in sorted(glob.glob(pattern)):
        path = os.path.join(run, "test_metric.npy")
        if not os.path.exists(path):
            continue
        metrics.append(float(np.load(path)))
        timing = os.path.join(run, "timing.json")
        if os.path.exists(timing):
            with open(timing) as f:
                t = json.load(f)
            ms_per_step.append(t["ms_per_train_step"])
            steps.append(t["steps"])
    return metrics, ms_per_step, steps


def fmt(values, accuracy):
    if not values:
        return "-"
    if accuracy:
        return f"{100 * np.mean(values):.2f} ± {100 * np.std(values):.2f} (n={len(values)})"
    return f"{np.mean(values):.4f} ± {np.std(values):.4f} (n={len(values)})"


def main():
    print(
        "| Dataset | Disc. | Metric | JAX (paper, outputs/) | PyTorch LinOSS | Mamba-3 | "
        "LinOSS ms/step | Mamba-3 ms/step |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name in DATASETS:
        with open(f"experiment_configs/repeats/LinOSS/{name}.json") as f:
            cfg = json.load(f)
        disc = cfg["linoss_discretization"]
        accuracy = cfg["metric"] == "accuracy"
        jax, _, _ = collect(f"outputs/LinOSS_{disc}/{name}/*")
        # Paper-size PyTorch LinOSS with the batch-shared dropout of the JAX code (any A_eps).
        size = f"hidden_dim_{cfg['hidden_dim']}_ssm_dim_{cfg['ssm_dim']}_"
        lin, lin_ms, _ = collect(f"outputs_torch/LinOSS_{disc}/{name}/*_lr_{float(cfg['lr'])}_num_blocks_{cfg['num_blocks']}_{size}*dropout_batch*")
        m3, m3_ms, _ = collect(f"outputs_torch/Mamba3/{name}/*")
        ms = lambda v: f"{np.mean(v):.1f}" if v else "-"
        print(
            f"| {name} | {disc} | {'accuracy %' if accuracy else 'MSE'} | {fmt(jax, accuracy)} | "
            f"{fmt(lin, accuracy)} | {fmt(m3, accuracy)} | {ms(lin_ms)} | {ms(m3_ms)} |"
        )


if __name__ == "__main__":
    main()
