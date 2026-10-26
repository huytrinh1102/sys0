"""Benchmark settings shared by the PyTorch and JAX benchmarks (no torch/JAX imports)."""

import json
import os

import numpy as np

SWEEP_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
SWEEP_BATCHES = [4, 32]
SWEEP_CONFIG = dict(num_blocks=2, N=7, ssm_size=64, H=128, output_dim=5, classification=True)

# name: (sequence length, test-split size, config file)
DATASETS = {
    "Heartbeat": (405, 62, "Heartbeat"),
    "SelfRegulationSCP1": (896, 85, "SelfRegulationSCP1"),
    "SelfRegulationSCP2": (1152, 57, "SelfRegulationSCP2"),
    "EthanolConcentration": (1751, 79, "EthanolConcentration"),
    "MotorImagery": (3000, 57, "MotorImagery"),
    "EigenWorms": (17984, 36, "EigenWorms"),
    "ppg": (49920, 106, "ppg"),
}


def model_config(dataset, data_dir="data_dir", config_dir="experiment_configs/repeats"):
    """Model kwargs of a paper config; input/output sizes are read from the exported JAX init weights."""
    with open(os.path.join(config_dir, "LinOSS", f"{dataset}.json")) as f:
        cfg = json.load(f)
    init = np.load(os.path.join(data_dir, "jax_init", dataset, f"seed_{cfg['seeds'][0]}.npz"))
    return dict(
        num_blocks=int(cfg["num_blocks"]),
        N=init["linear_encoder.weight"].shape[1],
        ssm_size=int(cfg["ssm_dim"]),
        H=int(cfg["hidden_dim"]),
        output_dim=init["linear_layer.weight"].shape[0],
        classification=cfg["metric"] == "accuracy",
        output_step=int(cfg.get("output_step", 1)),
        discretization=cfg["linoss_discretization"],
    )
