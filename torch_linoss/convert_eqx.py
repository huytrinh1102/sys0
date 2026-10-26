"""
Convert Equinox LinOSS weights (`models/LinOSS.py`) to the PyTorch state_dict of `torch_linoss.models`.
Run it in the JAX environment from the repository root.

Default use: export the initial weights the JAX `train.py` would use for every seed of a config, so a
PyTorch run can start from exactly the same model (`python -m torch_linoss.train --init_from_jax`):

    python -m torch_linoss.convert_eqx --dataset_name EigenWorms

Output: {data_dir}/jax_init/{dataset}/seed_{seed}.npz
"""

import argparse
import json
import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from models.generate_model import create_model


def eqx_to_state_dict(model):
    """Equinox LinOSS -> {torch parameter name: numpy array}."""
    sd = {
        "linear_encoder.weight": model.linear_encoder.weight,
        "linear_encoder.bias": model.linear_encoder.bias,
        "linear_layer.weight": model.linear_layer.weight,
        "linear_layer.bias": model.linear_layer.bias,
    }
    for i, block in enumerate(model.blocks):
        for name in ("A_diag", "B", "C", "D", "steps"):
            sd[f"blocks.{i}.ssm.{name}"] = getattr(block.ssm, name)
        sd[f"blocks.{i}.glu.proj.weight"] = jnp.concatenate([block.glu.w1.weight, block.glu.w2.weight])
        sd[f"blocks.{i}.glu.proj.bias"] = jnp.concatenate([block.glu.w1.bias, block.glu.w2.bias])
    return {k: np.asarray(v, dtype=np.float32) for k, v in sd.items()}


def initial_model(cfg, seed, data_dim, label_dim):
    """The model the JAX `create_dataset_model_and_train` builds for this seed."""
    key = jr.PRNGKey(seed)
    _, modelkey, _, _ = jr.split(key, 4)
    model, _ = create_model(
        "LinOSS",
        data_dim,
        None,
        1,
        None,
        label_dim,
        hidden_dim=int(cfg["hidden_dim"]),
        num_blocks=int(cfg["num_blocks"]),
        ssm_dim=int(cfg["ssm_dim"]),
        ssm_blocks=int(cfg["ssm_blocks"]),
        classification=cfg["metric"] == "accuracy",
        output_step=int(cfg.get("output_step", 1)),
        linoss_discretization=cfg["linoss_discretization"],
        key=modelkey,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--config_dir", default="experiment_configs/repeats")
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, "LinOSS", f"{args.dataset_name}.json")) as f:
        cfg = json.load(f)
    if args.dataset_name == "ppg":
        arrays = np.load(os.path.join(cfg["data_dir"], "processed_npz", "PPG", "ppg", "data.npz"))
        data_dim, label_dim = arrays["X_train"].shape[-1], 1
    else:
        arrays = np.load(
            os.path.join(cfg["data_dir"], "processed_npz", "UEA", args.dataset_name, "data.npz")
        )
        data_dim, label_dim = arrays["data"].shape[-1], int(arrays["labels"].max()) + 1
    data_dim += cfg["time"].lower() == "true"

    out = os.path.join(cfg["data_dir"], "jax_init", args.dataset_name)
    os.makedirs(out, exist_ok=True)
    for seed in cfg["seeds"]:
        sd = eqx_to_state_dict(initial_model(cfg, seed, data_dim, label_dim))
        np.savez(os.path.join(out, f"seed_{seed}.npz"), **sd)
        print(f"{args.dataset_name} seed {seed}: {len(sd)} tensors -> {out}")


if __name__ == "__main__":
    main()
