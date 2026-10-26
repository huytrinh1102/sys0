"""
One-time conversion of the processed datasets (pickled JAX arrays) to .npz, so the PyTorch code does not
need JAX. Run it in the JAX environment from the repository root:

    python -m torch_linoss.export_npz --data_dir data_dir

For every random-split dataset it also stores the 70/15/15 permutation the JAX pipeline uses for each
seed in the configs (`train.py` -> `create_dataset` -> `dataset_generator`), so both implementations train
and evaluate on exactly the same samples.

Output: {data_dir}/processed_npz/{UEA/<name>, PPG/ppg}/data.npz
"""

import argparse
import glob
import json
import os
import pickle

import jax.random as jr
import numpy as np


def load_pickle(path):
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f))


def jax_split_permutation(seed, N):
    """Permutation used by the JAX code for a random split with this seed."""
    key = jr.PRNGKey(seed)
    datasetkey, _, _, _ = jr.split(key, 4)
    permkey, _ = jr.split(datasetkey)
    return np.asarray(jr.permutation(permkey, N))


def config_seeds(config_dir, name):
    seeds = set()
    for path in glob.glob(os.path.join(config_dir, "*", f"{name}.json")):
        with open(path) as f:
            seeds.update(json.load(f)["seeds"])
    return sorted(seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_dir")
    parser.add_argument("--config_dir", default="experiment_configs/repeats")
    args = parser.parse_args()

    for folder in sorted(glob.glob(os.path.join(args.data_dir, "processed", "UEA", "*"))):
        name = os.path.basename(folder)
        data = load_pickle(os.path.join(folder, "data.pkl")).astype(np.float32)
        labels = load_pickle(os.path.join(folder, "labels.pkl")).astype(np.int64)
        arrays = {"data": data, "labels": labels}
        for seed in config_seeds(args.config_dir, name):
            arrays[f"perm_seed_{seed}"] = jax_split_permutation(seed, len(data))
        out = os.path.join(args.data_dir, "processed_npz", "UEA", name)
        os.makedirs(out, exist_ok=True)
        np.savez(os.path.join(out, "data.npz"), **arrays)
        print(f"{name}: data {data.shape}, labels {labels.shape}, seeds {config_seeds(args.config_dir, name)}")

    ppg = os.path.join(args.data_dir, "processed", "PPG", "ppg")
    if os.path.isdir(ppg):
        arrays = {}
        for split in ("train", "val", "test"):
            arrays[f"X_{split}"] = load_pickle(os.path.join(ppg, f"X_{split}.pkl")).astype(np.float32)
            arrays[f"y_{split}"] = load_pickle(os.path.join(ppg, f"y_{split}.pkl")).astype(np.float32)
        out = os.path.join(args.data_dir, "processed_npz", "PPG", "ppg")
        os.makedirs(out, exist_ok=True)
        np.savez(os.path.join(out, "data.npz"), **arrays)
        print("ppg:", {k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    main()
