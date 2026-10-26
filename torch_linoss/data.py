"""
PyTorch data pipeline that follows the original JAX setup (`data_dir/datasets.py`, `data_dir/dataloaders.py`):

- UEA datasets: the whole dataset is split 70/15/15 at random. The permutation is the one the JAX code
  draws for the same seed, exported by `export_npz.py`.
- PPG-DaLiA: the presplit train/val/test files.
- `include_time` prepends the channel t = T * i / L.
- Training batches: an endless stream that reshuffles every epoch and, like `Dataloader.loop`, only
  yields batches while `end < size`. Evaluation batches cover every sample once (`Dataloader.loop_epoch`).

Small datasets live on the GPU, so batching needs no host transfers.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch

# Datasets larger than this stay in host memory and are copied to the GPU batch by batch.
MAX_DEVICE_BYTES = 4 * 2**30


@dataclass
class Split:
    X: torch.Tensor  # (n, L, channels) float32
    y: torch.Tensor  # (n,) int64 class labels, or (n, L_out) float32 regression targets

    def __len__(self):
        return len(self.X)


@dataclass
class Dataset:
    name: str
    train: Split
    val: Split
    test: Split
    data_dim: int
    label_dim: int
    classification: bool


def add_time_channel(X, T):
    L = X.shape[1]
    ts = (T / L) * np.arange(L, dtype=np.float32)
    ts = np.broadcast_to(ts[None, :, None], (X.shape[0], L, 1))
    return np.concatenate([ts, X], axis=2)


def _to_torch(X, y, device):
    nbytes = X.nbytes + y.nbytes
    target = device if nbytes <= MAX_DEVICE_BYTES else "cpu"
    X, y = torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(np.ascontiguousarray(y))
    if target == "cpu" and torch.device(device).type == "cuda":
        return Split(X.pin_memory(), y.pin_memory())
    return Split(X.to(target), y.to(target))


def create_dataset(data_dir, name, include_time, T, use_presplit, seed, device="cuda"):
    """Load a processed dataset exported by `export_npz.py`."""
    if name == "ppg":
        arrays = np.load(os.path.join(data_dir, "processed_npz", "PPG", "ppg", "data.npz"))
        splits = {s: (arrays[f"X_{s}"], arrays[f"y_{s}"]) for s in ("train", "val", "test")}
        if not use_presplit:
            raise NotImplementedError("PPG is only used with its presplit files, as in the paper")
        classification = False
        label_dim = 1
    else:
        arrays = np.load(os.path.join(data_dir, "processed_npz", "UEA", name, "data.npz"))
        data, labels = arrays["data"], arrays["labels"]
        if use_presplit:
            raise NotImplementedError("UEA datasets are used with a random split, as in the paper")
        key = f"perm_seed_{seed}"
        if key in arrays:
            perm = arrays[key]
        else:
            print(f"No JAX split stored for seed {seed}; using a numpy permutation instead")
            perm = np.random.default_rng(seed).permutation(len(data))
        N = len(data)
        bound1, bound2 = int(N * 0.7), int(N * 0.85)
        idx = {"train": perm[:bound1], "val": perm[bound1:bound2], "test": perm[bound2:]}
        splits = {s: (data[i], labels[i]) for s, i in idx.items()}
        classification = True
        label_dim = int(labels.max()) + 1

    if include_time:
        splits = {s: (add_time_channel(X, T), y) for s, (X, y) in splits.items()}
    tensors = {s: _to_torch(X, y, device) for s, (X, y) in splits.items()}
    return Dataset(
        name=name,
        data_dim=splits["train"][0].shape[-1],
        label_dim=label_dim,
        classification=classification,
        **tensors,
    )


def _batch(split, idx, device):
    if split.X.device.type == "cpu":
        idx = idx.cpu()
        return (
            split.X[idx].to(device, non_blocking=True),
            split.y[idx].to(device, non_blocking=True),
        )
    return split.X[idx], split.y[idx]


def train_batches(split, batch_size, generator, device="cuda"):
    """Endless shuffled training batches, as `Dataloader.loop`."""
    size = len(split)
    if batch_size > size:
        raise ValueError("Batch size larger than dataset size")
    if batch_size == size:
        idx = torch.arange(size, device=split.X.device)
        while True:
            yield _batch(split, idx, device)
    while True:
        perm = torch.randperm(size, generator=generator).to(split.X.device)
        start, end = 0, batch_size
        while end < size:
            yield _batch(split, perm[start:end], device)
            start, end = end, end + batch_size


def epoch_batches(split, batch_size, device="cuda"):
    """Every sample once, in order, as `Dataloader.loop_epoch`."""
    for start in range(0, len(split), batch_size):
        idx = torch.arange(start, min(start + batch_size, len(split)), device=split.X.device)
        yield _batch(split, idx, device)
