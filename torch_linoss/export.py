"""
Convert a trained checkpoint into a deployable package.

    checkpoint (best.pt from train.py, either backend)
      -> LinOSSCuda (Triton `linoss::scan` kernels), eval mode, recurrence cached (`freeze`)
      -> torch.export with a dynamic batch dimension
      -> AOTInductor package (.pt2): compiled Triton/C++ code, loadable without the model source via
         torch._inductor.aoti_load_package (Python) or the AOTI C++ runtime.

The package is checked against the eager reference model (`LinOSS`, FFT ops) on the dataset's test split.

    python -m torch_linoss.export --checkpoint outputs_torch/.../best.pt --dataset_name EigenWorms --seed 2345
"""

import argparse
import json
import os
import time

import torch

from torch_linoss.data import create_dataset, epoch_batches
from torch_linoss.models import LinOSS, LinOSSCuda


def load_models(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cuda", weights_only=True)
    reference = LinOSS(**ckpt["config"]).cuda()
    reference.load_state_dict(ckpt["state_dict"])
    reference.eval()
    deploy = LinOSSCuda.from_reference(reference).freeze()
    return reference, deploy


def export_package(model, example, out_path, max_batch=1024):
    batch = torch.export.Dim("batch", min=1, max=max_batch)
    program = torch.export.export(model, (example,), dynamic_shapes=({0: batch},))
    return torch._inductor.aoti_compile_and_package(program, package_path=out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--seed", type=int, default=2345)
    parser.add_argument("--config_dir", default="experiment_configs/repeats")
    parser.add_argument("--out", help="defaults to linoss.pt2 next to the checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    with open(os.path.join(args.config_dir, "LinOSS", f"{args.dataset_name}.json")) as f:
        cfg = json.load(f)
    dataset = create_dataset(
        cfg["data_dir"],
        args.dataset_name,
        include_time=cfg["time"].lower() == "true",
        T=float(cfg["T"]),
        use_presplit=cfg["use_presplit"],
        seed=args.seed,
    )
    reference, deploy = load_models(args.checkpoint)
    out_path = args.out or os.path.join(os.path.dirname(args.checkpoint), "linoss.pt2")

    t0 = time.time()
    example = dataset.test.X[: args.batch_size].cuda()
    path = export_package(deploy, example, out_path)
    print(f"Exported {path} in {time.time() - t0:.1f}s")
    package = torch._inductor.aoti_load_package(path)

    max_err, agree, preds_ref, preds_pkg, labels = 0.0, 0, [], [], []
    with torch.inference_mode():
        for X, y in epoch_batches(dataset.test, args.batch_size):
            ref, pkg = reference(X), package(X)
            max_err = max(max_err, (ref - pkg).abs().max().item() / ref.abs().max().item())
            preds_ref.append(ref)
            preds_pkg.append(pkg)
            labels.append(y)
    ref, pkg, y = torch.cat(preds_ref), torch.cat(preds_pkg), torch.cat(labels)
    if reference.classification:
        acc_ref = (ref.argmax(1) == y).float().mean().item()
        acc_pkg = (pkg.argmax(1) == y).float().mean().item()
        print(f"Test accuracy: eager reference {acc_ref:.4f}, exported package {acc_pkg:.4f}")
    else:
        mse_ref = ((ref[:, :, 0] - y) ** 2).mean().item()
        mse_pkg = ((pkg[:, :, 0] - y) ** 2).mean().item()
        print(f"Test MSE: eager reference {mse_ref:.6f}, exported package {mse_pkg:.6f}")
    print(f"Max relative output difference: {max_err:.2e}")


if __name__ == "__main__":
    main()
