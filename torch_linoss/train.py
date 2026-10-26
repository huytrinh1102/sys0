"""
Train LinOSS in PyTorch with the experiment configs of the paper (`experiment_configs/repeats/LinOSS/*.json`).

The loop follows the JAX `train.py`:
- Adam with the config learning rate (and the config `lr_scheduler`, applied once)
- train/validation metrics every `print_steps` steps
- the test metric is recomputed whenever the validation metric matches or beats its best value
- early stopping after more than 10 evaluations without improvement
- steps.npy, all_train_metric.npy, all_val_metric.npy, all_time.npy and test_metric.npy in the output folder

Also saved: best.pt (weights at the best validation metric, plus the model config) and timing.json.

    python -m torch_linoss.train --dataset_name EigenWorms --seeds 2345 --backend cuda
    python -m torch_linoss.train --dataset_name ppg --model mamba3      # Mamba-3 baseline, same loop
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from torch_linoss.data import create_dataset, epoch_batches, train_batches
from torch_linoss.models import MODELS


def load_config(config_dir, dataset_name):
    with open(os.path.join(config_dir, "LinOSS", f"{dataset_name}.json")) as f:
        return json.load(f)


def output_dir_name(cfg, num_steps):
    # Same fields as the JAX output folders, minus the CDE-only ones.
    return (
        f"T_{float(cfg['T']):.2f}_time_{cfg['time'].lower() == 'true'}_nsteps_{num_steps}"
        f"_lr_{float(cfg['lr'])}_num_blocks_{cfg['num_blocks']}_hidden_dim_{cfg['hidden_dim']}"
        f"_ssm_dim_{cfg['ssm_dim']}"
    )


def loss_fn(model, pred, y):
    if model.classification:
        # The JAX loss, -sum(y * log(softmax + 1e-8)). Unlike cross_entropy, the 1e-8 caps the per-sample
        # loss (~18.4) and its gradient, which matters for inputs with a large scale such as EigenWorms.
        probs = torch.softmax(pred.float(), dim=-1)
        return -torch.log(probs.gather(1, y[:, None]) + 1e-8).mean()
    return F.mse_loss(pred[:, :, 0].float(), y)


@torch.inference_mode()
def evaluate(model, split, batch_size, amp):
    model.eval()
    preds, labels = [], []
    for X, y in epoch_batches(split, batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            preds.append(model(X).float())
        labels.append(y)
    model.train()
    pred, y = torch.cat(preds), torch.cat(labels)
    if model.classification:
        return (pred.argmax(1) == y).float().mean().item()
    return ((pred[:, :, 0] - y) ** 2).mean(1).mean(0).item()


def build_linoss(args, cfg, dataset, output_step, disc):
    return MODELS[args.backend](
        num_blocks=int(cfg["num_blocks"]),
        N=dataset.data_dim,
        ssm_size=int(cfg["ssm_dim"]),
        H=int(cfg["hidden_dim"]),
        output_dim=dataset.label_dim,
        classification=dataset.classification,
        output_step=output_step,
        discretization=disc,
        dropout_mask=args.dropout_mask,
        A_eps=args.A_eps,
    )


def run_name(args, cfg, num_steps, seed):
    """(model folder, run folder) under output_parent_dir/<model folder>/<dataset>/."""
    base = output_dir_name(cfg, num_steps)
    if args.model == "mamba3":
        return "Mamba3", f"{base}_mamba3_seed_{seed}"
    eps = f"_Aeps_{args.A_eps:g}" if args.A_eps else ""
    return (
        f"LinOSS_{cfg['linoss_discretization']}",
        f"{base}_backend_{args.backend}_dropout_{args.dropout_mask}{eps}_seed_{seed}",
    )


def train_one_seed(cfg, seed, args):
    torch.manual_seed(seed)
    device = "cuda"
    num_steps = args.num_steps or int(cfg["num_steps"])
    print_steps = args.print_steps or int(cfg["print_steps"])
    batch_size = int(cfg["batch_size"])
    metric = cfg["metric"]
    disc = cfg["linoss_discretization"]

    dataset = create_dataset(
        cfg["data_dir"],
        cfg["dataset_name"],
        include_time=cfg["time"].lower() == "true",
        T=float(cfg["T"]),
        use_presplit=cfg["use_presplit"],
        seed=seed,
        device=device,
    )
    output_step = int(cfg.get("output_step", 1))
    if args.model == "mamba3":
        from torch_linoss.models.mamba3_net import Mamba3Net

        model = Mamba3Net(
            num_blocks=int(cfg["num_blocks"]),
            N=dataset.data_dim,
            H=int(cfg["hidden_dim"]),
            output_dim=dataset.label_dim,
            classification=dataset.classification,
            output_step=output_step,
        ).to(device)
    else:
        model = build_linoss(args, cfg, dataset, output_step, disc).to(device)
    if args.init_from_jax:

        path = os.path.join(cfg["data_dir"], "jax_init", cfg["dataset_name"], f"seed_{seed}.npz")
        init = {k: torch.from_numpy(v) for k, v in np.load(path).items()}
        missing, unexpected = model.load_state_dict(init, strict=False)
        assert not unexpected and all(".norm." in k for k in missing), (missing, unexpected)
        print(f"Initial weights from {path}")
    step_model = torch.compile(model) if args.compile else model

    lr = eval(cfg["lr_scheduler"])(float(cfg["lr"]))
    opt = torch.optim.Adam(model.parameters(), lr=lr, fused=True)

    model_dir, run_dir = run_name(args, cfg, num_steps, seed)
    out_dir = os.path.join(args.output_parent_dir, model_dir, cfg["dataset_name"], run_dir)
    if os.path.isdir(out_dir):
        if not args.overwrite:
            raise ValueError(f"{out_dir} already exists, pass --overwrite to replace it")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    print(f"Output: {out_dir}")
    print(
        f"{cfg['dataset_name']}: train/val/test {len(dataset.train)}/{len(dataset.val)}/{len(dataset.test)}, "
        f"L={dataset.train.X.shape[1]}, channels={dataset.data_dim}, "
        f"params={sum(p.numel() for p in model.parameters())}"
    )

    if metric == "accuracy":
        better_or_equal, worse_or_equal, best_of = (lambda a, b: a >= b), (lambda a, b: a <= b), max
        init = 0.0
    else:
        better_or_equal, worse_or_equal, best_of = (lambda a, b: a <= b), (lambda a, b: a >= b), min
        init = 100.0
    all_train, all_val, best_vals, all_time = [init], [init], [init], []
    test_metric = None
    no_improvement = 0
    eval_steps = []
    train_time = 0.0
    eval_time = 0.0

    generator = torch.Generator().manual_seed(seed)
    batches = train_batches(dataset.train, batch_size, generator, device)
    running_loss = torch.zeros((), device=device)
    model.train()
    torch.cuda.synchronize()
    start = time.time()
    for step in range(num_steps):
        X, y = next(batches)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            pred = step_model(X)
        loss = loss_fn(model, pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running_loss += loss.detach()

        if (step + 1) % print_steps == 0:
            torch.cuda.synchronize()
            train_time += time.time() - start
            t0 = time.time()
            train_metric = evaluate(model, dataset.train, args.eval_batch_size, args.amp)
            val_metric = evaluate(model, dataset.val, args.eval_batch_size, args.amp)
            elapsed = time.time() - start
            print(
                f"Step: {step + 1}, Loss: {running_loss.item() / print_steps:.5f}, "
                f"Train metric: {train_metric:.4f}, Validation metric: {val_metric:.4f}, Time: {elapsed:.1f}"
            )
            running_loss.zero_()
            stop = False
            if worse_or_equal(val_metric, best_of(best_vals)):
                no_improvement += 1
                stop = no_improvement > 10
            else:
                no_improvement = 0
            if better_or_equal(val_metric, best_of(best_vals)):
                best_vals.append(val_metric)
                test_metric = evaluate(model, dataset.test, args.eval_batch_size, args.amp)
                print(f"Test metric: {test_metric:.4f}")
                torch.save(
                    {"config": model.config, "state_dict": model.state_dict(), "step": step + 1},
                    os.path.join(out_dir, "best.pt"),
                )
            eval_time += time.time() - t0
            all_train.append(train_metric)
            all_val.append(val_metric)
            all_time.append(elapsed)
            eval_steps.append(step + 1)
            np.save(os.path.join(out_dir, "steps.npy"), np.array([0] + eval_steps))
            np.save(os.path.join(out_dir, "all_train_metric.npy"), np.array(all_train))
            np.save(os.path.join(out_dir, "all_val_metric.npy"), np.array(all_val))
            np.save(os.path.join(out_dir, "all_time.npy"), np.array(all_time))
            np.save(os.path.join(out_dir, "test_metric.npy"), np.array(test_metric))
            if stop:
                break
            torch.cuda.synchronize()
            start = time.time()

    steps_done = eval_steps[-1] if eval_steps else 0
    timing = {
        "steps": steps_done,
        "train_seconds": train_time,
        "eval_seconds": eval_time,
        "ms_per_train_step": 1000 * train_time / max(steps_done, 1),
        "test_metric": test_metric,
        "best_val_metric": best_of(best_vals),
    }
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(f"Final test metric: {test_metric}  ({timing['ms_per_train_step']:.2f} ms/step)")
    return timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="EigenWorms")
    parser.add_argument("--config_dir", default="experiment_configs/repeats")
    parser.add_argument("--seeds", type=int, nargs="*", help="defaults to the config seeds")
    parser.add_argument("--model", choices=["linoss", "mamba3"], default="linoss")
    parser.add_argument("--backend", choices=list(MODELS), default="cuda", help="LinOSS implementation")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--amp", action="store_true", help="bf16 autocast (scan/FFT stay fp32)")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, help="override the config")
    parser.add_argument("--print_steps", type=int, help="override the config")
    parser.add_argument(
        "--dropout_mask",
        choices=["batch", "sample"],
        default="batch",
        help="'batch': one dropout mask per batch, as in the JAX code; 'sample': torch's per-sample masks",
    )
    parser.add_argument("--num_blocks", type=int, help="override the config (number of LinOSS blocks)")
    parser.add_argument("--hidden_dim", type=int, help="override the config (H, model width)")
    parser.add_argument("--ssm_dim", type=int, help="override the config (P, oscillators per block)")
    parser.add_argument("--lr", type=float, help="override the config")
    parser.add_argument(
        "--A_eps",
        type=float,
        default=0.0,
        help="A = relu(A_diag) + A_eps; 0 is the original model, e.g. 1e-4 prevents A from collapsing to 0",
    )
    parser.add_argument(
        "--init_from_jax",
        action="store_true",
        help="start from the JAX initial weights of each seed (export them with torch_linoss.convert_eqx)",
    )
    parser.add_argument("--output_parent_dir", default="outputs_torch")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = load_config(args.config_dir, args.dataset_name)
    overrides = {k: getattr(args, k) for k in ("num_blocks", "hidden_dim", "ssm_dim", "lr")}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.model == "mamba3" and args.init_from_jax:
        parser.error("--init_from_jax only applies to LinOSS")
    if overrides and args.init_from_jax:
        sizes = {k for k in overrides if k != "lr"}
        if sizes:
            parser.error(f"--init_from_jax needs the config's model size; drop it or {sorted(sizes)}")
    cfg.update({k: str(v) for k, v in overrides.items()})
    for seed in args.seeds or cfg["seeds"]:
        print(f"Running experiment with seed: {seed}")
        train_one_seed(cfg, seed, args)


if __name__ == "__main__":
    main()
