"""
Train and evaluate LinOSS 100 Hz → 1 kHz end-effector inpainting.

    python train.py --direction bi --head nll --seed 2345
    python train.py --baselines            # zero-order hold / linear / cubic spline on the test split

Loss on missing steps only (normalized units):
- head nll: Gaussian NLL 0.5·[(y−ŷ)²·e^(−s) + s]  (= KL(true ‖ predicted Gaussian) up to a constant),
            after `--warmup_mse` steps of plain MSE on the mean
- head mse: MSE on the mean (ablation)
Outputs: out/runs/<direction>_<head>_seed_<seed>/{best.pt, history.json, test_metrics.json, test_pred.npz}
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from baselines import BASELINES
from data import OUT, POS_IDX, STREAMS, STRIDE, InpaintData, rot6d_to_rotmat, rotation_angle_deg
from models import InpaintNet, gaussian_nll, n_params

NAMES = [n for n, _ in STREAMS]


# ---------------------------------------------------------------- metrics
class Metrics:
    """Accumulates errors over episodes, on missing (non-keyframe) steps only. Inputs in original units."""

    def __init__(self):
        self.se_pos = np.zeros(len(NAMES)); self.n_pos = 0
        self.se_rot = np.zeros(len(NAMES))
        self.se_vel = np.zeros(len(NAMES)); self.n_vel = 0
        self.se_phase = np.zeros(STRIDE); self.n_phase = np.zeros(STRIDE)
        self.se_res = 0.0; self.n_res = 0
        self.nll = 0.0; self.cov1 = 0.0; self.cov2 = 0.0; self.n_unc = 0

    def add(self, pred, true, scale, sigma=None):
        """pred/true (L, 54) original units; scale (54,) = std·res_std (one unit of interpolation error);
        sigma (L, 54) predicted std in original units, or None."""
        L = len(true)
        missing = (np.arange(L) % STRIDE) != 0
        P = pred.reshape(L, len(NAMES), 9); T = true.reshape(L, len(NAMES), 9)
        dpos = (P[..., :3] - T[..., :3])[missing]                       # (M, S, 3) metres
        self.se_pos += (dpos ** 2).sum(-1).sum(0); self.n_pos += missing.sum()
        ang = rotation_angle_deg(rot6d_to_rotmat(P[missing][..., 3:]), rot6d_to_rotmat(T[missing][..., 3:]))
        self.se_rot += (ang ** 2).sum(0)
        vp, vt = np.diff(P[..., :3], axis=0) * 1000, np.diff(T[..., :3], axis=0) * 1000  # m/s
        vm = missing[1:] | missing[:-1]
        self.se_vel += ((vp - vt)[vm] ** 2).sum(-1).sum(0); self.n_vel += vm.sum()
        k = np.arange(L) % STRIDE
        se_t = (dpos ** 2).sum(-1).mean(-1)                             # per missing step, mean over streams
        np.add.at(self.se_phase, k[missing], se_t); np.add.at(self.n_phase, k[missing], 1)
        z = ((pred - true) / scale)[missing]
        self.se_res += (z ** 2).sum(); self.n_res += z.size
        if sigma is not None:
            s = sigma[missing]
            zz = (pred - true)[missing] / s
            self.nll += (0.5 * (zz ** 2 + np.log(2 * np.pi)) + np.log(s / scale)).sum()  # residual units
            self.cov1 += (np.abs(zz) < 1).sum(); self.cov2 += (np.abs(zz) < 2).sum(); self.n_unc += zz.size

    def result(self):
        r = dict(
            rmse_res=float(np.sqrt(self.se_res / self.n_res)),  # 1 ≈ linear interpolation
            pos_rmse_mm=float(1000 * np.sqrt(self.se_pos.sum() / (self.n_pos * len(NAMES)) / 3)),
            rot_rmse_deg=float(np.sqrt(self.se_rot.sum() / (self.n_pos * len(NAMES)))),
            vel_rmse_mm_s=float(1000 * np.sqrt(self.se_vel.sum() / (self.n_vel * len(NAMES)) / 3)),
            per_stream={n: dict(pos_rmse_mm=float(1000 * np.sqrt(self.se_pos[i] / self.n_pos / 3)),
                                rot_rmse_deg=float(np.sqrt(self.se_rot[i] / self.n_pos)),
                                vel_rmse_mm_s=float(1000 * np.sqrt(self.se_vel[i] / self.n_vel / 3)))
                        for i, n in enumerate(NAMES)},
            pos_rmse_mm_by_gap_step={int(k): float(1000 * np.sqrt(self.se_phase[k] / self.n_phase[k] / 3))
                                     for k in range(1, STRIDE)},
        )
        if self.n_unc:
            r.update(nll=float(self.nll / self.n_unc), coverage_1sigma=float(self.cov1 / self.n_unc),
                     coverage_2sigma=float(self.cov2 / self.n_unc))
        return r


@torch.inference_mode()
def evaluate(model, data, names, keep=False):
    model.eval()
    m = Metrics(); kept = {}
    for n in names:
        x, r, lin, mask = data.full(n)
        mu, logvar = model(x[None])
        pred = data.denorm(data.prediction(lin, mu[0]).float().cpu().numpy())
        sigma = data.sigma(logvar[0].float().cpu().numpy()) if logvar is not None else None
        true = data.raw[n]
        m.add(pred, true, data.std * data.res_std, sigma)
        if keep:
            kept[n] = dict(pred=pred, true=true, sigma=sigma)
    model.train()
    return m.result(), kept


def evaluate_baselines(data):
    out = {}
    for name, fn in BASELINES.items():
        for split in ("val", "test"):
            m = Metrics()
            for n in data.split[split]:
                true = data.raw[n]
                m.add(fn(true.astype(np.float64)), true, data.std * data.res_std)
            out[name if split == "test" else f"{name}_val"] = m.result()
        print(f"baseline {name}: pos {out[name]['pos_rmse_mm']:.3f} mm, rot {out[name]['rot_rmse_deg']:.3f} deg, "
              f"vel {out[name]['vel_rmse_mm_s']:.1f} mm/s")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "baselines.json"), "w"), indent=1)
    return out


# ---------------------------------------------------------------- training
def loss_fn(model, x, target, lin, mask, step, args):
    """target: correction to the interpolation in residual units (O(1))."""
    mu, logvar = model(x)
    miss = (1.0 - mask)[..., None]
    denom = miss.sum() * target.shape[-1]
    mse = ((mu - target) ** 2 * miss).sum() / denom
    if logvar is None or step <= args.warmup_mse:
        return mse, dict(mse=mse.item())
    nll_el = gaussian_nll(target, mu, logvar)
    if args.beta > 0:  # β-NLL (Seitzer et al. 2022): stop-gradient σ^(2β) weight keeps the mean from being neglected
        nll_el = nll_el * torch.exp(args.beta * logvar).detach()
    nll = (nll_el * miss).sum() / denom
    return nll, dict(mse=mse.item(), nll=nll.item())


def quick_figure(kept, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = sorted(kept)[0]
    k = kept[n]
    t0 = 2000
    sl = slice(t0, t0 + 200)
    t = np.arange(len(k["true"]))[sl] / 1000
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for j, ax in enumerate(axes):
        c = POS_IDX[j]
        ax.plot(t, 1000 * k["true"][sl, c], color="#2b2b29", lw=2, label="ground truth (1 kHz)")
        ax.plot(t, 1000 * k["pred"][sl, c], color="#2a78d6", lw=1.4, label="LinOSS")
        if k["sigma"] is not None:  # ±2σ band of the NLL head, in mm
            s = 1000 * k["sigma"][sl, c]
            ax.fill_between(t, 1000 * k["pred"][sl, c] - 2 * s, 1000 * k["pred"][sl, c] + 2 * s,
                            color="#2a78d6", alpha=0.2, lw=0, label="±2σ")
        key = (np.arange(len(k["true"])) % STRIDE == 0)[sl]
        ax.scatter(t[key], 1000 * k["true"][sl, c][key], s=14, color="#eb6834", zorder=3, label="100 Hz keyframes")
        ax.set_ylabel(f"wrist {'xyz'[j]} (mm)")
    axes[0].legend(fontsize=8, frameon=False, ncol=3)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = InpaintData()
    model = InpaintNet(args.direction, args.head, H=args.H, depth=args.depth, P=args.P,
                       f_min=args.f_min, f_max=args.f_max).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    name = f"{args.direction}_{args.head}{args.tag}_seed_{args.seed}"
    run_dir = os.path.join(OUT, "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"{name}: {n_params(model):,} params; episodes {[len(v) for v in data.split.values()]}", flush=True)

    wb = None
    if args.wandb == "on":
        import wandb
        wb = wandb.init(project=args.project, name=name, config=dict(vars(args), params=n_params(model),
                        split={k: len(v) for k, v in data.split.items()}, streams=NAMES), dir=OUT,
                        reinit=True)

    best, bad, history, t_train = float("inf"), 0, [], 0.0
    running = {}
    torch.cuda.synchronize(); t0 = time.time()
    for step in range(1, args.max_steps + 1):
        x, y, lin, mask = data.batch(args.batch, args.window, rng)
        loss, parts = loss_fn(model, x, y, lin, mask, step, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        for k, v in parts.items():
            running[k] = running.get(k, 0.0) + v
        if step % args.eval_every == 0:
            torch.cuda.synchronize(); t_train += time.time() - t0
            val, _ = evaluate(model, data, data.split["val"])
            rec = dict(step=step, **{f"train_{k}": v / args.eval_every for k, v in running.items()},
                       **{f"val_{k}": v for k, v in val.items() if not isinstance(v, dict)},
                       ms_per_step=1000 * t_train / step, lr=opt.param_groups[0]["lr"])
            running = {}
            history.append(rec)
            improved = val["rmse_res"] < best
            if improved:
                best, bad = val["rmse_res"], 0
                torch.save(dict(config=model.config, state_dict=model.state_dict(), step=step,
                                norm=dict(mean=data.mean, std=data.std, res_std=data.res_std)),
                           os.path.join(run_dir, "best.pt"))
            else:
                bad += 1
            print(f"step {step}: " + ", ".join(f"{k} {v:.4g}" for k, v in rec.items() if k != "step")
                  + (" *" if improved else ""), flush=True)
            if wb:
                wb.log(rec, step=step)
            if bad >= args.patience:
                break
            torch.cuda.synchronize(); t0 = time.time()

    ckpt = torch.load(os.path.join(run_dir, "best.pt"), weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test, kept = evaluate(model, data, data.split["test"], keep=True)
    # latency: one median-length test episode, batch 1, eager
    n_med = sorted(data.split["test"], key=lambda n: len(data.raw[n]))[len(data.split["test"]) // 2]
    xm = data.full(n_med)[0][None]
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(xm)
        torch.cuda.synchronize(); ts = time.time()
        for _ in range(20):
            model(xm)
        torch.cuda.synchronize()
    test.update(name=name, params=n_params(model), best_step=ckpt["step"], steps=history[-1]["step"],
                ms_per_step=history[-1]["ms_per_step"], infer_ms_per_episode=1000 * (time.time() - ts) / 20,
                infer_episode_steps=int(xm.shape[1]))
    np.savez_compressed(os.path.join(run_dir, "test_pred.npz"),
                        **{f"{n}/{k}": v for n, d in kept.items() for k, v in d.items() if v is not None})
    json.dump(history, open(os.path.join(run_dir, "history.json"), "w"), indent=1)
    json.dump(test, open(os.path.join(run_dir, "test_metrics.json"), "w"), indent=1)
    print(f"TEST {name}: pos {test['pos_rmse_mm']:.3f} mm, rot {test['rot_rmse_deg']:.3f} deg, "
          f"vel {test['vel_rmse_mm_s']:.1f} mm/s" + (f", NLL {test['nll']:.3f}, cov1σ {test['coverage_1sigma']:.3f}"
          if "nll" in test else "") + f" (best step {ckpt['step']})", flush=True)
    if wb:
        import wandb
        wb.summary.update({f"test_{k}": v for k, v in test.items() if not isinstance(v, dict)})
        tbl = wandb.Table(columns=["stream", "pos_rmse_mm", "rot_rmse_deg", "vel_rmse_mm_s"],
                          data=[[s, v["pos_rmse_mm"], v["rot_rmse_deg"], v["vel_rmse_mm_s"]]
                                for s, v in test["per_stream"].items()])
        gap = wandb.Table(columns=["gap_step_ms", "pos_rmse_mm"],
                          data=[[k, v] for k, v in test["pos_rmse_mm_by_gap_step"].items()])
        fig_path = os.path.join(run_dir, "quick_wrist.png")
        quick_figure(kept, fig_path)
        wb.log({"test/per_stream": tbl, "test/by_gap_step": gap, "test/wrist_closeup": wandb.Image(fig_path)})
        art = wandb.Artifact(f"model-{name}", type="model")
        art.add_file(os.path.join(run_dir, "best.pt"))
        wb.log_artifact(art)
        wb.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--direction", choices=["bi", "causal"], default="bi")
    p.add_argument("--head", choices=["nll", "mse"], default="nll")
    p.add_argument("--seed", type=int, default=2345)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--P", type=int, default=128)
    p.add_argument("--f_min", type=float, default=0.5)
    p.add_argument("--f_max", type=float, default=100.0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--window", type=int, default=2001)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--warmup_mse", type=int, default=500)
    p.add_argument("--beta", type=float, default=0.5, help="β-NLL weight exponent; 0 = plain NLL")
    p.add_argument("--wandb", choices=["on", "off"], default="on")
    p.add_argument("--project", default="linoss-inpainting")
    p.add_argument("--tag", default="")
    p.add_argument("--baselines", action="store_true")
    args = p.parse_args()
    # Full fp32: TF32 keeps ~3 significant digits, and the corrections are ~1e-3 of the signal scale.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.baselines:
        evaluate_baselines(InpaintData())
    else:
        train(args)


if __name__ == "__main__":
    main()
