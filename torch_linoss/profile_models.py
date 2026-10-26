"""
Parameters, forward GFLOPs and inference latency of LinOSS and Mamba-3 models for a given sequence length,
predicting at every step (output_step=1).

FLOPs: torch's FlopCounterMode counts the dense ops (linear layers, GLU, ...). The sequence-mixing kernels
are custom (LinOSS's Triton scan, Mamba-3's Triton kernels), so their cost is added analytically from the
recurrent form (multiply and add each count as one FLOP):
- LinOSS: per step and real channel (2P per block): x = M x + f u (2x2 matvec + 2 MACs) -> 12 FLOPs.
- Mamba-3 SISO: per step and head, state h (headdim x d_state): h = a h + B x^T, y = h C -> ~5 * headdim * d_state.

    python -m torch_linoss.profile_models --seq_len 100
    CUDA_HOME=... python -m torch_linoss.profile_models --seq_len 100 --versions   # every GPU execution mode
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from torch_linoss.models import LinOSSCuda

N_IN = 7  # PPG input channels (time + 6 sensors)


def models(seq_len):
    linoss = [
        ("LinOSS 6×16×64 (paper)", 6, 16, 64),
        ("LinOSS 4×64×128", 4, 64, 128),
        ("LinOSS 4×128×128", 4, 128, 128),
        ("LinOSS 4×224×224", 4, 224, 224),
    ]
    out = []
    for label, nb, H, P in linoss:
        kw = dict(num_blocks=nb, N=N_IN, ssm_size=P, H=H, output_dim=1, classification=False, output_step=1)
        out.append((label, "linoss", kw, 12 * 2 * P * nb * seq_len))
    from torch_linoss.models.mamba3_net import Mamba3Net

    kw = dict(num_blocks=6, N=N_IN, H=16, output_dim=1, classification=False, output_step=1)
    m = Mamba3Net(**kw)
    c = m.config
    nheads = c["expand"] * c["H"] // c["headdim"]
    out.append(("Mamba-3 6×16 (PPG config)", "mamba3", kw,
                5 * c["headdim"] * c["d_state"] * nheads * c["num_blocks"] * seq_len))
    return out


def dense_flops(model, x):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model(x)
    return fc.get_total_flops()


def latency_ms(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batches", type=int, nargs="*", default=[1, 32])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    L = args.seq_len

    head = "| Model | Params | GFLOPs / sequence (dense + scan) |"
    head += "".join(f" Latency b={b} (ms) |" for b in args.batches)
    head += " Per-sequence at b=32 (ms) |" if 32 in args.batches else ""
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for label, kind, kw, scan_flops in models(L):
        torch.manual_seed(0)
        if kind == "linoss":
            model = LinOSSCuda(**kw).cuda().freeze()  # the deployed Triton model
        else:
            from torch_linoss.models.mamba3_net import Mamba3Net

            model = Mamba3Net(**kw).cuda().eval()
        params = sum(p.numel() for p in model.parameters())
        x1 = torch.randn(1, L, N_IN, device="cuda")
        dense = dense_flops(model, x1)  # custom scan kernels are not counted here
        gflops = (dense + scan_flops) / 1e9
        row = f"| {label} | {params:,} | {gflops:.4f} ({dense / 1e9:.4f} + {scan_flops / 1e9:.4f}) |"
        per_seq = None
        for b in args.batches:
            x = torch.randn(b, L, N_IN, device="cuda")
            with torch.inference_mode():
                ms = latency_ms(lambda: model(x))
            row += f" {ms:.3f} |"
            if b == 32:
                per_seq = ms / 32
        if per_seq is not None:
            row += f" {per_seq:.4f} |"
        print(row, flush=True)


def runners(kind, kw, out_dir):
    """(name, callable factory) for every way of running the model on the GPU."""
    from torch_linoss.models import LinOSS

    def eager_ref():
        return LinOSS(**kw).cuda().eval()

    def triton():
        torch.manual_seed(0)
        return LinOSSCuda(**kw).cuda().freeze()

    def mamba():
        from torch_linoss.models.mamba3_net import Mamba3Net

        torch.manual_seed(0)
        return Mamba3Net(**kw).cuda().eval()

    def aoti(build):
        def make(x):
            from torch_linoss.export import export_package

            # Export with batch >= 2: a size-1 example would make torch.export specialize the batch dimension.
            example = x if x.shape[0] > 1 else x.expand(2, *x.shape[1:]).contiguous()
            path = export_package(build(), example, os.path.join(out_dir, f"profile_{id(x)}.pt2"))
            return torch._inductor.aoti_load_package(path)
        return make

    if kind == "linoss":
        return [
            ("FFT reference (ops)", lambda x: eager_ref()),
            ("Triton (eager)", lambda x: triton()),
            ("Triton + torch.compile", lambda x: torch.compile(triton())),
            ("Triton + CUDA graphs", lambda x: torch.compile(triton(), mode="reduce-overhead")),
            ("AOTInductor .pt2", aoti(triton)),
        ]
    return [
        ("eager", lambda x: mamba()),
        ("torch.compile + CUDA graphs", lambda x: torch.compile(mamba(), mode="reduce-overhead")),
    ]


def versions(seq_len, batches, out_dir):
    print(f"CUDA runtime {torch.version.cuda}, driver {torch.cuda.driver_version() if hasattr(torch.cuda, 'driver_version') else '-'}, "
          f"torch {torch.__version__}, triton {__import__('triton').__version__}, GPU {torch.cuda.get_device_name()}\n")
    print("| Model | Version |" + "".join(f" b={b} (ms) |" for b in batches))
    print("|---|---|" + "---|" * len(batches))
    for label, kind, kw, _ in models(seq_len):
        for name, make in runners(kind, kw, out_dir):
            cells = []
            for b in batches:
                x = torch.randn(b, seq_len, N_IN, device="cuda")
                try:
                    model = make(x)
                    with torch.inference_mode():
                        cells.append(f"{latency_ms(lambda: model(x)):.3f}")
                except Exception as e:  # report and keep going
                    cells.append(f"failed ({type(e).__name__})")
                torch._dynamo.reset()
            print(f"| {label} | {name} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    import sys

    if "--versions" in sys.argv:
        sys.argv.remove("--versions")
        ap = argparse.ArgumentParser()
        ap.add_argument("--seq_len", type=int, default=100)
        ap.add_argument("--batches", type=int, nargs="*", default=[1, 32])
        ap.add_argument("--out_dir", default="/tmp")
        a = ap.parse_args()
        torch.backends.cuda.matmul.allow_tf32 = True
        versions(a.seq_len, a.batches, a.out_dir)
    else:
        main()
