"""
Inference (and training-step) speed of the PyTorch LinOSS models. The JAX counterpart is
`torch_linoss/benchmark_jax.py`; `torch_linoss/benchmark_table.py` merges both into markdown tables.

Two suites:
- "sweep": one model (the EigenWorms config: 2 blocks, H=128, P=64, IM) at increasing sequence lengths.
- "datasets": every paper config at its real sequence length, one pass over the test split (batch 32).

    python -m torch_linoss.benchmark --out bench_torch.json
"""

import argparse
import json

import numpy as np
import torch

from torch_linoss.bench_config import DATASETS, SWEEP_BATCHES, SWEEP_CONFIG, SWEEP_LENGTHS, model_config
from torch_linoss.models import LinOSS, LinOSSCuda



def cuda_time(fn, min_time=1.0, min_iters=5, warmup=3):
    """Median wall time (ms) of fn() measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    total = 0.0
    while len(times) < min_iters or total < min_time:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        total += times[-1] / 1000
        if len(times) >= 200:
            break
    return float(np.median(times))


def build(impl, config):
    """Return an inference callable model(X) for an implementation name."""
    torch.manual_seed(0)
    if impl == "torch_fft":
        model = LinOSS(**config).cuda().eval()
    elif impl == "torch_triton":
        model = LinOSSCuda(**config).cuda().freeze()
    elif impl == "torch_triton_compiled":
        model = torch.compile(LinOSSCuda(**config).cuda().freeze(), dynamic=False)
    else:
        raise ValueError(impl)
    return model


INFER_IMPLS = ["torch_fft", "torch_triton", "torch_triton_compiled"]
TRAIN_IMPLS = ["torch_fft", "torch_triton"]


def bench_inference(impl, config, batch, L):
    model = build(impl, config)
    X = torch.randn(batch, L, config["N"], device="cuda")

    def run():
        with torch.inference_mode():
            model(X)

    torch.cuda.reset_peak_memory_stats()
    ms = cuda_time(run)
    return ms, torch.cuda.max_memory_allocated() / 2**20


def bench_train_step(impl, config, batch, L):
    torch.manual_seed(0)
    cls = LinOSS if impl == "torch_fft" else LinOSSCuda
    model = cls(**config).cuda().train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
    X = torch.randn(batch, L, config["N"], device="cuda")
    y = torch.randint(0, config["output_dim"], (batch,), device="cuda")

    def step():
        logits = model(X)
        loss = -torch.log(torch.softmax(logits, -1).gather(1, y[:, None]) + 1e-8).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.cuda.reset_peak_memory_stats()
    ms = cuda_time(step)
    return ms, torch.cuda.max_memory_allocated() / 2**20


def guarded(fn, *args):
    try:
        return fn(*args)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench_torch.json")
    parser.add_argument("--suites", nargs="*", default=["sweep", "train", "datasets"])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    results = []

    if "sweep" in args.suites:
        for batch in SWEEP_BATCHES:
            for L in SWEEP_LENGTHS:
                for impl in INFER_IMPLS:
                    ms, mem = guarded(bench_inference, impl, SWEEP_CONFIG, batch, L)
                    results.append(dict(suite="sweep", impl=impl, batch=batch, L=L, ms=ms, mem_mb=mem))
                    print(results[-1], flush=True)
                    torch._dynamo.reset()

    if "train" in args.suites:
        for L in SWEEP_LENGTHS:
            for impl in TRAIN_IMPLS:
                ms, mem = guarded(bench_train_step, impl, SWEEP_CONFIG, 4, L)
                results.append(dict(suite="train", impl=impl, batch=4, L=L, ms=ms, mem_mb=mem))
                print(results[-1], flush=True)

    if "datasets" in args.suites:
        for name, (L, n_test, cfg_name) in DATASETS.items():
            config = model_config(cfg_name)
            for impl in INFER_IMPLS:
                model = build(impl, config)
                batches = [32] * (n_test // 32) + ([n_test % 32] if n_test % 32 else [])
                Xs = [torch.randn(b, L, config["N"], device="cuda") for b in batches]

                def run():
                    with torch.inference_mode():
                        for X in Xs:
                            model(X)

                ms = cuda_time(run)
                results.append(dict(suite="datasets", impl=impl, dataset=name, L=L, n=n_test, ms=ms))
                print(results[-1], flush=True)
                torch._dynamo.reset()

    with open(args.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(), "torch": torch.__version__, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
