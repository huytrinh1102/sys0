"""
Merge bench_torch.json and bench_jax.json into markdown tables.

    python -m torch_linoss.benchmark_table bench_torch.json bench_jax.json
"""

import json
import sys

NAMES = {
    "jax": "JAX (original)",
    "torch_fft": "PyTorch FFT conv",
    "torch_triton": "PyTorch Triton scan",
    "torch_triton_compiled": "PyTorch Triton + compile",
}


def fmt(ms):
    return "OOM" if ms is None else f"{ms:.2f}"


def speedup(base, ms):
    return "" if base is None or ms is None else f" ({base / ms:.1f}x)"


def table(results, suite, key, rows, impls, batch=None):
    sel = [r for r in results if r["suite"] == suite and (batch is None or r.get("batch") == batch)]
    lookup = {(r[key], r["impl"]): r["ms"] for r in sel}
    head = "| " + " | ".join([key if key != "L" else "Seq. length"] + [NAMES[i] for i in impls]) + " |"
    lines = [head, "|" + "---|" * (len(impls) + 1)]
    for row in rows:
        base = lookup.get((row, "jax"))
        cells = [fmt(lookup.get((row, i))) + (speedup(base, lookup.get((row, i))) if i != "jax" else "") for i in impls]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    results = []
    meta = []
    for path in sys.argv[1:]:
        with open(path) as f:
            data = json.load(f)
        results += data["results"]
        meta.append({k: v for k, v in data.items() if k != "results"})
    print("Environment:", meta, "\n")
    impls = ["jax", "torch_fft", "torch_triton", "torch_triton_compiled"]
    lengths = sorted({r["L"] for r in results if r["suite"] == "sweep"})
    for batch in sorted({r["batch"] for r in results if r["suite"] == "sweep"}):
        print(f"### Inference time (ms), batch {batch}, EigenWorms model (2 blocks, H=128, P=64)\n")
        print(table(results, "sweep", "L", lengths, impls, batch), "\n")
    print("### Training step time (ms: forward + backward + Adam), batch 4\n")
    print(table(results, "train", "L", lengths, ["jax", "torch_fft", "torch_triton"], 4), "\n")
    datasets = []
    for r in results:
        if r["suite"] == "datasets" and r["dataset"] not in datasets:
            datasets.append(r["dataset"])
    print("### Test-set inference time (ms) per dataset at its real sequence length (batch 32)\n")
    sel = {(r["dataset"], r["impl"]): r for r in results if r["suite"] == "datasets"}
    print("| Dataset | Seq. length | Test samples | " + " | ".join(NAMES[i] for i in impls) + " |")
    print("|" + "---|" * (len(impls) + 3))
    for d in datasets:
        any_r = next(r for (dd, _), r in sel.items() if dd == d)
        base = sel.get((d, "jax"), {}).get("ms")
        cells = []
        for i in impls:
            ms = sel.get((d, i), {}).get("ms")
            cells.append(fmt(ms) + (speedup(base, ms) if i != "jax" else ""))
        print(f"| {d} | {any_r['L']} | {any_r['n']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
