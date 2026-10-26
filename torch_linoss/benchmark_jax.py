"""
The same benchmark suites as `torch_linoss/benchmark.py`, for the original JAX/Equinox LinOSS
(`models/LinOSS.py`). Run it in the JAX environment from the repository root:

    PYTHONPATH=. python -m torch_linoss.benchmark_jax --out bench_jax.json
"""

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from models.LinOSS import LinOSS
from torch_linoss.bench_config import DATASETS, SWEEP_BATCHES, SWEEP_CONFIG, SWEEP_LENGTHS, model_config


def make_model(config):
    model = LinOSS(
        config["num_blocks"],
        config["N"],
        config["ssm_size"],
        config["H"],
        config["output_dim"],
        config["classification"],
        config.get("output_step", 1),
        config.get("discretization", "IM"),
        key=jr.PRNGKey(0),
    )
    return model, eqx.nn.State(model)


@eqx.filter_jit
def infer(model, X, state, key):
    # As `calc_output` in train.py, for a stateful, nondeterministic model.
    out, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None))(X, state, key)
    return out


def wall_time(fn, min_time=1.0, min_iters=5, warmup=3):
    """Median wall time (ms); fn must block until its result is ready."""
    for _ in range(warmup):
        fn()
    times = []
    total = 0.0
    while len(times) < min_iters or total < min_time:
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
        total += times[-1] / 1000
        if len(times) >= 200:
            break
    return float(np.median(times))


def bench_inference(config, batch, L):
    model, state = make_model(config)
    model = eqx.tree_inference(model, value=True)
    X = jr.normal(jr.PRNGKey(1), (batch, L, config["N"]))
    key = jr.PRNGKey(2)
    return wall_time(lambda: infer(model, X, state, key).block_until_ready())


def bench_train_step(config, batch, L):
    model, state = make_model(config)
    opt = optax.adam(1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    X = jr.normal(jr.PRNGKey(1), (batch, L, config["N"]))
    y = jax.nn.one_hot(jnp.zeros(batch, dtype=jnp.int32), config["output_dim"])

    @eqx.filter_jit
    def step(model, state, opt_state, key):
        def loss_fn(diff, static):
            m = eqx.combine(diff, static)
            out, st = jax.vmap(m, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None))(X, state, key)
            return jnp.mean(-jnp.sum(y * jnp.log(out + 1e-8), axis=1)), st

        diff, static = eqx.partition(model, eqx.is_inexact_array)
        (loss, st), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(diff, static)
        updates, opt_state = opt.update(grads, opt_state)
        return eqx.apply_updates(model, updates), st, opt_state, loss

    carry = [model, state, opt_state]
    key = jr.PRNGKey(2)

    def run():
        carry[0], carry[1], carry[2], loss = step(carry[0], carry[1], carry[2], key)
        loss.block_until_ready()

    return wall_time(run)


def guarded(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # XLA raises RESOURCE_EXHAUSTED as a generic runtime error
        if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
            return None
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench_jax.json")
    parser.add_argument("--suites", nargs="*", default=["sweep", "train", "datasets"])
    args = parser.parse_args()
    jax_config = dict(SWEEP_CONFIG, discretization="IM")
    results = []

    if "sweep" in args.suites:
        for batch in SWEEP_BATCHES:
            for L in SWEEP_LENGTHS:
                ms = guarded(bench_inference, jax_config, batch, L)
                results.append(dict(suite="sweep", impl="jax", batch=batch, L=L, ms=ms))
                print(results[-1], flush=True)

    if "train" in args.suites:
        for L in SWEEP_LENGTHS:
            ms = guarded(bench_train_step, jax_config, 4, L)
            results.append(dict(suite="train", impl="jax", batch=4, L=L, ms=ms))
            print(results[-1], flush=True)

    if "datasets" in args.suites:
        for name, (L, n_test, cfg_name) in DATASETS.items():
            config = model_config(cfg_name)
            model, state = make_model(config)
            model = eqx.tree_inference(model, value=True)
            batches = [32] * (n_test // 32) + ([n_test % 32] if n_test % 32 else [])
            Xs = [jr.normal(jr.PRNGKey(i), (b, L, config["N"])) for i, b in enumerate(batches)]
            key = jr.PRNGKey(2)

            def run():
                for X in Xs:
                    out = infer(model, X, state, key)
                out.block_until_ready()

            ms = guarded(wall_time, run)
            results.append(dict(suite="datasets", impl="jax", dataset=name, L=L, n=n_test, ms=ms))
            print(results[-1], flush=True)

    with open(args.out, "w") as f:
        json.dump({"device": str(jax.devices()[0]), "jax": jax.__version__, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
