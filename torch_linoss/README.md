# torch_linoss: LinOSS in PyTorch

A PyTorch port of the JAX/Equinox LinOSS in this repo. It has the same math, the same initialization and Equinox's BatchNorm semantics, so starting from the same weights reproduces the JAX losses and gradients.

```
torch_linoss/
  ops/          pure-PyTorch ops: discretization, length-L kernel, FFT causal conv (causal-conv1d interface), reference recurrence
  cuda_ops/     Triton kernels for the recurrence, registered as the custom op linoss::scan (fwd + bwd), with a CPU fallback
  models/       LinOSS (FFT ops) and LinOSSCuda (Triton): same parameter names, weights move freely between them
  data.py       data pipeline that follows data_dir/datasets.py (same 70/15/15 splits per seed)
  train.py      training loop of the paper (Adam, eval every 1000 steps, early stopping after 10 evals without improvement)
  export.py     checkpoint -> LinOSSCuda -> torch.export -> AOTInductor .pt2
  convert_eqx.py / export_npz.py   run in the JAX env: Equinox weights / processed data -> numpy
  benchmark*.py speed comparison against the original JAX model
```

## Setup and usage

```bash
# JAX env (original preprocessing), from the repo root
python data_dir/process_uea.py && python data_dir/process_ppg.py
python -m torch_linoss.export_npz                             # data + JAX split permutations -> .npz
python -m torch_linoss.convert_eqx --dataset_name EigenWorms  # JAX initial weights per seed

# PyTorch env (torch >= 2.6 with Triton)
pytest tests/test_torch_linoss.py
python -m torch_linoss.train --dataset_name EigenWorms --backend cuda --init_from_jax
bash torch_linoss/run_all.sh cuda                             # every paper config, 5 seeds
CUDA_HOME=... python -m torch_linoss.export --checkpoint <run>/best.pt --dataset_name EigenWorms
```

`export.py` needs the CUDA toolkit headers for AOTInductor. Without a system toolkit, the pip packages `nvidia-cuda-nvcc-cu12` and `nvidia-cuda-cccl-cu12` provide them: point `CUDA_HOME` at a folder whose `include/` merges the `include/` folders of `nvidia/cuda_runtime`, `nvidia/cuda_nvcc` and `nvidia/cuda_cccl`, and whose `lib64/` has `libcudart.so`.

## Things that matter for matching the JAX results

- **BatchNorm.** Equinox 0.11's BatchNorm normalizes with its *running* statistics even in training. These start from the first batch and then move as `0.01 * batch + 0.99 * running`. torch's `BatchNorm1d` normalizes with batch statistics, and with LinOSS's large activations that makes training stall after the first Adam step. `models/common.py:BatchNorm` reimplements the Equinox behaviour.
- **Loss.** The loss is `-log(softmax + 1e-8)` as in JAX. The 1e-8 caps the per-sample loss and its gradient; plain `cross_entropy` behaves differently on unnormalized inputs such as EigenWorms.
- **Initialization sensitivity, inherited from the original.** A ~ U[0,1) is passed through relu. If some A starts below about 1e-3, one Adam step at lr 1e-3 can make relu(A) = 0. That oscillator then becomes a double integrator, and over long sequences the logits explode. Roughly 1 in 4 random initializations of the EigenWorms model (128 oscillators) has an A below 2e-3. `--init_from_jax` starts from the exact JAX initial weights of each seed. That avoids the problem for the paper's EigenWorms seeds, but not for PPG seed 6789, whose initial A = 0.0002 diverged in PyTorch.
- **`--A_eps` (A = relu(A_diag) + A_eps).** This optional floor fixes the problem above; the default 0 is the original model. A floor of 1e-4 is too small, because the oscillator's DC gain (~1/A) still saturates the logits. 1e-3 is the smallest value that rescued both known failures (EigenWorms from torch's seed-2345 init, PPG seed 6789). The floor also caps the slowest memory: with IM the decay time constant is about 2 / (dt^2 * A_eps), roughly 8k steps for 1e-3 and 800 steps for 1e-2.
- **Dropout.** The JAX training loop broadcasts the dropout key over the batch, so all samples in a batch share one mask. `--dropout_mask batch` (the default) does the same, and `sample` uses torch's per-sample masks.
