# LinOSS trajectory inpainting: 100 Hz → 1 kHz

Reconstruct the 1 kHz end-effector trajectory of the Pi-2 robot from its 100 Hz samples (every 10th step),
on the simulated pick-up-mug dataset (`/home/azureuser/huy/dataset/pi2_limit/pickup_mug_augmented`,
268 episodes).

**Signals.** Right wrist + 5 right fingertips (`data/observation/end_effector`), each as position (3) and
6D rotation (6): 54 channels. The left side barely moves in this task (~1 cm).

**Task.** Keyframes are steps 0, 10, 20, …; the model fills in the 9 steps of every 10 ms gap. It takes the
linear interpolation of the keyframes, a keyframe mask and the position within the gap, and predicts a
correction to the interpolation, scaled by the training-set spread of that correction (so the target is
O(1)). Keyframes are reproduced exactly.

**Models.** Per-step MLP encoder → 4 LinOSS blocks (Triton scan, timescale init 0.5–100 Hz at 1 kHz) →
probabilistic head (mean and log-variance per step and channel).
- `--direction causal`: forward only; sees at most 10 ms ahead (the next keyframe).
- `--direction bi`: every layer runs forward and backward and sums the two.
- `--head nll`: Gaussian negative log-likelihood (i.e. KL(true ‖ predicted Gaussian) up to a constant),
  after 500 MSE warm-up steps. `--head mse`: plain MSE, for the ablation.

**Split.** 214 / 27 / 27 episodes (`out/splits.json`, seed 0).

## Usage
```bash
python data.py                      # build the cache, print data checks
python train.py --baselines         # zero-order hold / linear / cubic on the test split
python train.py --direction bi --head nll --seed 2345      # logs to wandb project linoss-inpainting
bash run_queue.sh                   # both directions × 3 seeds (HEAD=mse bash run_queue.sh for the ablation)
python plot.py                      # figures + results table in out/figures/
pytest test_inpainting.py
```

## Stage 2 (planned)
`data.load_conditions(episode)` and `InpaintNet(cond_dim=…)` / `forward(x, cond)` are the hooks for SigLIP
camera embeddings (5 cameras at 50 fps) and the 6D tactile signal (contact point + force per finger link at
250 Hz), aligned to the 1 kHz grid and concatenated to the encoder input.
