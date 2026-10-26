#!/usr/bin/env bash
# Train on every paper config with all 5 seeds, then summarize against the JAX results in outputs/.
#
# LinOSS (starts from the JAX initial weights):
#   [DATASETS="ppg EigenWorms"] bash torch_linoss/run_all.sh [backend] [extra train.py args...]
# Mamba-3 baseline (same data, splits and training loop):
#   MODEL=mamba3 bash torch_linoss/run_all.sh
set -euo pipefail
BACKEND=${1:-cuda}
shift || true
PY=${PY:-python}
MODEL=${MODEL:-linoss}
export PYTHONUNBUFFERED=1  # progress shows up in the logs as it happens
mkdir -p outputs_torch/logs
if [ "$MODEL" = "mamba3" ]; then
  MODEL_ARGS=(--model mamba3)
  TAG=mamba3
else
  MODEL_ARGS=(--model linoss --backend "$BACKEND" --init_from_jax)
  TAG=$BACKEND
fi
DATASETS=${DATASETS:-"Heartbeat SelfRegulationSCP1 SelfRegulationSCP2 EthanolConcentration MotorImagery EigenWorms ppg"}
for DATASET in $DATASETS; do
  echo "=== ${DATASET} $(date)"
  "$PY" -m torch_linoss.train --dataset_name "$DATASET" "${MODEL_ARGS[@]}" --overwrite "$@" \
    > "outputs_torch/logs/${DATASET}_${TAG}.log" 2>&1
  grep -E "Final test metric" "outputs_torch/logs/${DATASET}_${TAG}.log"
done
"$PY" -m torch_linoss.summarize_results
