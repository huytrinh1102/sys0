#!/usr/bin/env bash
# Stage-1 runs: both directions × 3 seeds, NLL head, logged to wandb. Skips finished runs.
cd "$(dirname "$0")"
PY=/home/azureuser/miniforge3/envs/linoss/bin/python
export PYTHONUNBUFFERED=1
mkdir -p out/logs
HEAD=${HEAD:-nll}
for s in 2345 3456 4567; do
  for d in bi causal; do
    name="${d}_${HEAD}_seed_${s}"
    if [ -f "out/runs/${name}/test_metrics.json" ]; then echo "skip ${name}"; continue; fi
    $PY -u train.py --direction "$d" --head "$HEAD" --seed "$s" > "out/logs/${name}.log" 2>&1
    grep "^TEST" "out/logs/${name}.log" || tail -3 "out/logs/${name}.log"
  done
done
