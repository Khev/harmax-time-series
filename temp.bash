#!/usr/bin/env bash
set -euo pipefail

# List of datasets to featurize & run
datasets=(
  "SyntheticControl"
  "ShapeletSim"
  "TwoPatterns"
  "Coffee"
  "OliveOil"
  "GunPoint"
  "ECG5000"
  "ItalyPowerDemand"
  "Wafer"
  "FordA"
)

# Optional: tweak these if you want non‐default args
SCRIPT="temp.py"
COMMON_ARGS="--featurize --epochs 100 --batch 32 --lr 3e-3 --wd 0"

for ds in "${datasets[@]}"; do
  echo "===== Running featurized on ${ds} ====="
  python "${SCRIPT}" --dataset "${ds}" ${COMMON_ARGS}
done

