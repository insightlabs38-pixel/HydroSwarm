#!/usr/bin/env bash
# Milestone 1.3/1.4 (experiments.txt): run the full arm x seed matrix
# sequentially. Two seeds for rapid screening of all three arms
# (experiments.txt seed policy); a third seed is added later only for
# whichever arm the promotion rule provisionally selects.
set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS=(31874 20260814)
ARMS=(A B C)

for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== arm=${arm} seed=${seed} starting $(date -u +%FT%TZ) ==="
    .venv/bin/python scripts/hydrocore_v5/run_m1_arm.py --arm "${arm}" --seed "${seed}"
    echo "=== arm=${arm} seed=${seed} done $(date -u +%FT%TZ) ==="
  done
done
echo "ALL M1 ARM/SEED RUNS COMPLETE"
