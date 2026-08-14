#!/usr/bin/env bash
# Milestone 2.2 (experiments.txt): run ablation Arms B and C sequentially,
# one seed each (matching run_m2_conflict.py's SCREENING_SEED base model).
set -euo pipefail
cd "$(dirname "$0")/../.."

SEED=31874
ARMS=(B C)

for arm in "${ARMS[@]}"; do
  echo "=== m2 arm=${arm} seed=${SEED} starting $(date -u +%FT%TZ) ==="
  .venv/bin/python scripts/hydrocore_v5/run_m2_arm.py --arm "${arm}" --seed "${SEED}"
  echo "=== m2 arm=${arm} seed=${SEED} done $(date -u +%FT%TZ) ==="
done
echo "ALL M2 ARM RUNS COMPLETE"
