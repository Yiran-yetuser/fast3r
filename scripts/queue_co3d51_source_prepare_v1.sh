#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/yyz/fast3r
exec >> results/co3d51_source_prepare_v1.log 2>&1
trap 'echo "[$(date -Is)] CO3D source51 actual preparation failed at line $LINENO; completed commits retained"' ERR
TASK_PYTHON=/home/yyz/miniconda3/envs/fast3r/bin/python
export PYTHONPATH=.:scripts PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fast3r-mpl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p results/co3d51_source_prepare_v1_progress
exec 9>results/co3d51_source_prepare_v1_progress/.lock
flock -n 9
if [[ ! -f results/co3d51_source_prepare_v1_summary_20261003.json ]]; then
  "$TASK_PYTHON" -u scripts/prepare_co3d51_source_v1.py --dry-run
  echo "[$(date -Is)] Start NEW bounded source51/1000 actual CPU preparation; no GPU/eviction"
  "$TASK_PYTHON" -u scripts/prepare_co3d51_source_v1.py
fi
"$TASK_PYTHON" -u scripts/prepare_co3d51_source_v1.py --verify-only
if [[ -f results/co3d51_source_inputs_full_verified_20261003.json ]]; then
  "$TASK_PYTHON" -u scripts/verify_co3d51_source_inputs_v1.py
else
  "$TASK_PYTHON" -u scripts/verify_co3d51_source_inputs_v1.py --output results/co3d51_source_inputs_full_verified_20261003.json
fi
echo "[$(date -Is)] Actual candidate preparation and offline input replay complete; NO Table1 score/GPU"
