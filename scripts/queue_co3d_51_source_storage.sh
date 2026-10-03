#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/yyz/fast3r
exec >> results/co3d_51_source_storage_v1.log 2>&1
trap 'echo "[$(date -Is)] CO3D 51 source directory budget failed at line $LINENO"' ERR
TASK_PYTHON=/home/yyz/miniconda3/envs/fast3r/bin/python
export PYTHONPATH=.:scripts PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fast3r-mpl
mkdir -p data/co3d_51_source_storage_v1
exec 9>data/co3d_51_source_storage_v1/.lock
flock -n 9
if [[ -f results/co3d_51_source_storage_v1_20261003.json ]]; then
  "$TASK_PYTHON" -u scripts/plan_co3d_51_source_storage.py --verify-only
  exit 0
fi
echo "[$(date -Is)] Source-sampler storage audit; NO raw images/GPU, frozen indices reused"
"$TASK_PYTHON" -u scripts/plan_co3d_51_source_storage.py
"$TASK_PYTHON" -u scripts/plan_co3d_51_source_storage.py --verify-only
echo "[$(date -Is)] Nominal+retry directory storage audit complete; no Table1 score"
