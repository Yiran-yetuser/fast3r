#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/yyz/fast3r
exec >> results/co3d_landscape_input_diagnostic_v1.log 2>&1
trap 'echo "[$(date -Is)] CO3D landscape input diagnostic failed at line $LINENO"' ERR
TASK_PYTHON=/home/yyz/miniconda3/envs/fast3r/bin/python
export PYTHONPATH=.:scripts MPLCONFIGDIR=/tmp/fast3r-mpl
mkdir -p results/co3d_landscape_input_diagnostic_v1_progress
exec 9>results/co3d_landscape_input_diagnostic_v1_progress/.lock
flock -n 9
"$TASK_PYTHON" -u scripts/diagnose_co3d_landscape_inputs.py --dry-run
if [[ -f results/co3d_landscape_input_diagnostic_v1_20261003.json ]]; then
  "$TASK_PYTHON" -u scripts/diagnose_co3d_landscape_inputs.py --verify-only
  exit 0
fi
while true; do
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
  if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && (( FREE_MIB >= 10240 )); then
    sleep 15
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
    if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && (( FREE_MIB >= 10240 )); then break; fi
  fi
  echo "[$(date -Is)] Waiting for >=10240MiB; other GPU jobs untouched"
  sleep 30
done
"$TASK_PYTHON" -u scripts/diagnose_co3d_landscape_inputs.py
"$TASK_PYTHON" -u scripts/diagnose_co3d_landscape_inputs.py --verify-only
echo "[$(date -Is)] Fixed-frame input diagnostic complete; historical inputs/results untouched"
