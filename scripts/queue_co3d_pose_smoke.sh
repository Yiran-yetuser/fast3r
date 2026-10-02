#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/yyz/fast3r
exec >> results/co3d_pose_smoke_pipeline.log 2>&1
trap 'echo "[$(date -Is)] CO3D SMOKE FAILED line $LINENO"' ERR
TASK_PYTHON=/home/yyz/miniconda3/envs/fast3r/bin/python
export PYTHONPATH=.:scripts
export MPLCONFIGDIR=/tmp/fast3r-mpl
mkdir -p results/co3d_pose_smoke_progress
exec 9>results/co3d_pose_smoke_progress/.lock
flock -n 9
echo "[$(date -Is)] CO3D one-draw input preflight; never full Table1"
"$TASK_PYTHON" -u scripts/fast3r_hf_co3d_pose_smoke.py --dry-run
if [[ -f results/co3d_pose_draw0_seed42_20261002.json ]]; then
  "$TASK_PYTHON" -u scripts/fast3r_hf_co3d_pose_smoke.py --verify-only
  echo "[$(date -Is)] CO3D SMOKE ALREADY VERIFIED"
  exit 0
fi
echo "[$(date -Is)] Waiting for >=10240MiB free GPU; no other tasks stopped"
while true; do
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
  if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && (( FREE_MIB >= 10240 )); then
    sleep 15
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
    if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && (( FREE_MIB >= 10240 )); then break; fi
  fi
  sleep 30
done
"$TASK_PYTHON" -u scripts/fast3r_hf_co3d_pose_smoke.py
"$TASK_PYTHON" -u scripts/fast3r_hf_co3d_pose_smoke.py --verify-only
echo "[$(date -Is)] CO3D SMOKE COMPLETE; no full Table1 claim"
