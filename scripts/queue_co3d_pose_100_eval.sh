#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/yyz/fast3r
exec >> results/co3d_pose_100_eval_v3.log 2>&1
trap 'echo "[$(date -Is)] CO3D 100-request pose evaluation failed at line $LINENO"' ERR
TASK_PYTHON=/home/yyz/miniconda3/envs/fast3r/bin/python
export PYTHONPATH=.:scripts MPLCONFIGDIR=/tmp/fast3r-mpl
mkdir -p results/co3d_pose_100_seed42_progress_v3
exec 9>results/co3d_pose_100_seed42_progress_v3/.lock
flock -n 9
echo "[$(date -Is)] Candidate 100-request CO3D pose evaluation; never claim Table 1"
"$TASK_PYTHON" -u scripts/fast3r_hf_co3d_100_pose_eval.py --dry-run
if [[ -f results/co3d_pose_100_seed42_adaptation_v3.json ]]; then
  "$TASK_PYTHON" -u scripts/fast3r_hf_co3d_100_pose_eval.py --verify-only
  echo "[$(date -Is)] Existing final candidate report verified"
  exit 0
fi
while true; do
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
  if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && (( FREE_MIB >= 10240 )); then
    sleep 15
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
    if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && (( FREE_MIB >= 10240 )); then break; fi
  fi
  echo "[$(date -Is)] Waiting for >=10240 MiB free GPU; no other jobs stopped"
  sleep 30
done
"$TASK_PYTHON" -u scripts/fast3r_hf_co3d_100_pose_eval.py
"$TASK_PYTHON" -u scripts/fast3r_hf_co3d_100_pose_eval.py --verify-only
echo "[$(date -Is)] Candidate evaluation complete; no formal Table 1 claim"
