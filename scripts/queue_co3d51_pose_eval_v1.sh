#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/yyz/fast3r
exec >> results/co3d51_pose_eval_v1.log 2>&1
trap 'echo "[$(date -Is)] source51 pose queue/evaluation failed at line $LINENO; checkpoints preserved"' ERR
TASK_PYTHON=/home/yyz/miniconda3/envs/fast3r/bin/python
export PYTHONPATH=.:scripts PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fast3r-mpl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p results/co3d51_pose_seed42_progress_v1
exec 9>results/co3d51_pose_seed42_progress_v1/.lock
flock -n 9
echo "[$(date -Is)] Waiting for all1000 source51 inputs, full offline proof and preparation service exit; NO model yet"
while true; do
  TASK_PREP_STATE=$(systemctl --user show fast3r-co3d51-source-prepare-v1.service -p ActiveState --value)
  if [[ "$TASK_PREP_STATE" == failed ]]; then
    echo "[$(date -Is)] Preparation dependency failed; no partial-input inference allowed"
    exit 3
  fi
  if [[ -f results/co3d51_source_prepare_v1_summary_20261003.json && -f results/co3d51_source_inputs_full_verified_20261003.json && "$TASK_PREP_STATE" == inactive ]]; then
    break
  fi
  if [[ "$TASK_PREP_STATE" == inactive ]]; then
    echo "[$(date -Is)] Preparation exited without complete summary/proof; preserve checkpoints and diagnose"
    exit 3
  fi
  sleep 30
done
verify_final() {
  "$TASK_PYTHON" -u scripts/fast3r_hf_co3d51_pose_eval_v1.py --verify-only
  if [[ -f results/co3d51_pose_seed42_verified_20261003.json ]]; then
    "$TASK_PYTHON" -u scripts/verify_co3d51_candidate_pose_v1.py
  else
    "$TASK_PYTHON" -u scripts/verify_co3d51_candidate_pose_v1.py --output
  fi
}
if [[ -f results/co3d51_pose_seed42_candidate_v1.json ]]; then
  verify_final
  echo "[$(date -Is)] Existing full1000 candidate report independently verified; no model rerun"
  exit 0
fi
"$TASK_PYTHON" -u scripts/fast3r_hf_co3d51_pose_eval_v1.py --dry-run
echo "[$(date -Is)] Complete actual inputs freshly replayed; waiting for>=10240MiB free GPU"
while true; do
  TASK_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
  if [[ "$TASK_FREE_MIB" =~ ^[0-9]+$ ]] && (( TASK_FREE_MIB >= 10240 )); then
    sleep 15
    TASK_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
    if [[ "$TASK_FREE_MIB" =~ ^[0-9]+$ ]] && (( TASK_FREE_MIB >= 10240 )); then break; fi
  fi
  sleep 30
done
"$TASK_PYTHON" -u scripts/fast3r_hf_co3d51_pose_eval_v1.py
verify_final
echo "[$(date -Is)] Full1000 source51 candidate pose evaluation finished; still NOT confirmed paper Table1"
