#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yyz/miniconda3/envs/fast3r/bin/python}"
cd "$ROOT_DIR"
mkdir -p results
exec >> results/7scenes_pipeline.log 2>&1
trap 'echo "[$(date -Is)] PIPELINE FAILED at line $LINENO"' ERR
echo "[$(date -Is)] preparing official 7-Scenes test trajectories at stride20"
"$PYTHON_BIN" -u scripts/prepare_7scenes_sparse.py
"$PYTHON_BIN" -u scripts/fast3r_hf_dtu_eval.py --dataset 7scenes --device cpu \
  --dry-run --max-scenes 1 --output-json results/7scenes_dry_run.json
echo "[$(date -Is)] waiting for previous NRGBD ablation and >=10240MiB free GPU memory"
while true; do
  FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
  if ! systemctl --user is-active --quiet fast3r-nrgbd-paired.service && \
     [[ "$FREE_MB" =~ ^[0-9]+$ ]] && (( FREE_MB >= 10240 )); then break; fi
  sleep 30
done
if [[ ! -f results/7scenes_paired_seed42_stride20.json ]]; then
  "$PYTHON_BIN" -u scripts/fast3r_hf_dtu_eval.py --dataset 7scenes --head both --device cuda \
    --seed 42 --kf-every 20 --head-chunk-size 2 \
    --output-json results/7scenes_paired_seed42_stride20.json
fi
echo "[$(date -Is)] PIPELINE COMPLETE"
