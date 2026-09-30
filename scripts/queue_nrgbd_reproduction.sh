#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=/home/yyz/miniconda3/envs/fast3r/bin/python
WAIT_PID="${1:?Provide the current dataset download PID}"
cd "$ROOT_DIR"
mkdir -p results
exec >> results/nrgbd_pipeline.log 2>&1
trap 'echo "[$(date -Is)] PIPELINE FAILED at line $LINENO"' ERR
echo "[$(date -Is)] waiting for initial download PID $WAIT_PID"
while ps -p "$WAIT_PID" >/dev/null 2>&1; do sleep 30; done
if [[ $(stat -c '%s' data/neural_rgbd_data.zip.part 2>/dev/null || echo 0) != 7785287298 ]]; then
  echo "[$(date -Is)] resuming official dataset download"
  curl --fail --location --continue-at - --retry 3 --connect-timeout 20 \
    --silent --show-error --output data/neural_rgbd_data.zip.part \
    https://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip
fi
if [[ ! -f results/nrgbd_data_manifest.json ]]; then
  "$PYTHON_BIN" -u scripts/prepare_nrgbd_archive.py
fi
"$PYTHON_BIN" -u scripts/fast3r_hf_dtu_eval.py --dataset nrgbd --device cpu \
  --dry-run --max-scenes 1 --output-json results/nrgbd_dry_run.json
echo "[$(date -Is)] waiting for >=8192MiB free GPU memory"
while true; do
  FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
  if [[ "$FREE_MB" =~ ^[0-9]+$ ]] && (( FREE_MB >= 8192 )); then break; fi
  sleep 30
done
if [[ ! -f results/nrgbd_seed42_stride40.json ]]; then
  "$PYTHON_BIN" -u scripts/fast3r_hf_dtu_eval.py --dataset nrgbd --device cuda \
    --seed 42 --kf-every 40 --head-chunk-size 2 \
    --output-json results/nrgbd_seed42_stride40.json
fi
echo "[$(date -Is)] PIPELINE COMPLETE"
