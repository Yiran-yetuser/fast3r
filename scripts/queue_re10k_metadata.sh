#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${TASK_PYTHON:-/home/yyz/miniconda3/envs/fast3r/bin/python}"
cd "$TASK_ROOT"
mkdir -p results data/RealEstate10K
exec >> results/re10k_metadata_pipeline.log 2>&1
exec 9>data/RealEstate10K/.metadata.lock
flock -n 9 || exit 0
trap 'echo "[$(date -Is)] METADATA PIPELINE FAILED at line $LINENO"' ERR
echo "[$(date -Is)] Preparing prescribed RE10K test metadata only (not RGB)"
"$TASK_PYTHON" -u scripts/prepare_re10k_metadata.py
echo "[$(date -Is)] METADATA PIPELINE COMPLETE"
