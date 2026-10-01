#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${TASK_PYTHON:-/home/yyz/miniconda3/envs/fast3r/bin/python}"
cd "$TASK_ROOT"
mkdir -p results data/RealEstate10K
exec >> results/re10k_rgb_archive.log 2>&1
exec 9>data/RealEstate10K/.rgb-archive.lock
flock -n 9 || exit 0
trap 'echo "[$(date -Is)] RGB ARCHIVE PIPELINE FAILED at line $LINENO"' ERR
echo "[$(date -Is)] Downloading author test-only mirror (not claiming ready/equivalent)"
"$TASK_PYTHON" -u scripts/download_re10k_rgb_archive.py
echo "[$(date -Is)] RGB ARCHIVE PIPELINE COMPLETE"
