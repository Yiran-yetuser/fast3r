#!/usr/bin/env bash
set -Eeuo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${TASK_PYTHON:-/home/yyz/miniconda3/envs/fast3r/bin/python}"
cd "$TASK_ROOT"
exec >> results/re10k_rgb_prepare.log 2>&1
trap 'echo "[$(date -Is)] RGB PREPARATION FAILED at line $LINENO"' ERR
echo "[$(date -Is)] Auditing covered prescribed scenes; missing IDs remain incomplete"
"$TASK_PYTHON" -u scripts/prepare_re10k_rgb_from_archive.py
echo "[$(date -Is)] RGB PREPARATION PIPELINE COMPLETE (inspect full coverage before evaluation)"
