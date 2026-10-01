#!/usr/bin/env bash
set -euo pipefail
cd /home/yyz/fast3r
exec >> results/co3d_seen41_storage.log 2>&1
date --iso-8601=seconds
exec /home/yyz/miniconda3/envs/fast3r/bin/python -u scripts/plan_co3d_seen41_storage.py
