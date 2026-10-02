#!/usr/bin/env bash
set -euo pipefail
cd /home/yyz/fast3r
exec >> results/co3d_camera_metadata_audit.log 2>&1
date --iso-8601=seconds
exec /home/yyz/miniconda3/envs/fast3r/bin/python -u scripts/audit_co3d_camera_metadata.py
