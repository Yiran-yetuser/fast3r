#!/usr/bin/env bash
set -euo pipefail
cd /home/yyz/fast3r
export MPLCONFIGDIR=/tmp/fast3r-mpl
export PYTHONPATH=.:scripts
exec flock -n results/co3d_continuous_prepare_v2.lock \
 /home/yyz/miniconda3/envs/fast3r/bin/python -u scripts/prepare_co3d_continuous_v2.py
