#!/usr/bin/env bash
set -euo pipefail
cd /home/yyz/fast3r
exec >> results/re10k_missing_candidates_stream.log 2>&1
date --iso-8601=seconds
exec /home/yyz/miniconda3/envs/fast3r/bin/python -u scripts/stream_re10k_missing_candidates.py \
  --full-scan --output-json results/re10k_missing_candidates_full_source.json
