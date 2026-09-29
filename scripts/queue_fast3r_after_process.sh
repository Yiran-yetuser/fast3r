#!/usr/bin/env bash
set -Eeuo pipefail

# Queue the Fast3R DTU evaluation behind another GPU job.  This helper is
# intentionally conservative: it only starts after the exact PID disappears,
# waits for enough VRAM, and refuses to overwrite a completed result.

WAIT_PID="${1:?usage: $0 PID [output_json]}"
OUTPUT_JSON="${2:-demo_outputs/paper_eval/dtu_all.json}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${FAST3R_PYTHON:-/home/yyz/miniconda3/envs/fast3r/bin/python}"
LOG_DIR="$(dirname "$ROOT_DIR/$OUTPUT_JSON")"
LOG_FILE="$LOG_DIR/dtu_all.log"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

echo "[$(date -Is)] waiting for PID $WAIT_PID to finish" >> "$LOG_FILE"
while ps -p "$WAIT_PID" >/dev/null 2>&1; do
  sleep 60
done
echo "[$(date -Is)] PID $WAIT_PID finished; waiting for GPU memory" >> "$LOG_FILE"

# The desktop compositor still uses some VRAM.  Require at least 8 GiB free so
# the ViT-Large inference does not start alongside another compute workload.
while true; do
  FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
  if [[ "$FREE_MB" =~ ^[0-9]+$ ]] && (( FREE_MB >= 8192 )); then
    break
  fi
  echo "[$(date -Is)] only ${FREE_MB:-unknown} MiB free; retrying" >> "$LOG_FILE"
  sleep 60
done

if [[ -s "$OUTPUT_JSON" ]]; then
  echo "[$(date -Is)] result already exists at $OUTPUT_JSON; nothing to do" >> "$LOG_FILE"
  exit 0
fi

if pgrep -af 'fast3r_hf_dtu_eval.py' >/dev/null 2>&1; then
  echo "[$(date -Is)] Fast3R evaluator already running; nothing to do" >> "$LOG_FILE"
  exit 0
fi

echo "[$(date -Is)] starting Fast3R DTU evaluation" >> "$LOG_FILE"
exec "$PYTHON_BIN" scripts/fast3r_hf_dtu_eval.py \
  --device cuda \
  --output-json "$OUTPUT_JSON" \
  >> "$LOG_FILE" 2>&1
