#!/usr/bin/env bash
set -euo pipefail

# Run from this repository regardless of the caller's working directory.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Override VIDEO to score a different file, for example:
# VIDEO=/path/to/video.mp4 ./test_vith.sh
VIDEO="${VIDEO}"
MAXFRAMES="${MAXFRAMES:-150}"
WINDOW_SIZE="${WINDOW_SIZE:-16}"
STRIDE="${STRIDE:-8}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-8}"
MODE="${MODE:-mean}"
EVAL_MAX="${EVAL_MAX:--1}"
EXTRA_ARGS=()
if [[ "${DIRECT_PATHS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--direct_paths)
fi
if [[ -n "${OUTPUT_PATH:-}" ]]; then
  EXTRA_ARGS+=(--output_path "$OUTPUT_PATH")
fi

command -v python >/dev/null 2>&1 || {
  echo "Error: python was not found in PATH. Activate the target environment before running." >&2
  exit 1
}

# [[ "$MODE" == "mean" || "$MODE" == "max" ]] || {
#   echo "Error: MODE must be one of: mean, max (got: $MODE)" >&2
#   exit 1
# }

[[ -f "$VIDEO" ]] || {
  echo "Error: input video not found: $VIDEO" >&2
  exit 1
}

PYTHONUNBUFFERED=1 python -u compute_wmreward.py \
  --video_path "$VIDEO" \
  --model vith \
  --window_size "$WINDOW_SIZE" \
  --context_frames "$CONTEXT_FRAMES" \
  --max_frames "$MAXFRAMES" \
  --eval_max "$EVAL_MAX" \
  --stride "$STRIDE" \
  --mode "$MODE" \
  "${EXTRA_ARGS[@]}"
