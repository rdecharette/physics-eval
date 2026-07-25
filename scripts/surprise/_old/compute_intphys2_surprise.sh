#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${MODE:-mean}"
MAXFRAMES="${MAXFRAMES:-250}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-32}"
STRIDE="${STRIDE:-16}"

cd "$ROOT_DIR/WMReward"
mkdir -p "$ROOT_DIR/logs"

CUDA_VISIBLE_DEVICES=4 \
MODE="$MODE" \
MAXFRAMES="$MAXFRAMES" \
WINDOW_SIZE="$WINDOW_SIZE" \
CONTEXT_FRAMES="$CONTEXT_FRAMES" \
STRIDE="$STRIDE" \
VIDEO="$ROOT_DIR/intphys2_25fps_impossible.txt" ./test_vith.sh 2>&1 | tee "$ROOT_DIR/logs/intphys2_25fps_impossible.log" &

CUDA_VISIBLE_DEVICES=5 \
MODE="$MODE" \
MAXFRAMES="$MAXFRAMES" \
WINDOW_SIZE="$WINDOW_SIZE" \
CONTEXT_FRAMES="$CONTEXT_FRAMES" \
STRIDE="$STRIDE" \
VIDEO="$ROOT_DIR/intphys2_25fps_possible.txt" ./test_vith.sh 2>&1 | tee "$ROOT_DIR/logs/intphys2_25fps_possible.log" &

wait