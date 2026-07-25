#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-mean}"

(
  CUDA_VISIBLE_DEVICES=0 MODE="$MODE" bash "$SCRIPT_DIR/compute_newtphys_surprise.sh"
  CUDA_VISIBLE_DEVICES=0 MODE="$MODE" bash "$SCRIPT_DIR/compute_physionpp_surprise.sh"
)
(
  CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/compute_physbench_surprise.sh"
  CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/compute_contphy_surprise.sh"
) &
(
  CUDA_VISIBLE_DEVICES=2,3 MODE="$MODE" bash "$SCRIPT_DIR/compute_intphys_surprise.sh"
  CUDA_VISIBLE_DEVICES=2,3 MODE="$MODE" bash "$SCRIPT_DIR/compute_physicsiqverified_surprise.sh"
) &
CUDA_VISIBLE_DEVICES=4,5 MODE="$MODE" bash "$SCRIPT_DIR/compute_intphys2_surprise.sh" &

wait