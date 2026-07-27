#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WINDOW_SIZE="${WINDOW_SIZE:-48}"
export CONTEXT_FRAMES="${CONTEXT_FRAMES:-32}"
export STRIDE="${STRIDE:-8}"

bash "$SCRIPT_DIR/job_newtphys_surprise.sh"
bash "$SCRIPT_DIR/job_physionpp_surprise.sh"
bash "$SCRIPT_DIR/job_physbench_surprise.sh"
bash "$SCRIPT_DIR/job_contphy_surprise.sh"
bash "$SCRIPT_DIR/job_intphys_surprise.sh"
bash "$SCRIPT_DIR/job_physicsiqverified_surprise.sh"
bash "$SCRIPT_DIR/job_intphys2_surprise.sh"