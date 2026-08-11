#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_MAX="${EVAL_MAX:-500}"
export VARIANT="${VARIANT:-original}"

# export DIMENSION_LIST="background_consistency"

bash "$SCRIPT_DIR/job_physionpp_vbench.sh"
bash "$SCRIPT_DIR/job_physbench_vbench.sh"
bash "$SCRIPT_DIR/job_contphy_vbench.sh"
bash "$SCRIPT_DIR/job_intphys_vbench.sh"
bash "$SCRIPT_DIR/job_physicsiqverified_vbench.sh"
bash "$SCRIPT_DIR/job_intphys2_vbench.sh"
bash "$SCRIPT_DIR/job_newtphys_vbench.sh"
bash "$SCRIPT_DIR/job_pisabench_vbench.sh"
