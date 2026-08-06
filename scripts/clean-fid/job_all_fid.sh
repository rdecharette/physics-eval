#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REFERENCE_PATH="/nfs/data/workspaces/rdechare/codes/physics-eval/datasets/pisabench/real"
export REFERENCE_SET="${REFERENCE_SET:-pisabench_real}"
export FID_MAX_FRAMES=10000
# export TYPE="fid"
export TYPE="${TYPE:-kid}"

# bash "$SCRIPT_DIR/job_newtphys_fid.sh"
# bash "$SCRIPT_DIR/job_physionpp_fid.sh"
# bash "$SCRIPT_DIR/job_physbench_fid.sh"
bash "$SCRIPT_DIR/job_contphy_fid.sh"
# bash "$SCRIPT_DIR/job_intphys_fid.sh"
# bash "$SCRIPT_DIR/job_physicsiqverified_fid.sh"
# bash "$SCRIPT_DIR/job_intphys2_fid.sh"