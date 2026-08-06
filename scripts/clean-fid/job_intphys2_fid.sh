#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/slurm_job_clean_fid.sh"

FID_FRAME_STRIDE=5

submit_fid_job "$ROOT_DIR/intphys2_impossible.txt"
submit_fid_job "$ROOT_DIR/intphys2_possible.txt"