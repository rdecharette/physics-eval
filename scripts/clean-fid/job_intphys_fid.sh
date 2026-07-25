#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/slurm_job_clean_fid.sh"

FID_FRAME_STRIDE=2
FID_MAX_FRAMES_PER_VIDEO=50

submit_fid_job "intphys_impossible.txt"
submit_fid_job "intphys_possible.txt"