#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/phyprobe/slurm_job_phyprobe.sh"

export REQUIRED_DATASET="intphys"

# submit_phyprobe_job "intphys_impossible"
submit_phyprobe_job "intphys_possible"
