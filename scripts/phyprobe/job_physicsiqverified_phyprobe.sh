#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/phyprobe/slurm_job_phyprobe.sh"

export REQUIRED_DATASET="physics-iq-verified"


submit_phyprobe_job "physics-iq-verified"
