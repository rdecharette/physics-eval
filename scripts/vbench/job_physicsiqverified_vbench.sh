#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/vbench/slurm_job_vench.sh"

export REQUIRED_DATASET="physics-iq-verified"


submit_vbench_job "physics-iq-verified"
