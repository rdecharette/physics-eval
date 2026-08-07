#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/vbench/slurm_job_vench.sh"

export REQUIRED_DATASET="PhysBench"


submit_vbench_job "PhysBench"
