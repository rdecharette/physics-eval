#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/geophys/slurm_job_geophys.sh"

export REQUIRED_DATASET="NewtPhys"

submit_geophys_job "NewtPhys"
