#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/surprise/slurm_job_surprise.sh"

submit_surprise_job "$ROOT_DIR/contphy.txt"