#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_MAX="${EVAL_MAX:-500}"
export VARIANT="${VARIANT:-512p_30fps}"
for dataset in contphy intphys intphys2 newtphys physbench physicsiqverified physionpp pisabench; do
    bash "$SCRIPT_DIR/job_${dataset}_phyprobe.sh"
done
