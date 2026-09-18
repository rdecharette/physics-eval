#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DIMENSION_LIST=(
  "subject_consistency"
  "background_consistency"
  "motion_smoothness"
  "dynamic_degree"
  "aesthetic_quality"
  "imaging_quality"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_MAX="${EVAL_MAX:-500}"
export VARIANT="${VARIANT:-512p_30fps}"

for dimension in "${DEFAULT_DIMENSION_LIST[@]}"; do
    export DIMENSION_LIST="$dimension"

    bash "$SCRIPT_DIR/job_all.sh"
done