#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# shellcheck source=/dev/null
source "$ROOT_DIR/scripts/job_hub.sh"

submit_vbench_job() {
    local dataset_name="$1"
    local slurm_log_dir="$ROOT_DIR/logs/vbench"

    local job_name="vb_$dataset_name"

    local q_dataset_name
    printf -v q_dataset_name '%q' "$dataset_name"

    local -a extra_args=()
    if [[ -n "${EVAL_MAX:-}" ]]; then
        extra_args+=(--eval-max "$EVAL_MAX")
    fi
    if [[ -n "${FORMAT:-}" ]]; then
        extra_args+=(--format "$FORMAT")
    fi
    
    vbench_runner_body() {
        cat <<EOF
DATASET_NAME=$q_dataset_name
export DATASET_NAME

python -u scripts/vbench/compute_vbench.py \
    "\$DATASET_NAME" ${extra_args[@]}
EOF
    }

    job_submit "$job_name" "$slurm_log_dir" "$ROOT_DIR" vbench_runner_body
}