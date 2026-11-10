#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ROOT_DIR/scripts/job_hub.sh"

submit_geophys_job() {
    local dataset_name="$1"
    if (( $# > 1 )); then
        dataset_name="all_combined"
        if [[ -n "${REFERENCE_DATASET:-}" ]]; then
            dataset_name="all_ref-${REFERENCE_DATASET}"
        fi
    fi
    local geophys_python="$ROOT_DIR/.venv/geophys/bin/python"
    if [[ ! -x "$geophys_python" ]]; then
        echo "Error: GeoPhys virtualenv is missing. Run: bash $ROOT_DIR/scripts/geophys/setup.sh" >&2
        return 1
    fi
    local -a command=("$geophys_python" -u scripts/geophys/compute_geophys.py "$@")
    local variable option
    for variable in EVAL_MAX VARIANT REFERENCE REFERENCE_DATASET BACKBONE LAYER PREP DEVICE OUTPUT_ROOT OUTPUT_DIR; do
        if [[ -n "${!variable:-}" ]]; then
            option="${variable,,}"
            command+=("--${option//_/-}" "${!variable}")
        fi
    done
    geophys_runner_body() {
        # Combined jobs stage every source dataset on Jean-Zay.
        if declare -p REQUIRED_DATASETS &>/dev/null; then
            printf 'if [[ -n "${SCRATCH:-}" ]]; then\n'
            local dataset
            for dataset in "${REQUIRED_DATASETS[@]}"; do
                printf '    bash %q %q\n' "$ROOT_DIR/jz_assess_dataset.sh" "$dataset"
            done
            printf 'fi\n'
        fi
        printf '%q ' "${command[@]}"
        printf '\n'
    }
    job_submit "gp_$dataset_name" "$ROOT_DIR/logs/geophys" "$ROOT_DIR" geophys_runner_body
}
