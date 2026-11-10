#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$ROOT_DIR/scripts/job_hub.sh"

submit_phyprobe_job() {
    local dataset_name="$1"
    local phyprobe_python="$ROOT_DIR/.venv/phyprobe/bin/python"
    if [[ ! -x "$phyprobe_python" ]]; then
        echo "Error: PhyProbe virtualenv is missing. Run: bash $ROOT_DIR/scripts/phyprobe/setup.sh" >&2
        return 1
    fi
    local -a command=("$phyprobe_python" -u scripts/phyprobe/compute_phyprobe.py "$dataset_name")
    local variable option
    for variable in EVAL_MAX VARIANT BATCH_SIZE CLIP_SEC CHECKPOINT DEVICE CACHE_DIR OUTPUT_DIR; do
        if [[ -n "${!variable:-}" ]]; then
            option="${variable,,}"
            command+=("--${option//_/-}" "${!variable}")
        fi
    done
    if [[ "${FORCE:-0}" == 1 ]]; then command+=(--force); fi
    phyprobe_runner_body() {
        printf '%q ' "${command[@]}"
        printf '\n'
    }
    job_submit "pp_$dataset_name" "$ROOT_DIR/logs/phyprobe" "$ROOT_DIR" phyprobe_runner_body
}
