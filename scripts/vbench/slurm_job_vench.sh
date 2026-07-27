#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SLURM_PARTITION="${SLURM_PARTITION:-debug}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-2}"
SLURM_GPUS_PER_TASK="${SLURM_GPUS_PER_TASK:-1}"
SLURM_MEM="${SLURM_MEM:-24G}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda)}"

submit_vbench_job() {
    local dataset_name="$1"
    local slurm_log_dir="$ROOT_DIR/logs/vbench"

    local job_name="vb_$dataset_name"

    mkdir -p "$slurm_log_dir"

    local q_dataset_name q_conda_exe
    printf -v q_dataset_name '%q' "$dataset_name"
    printf -v q_conda_exe '%q' "$CONDA_EXE"

    local -a extra_args=()
    if [[ -n "${EVAL_MAX:-}" ]]; then
        extra_args=(--eval-max "$EVAL_MAX")
    fi
    if [[ -n "${FORMAT:-}" ]]; then
        extra_args=(--format "$FORMAT")
    fi

    local -a sbatch_args=(
        --parsable
        --job-name "$job_name"
        --partition "$SLURM_PARTITION"
        --cpus-per-task "$SLURM_CPUS_PER_TASK"
        --gres "gpu:${SLURM_GPUS_PER_TASK}"
        --mem "$SLURM_MEM"
        --chdir "$ROOT_DIR"
        --output "$slurm_log_dir/%j_%x.log"
        --error "$slurm_log_dir/%j_%x.log"
    )

    if [[ -n "${SLURM_TIME:-}" ]]; then
        sbatch_args+=(--time "$SLURM_TIME")
    fi
    if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
        sbatch_args+=(--account "$SLURM_ACCOUNT")
    fi
    if [[ -n "${SLURM_QOS:-}" ]]; then
        sbatch_args+=(--qos "$SLURM_QOS")
    fi

    local job_id
    job_id=$(sbatch "${sbatch_args[@]}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node: \$(hostname)"
echo "Start: \$(date)"
echo "SLURM_JOB_GPUS: \${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-unset}"

echo "Effective CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-unset}"

cd "$ROOT_DIR"
"$q_conda_exe" run --no-capture-output -n wmreward-score python scripts/vbench/compute_vbench.py \
    "$q_dataset_name" ${extra_args[@]}
EOF
)

    printf 'Submitted %s as job %s\n' "$job_name" "$job_id"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ $# -ne 1 ]]; then
        echo "Usage: $0 <dataset_name>" >&2
        exit 1
    fi

    submit_vbench_job "$1"
fi