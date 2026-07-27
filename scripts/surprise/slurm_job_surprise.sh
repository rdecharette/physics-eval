#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODE="${MODE:-mean}"
MAXFRAMES="${MAXFRAMES:-250}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-32}"
STRIDE="${STRIDE:-16}"
SLURM_PARTITION="${SLURM_PARTITION:-debug}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-2}"
SLURM_GPUS_PER_TASK="${SLURM_GPUS_PER_TASK:-1}"
SLURM_MEM="${SLURM_MEM:-24G}"

submit_surprise_job() {
    local video_list="$1"
    local dataset_name
    dataset_name="$(basename "$video_list")"
    dataset_name="${dataset_name%.txt}"

    local job_name="s_$dataset_name"
    local slurm_log_dir="$ROOT_DIR/logs/surprise"

    mkdir -p "$slurm_log_dir"

    local q_mode q_maxframes q_window_size q_context_frames q_stride q_video
    printf -v q_mode '%q' "$MODE"
    printf -v q_maxframes '%q' "$MAXFRAMES"
    printf -v q_window_size '%q' "$WINDOW_SIZE"
    printf -v q_context_frames '%q' "$CONTEXT_FRAMES"
    printf -v q_stride '%q' "$STRIDE"
    printf -v q_video '%q' "../$video_list"

    local -a sbatch_args=(
        --parsable
        --job-name "$job_name"
        --partition "$SLURM_PARTITION"
        --cpus-per-task "$SLURM_CPUS_PER_TASK"
        --gres "gpu:${SLURM_GPUS_PER_TASK}"
        --mem "$SLURM_MEM"
        --chdir "$ROOT_DIR/WMReward"
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

# Guard against stale/out-of-range GPU ordinals (e.g. CUDA_VISIBLE_DEVICES=5 on a 5-GPU node indexed 0-4).
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=\$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    first_visible="\${CUDA_VISIBLE_DEVICES%%,*}"
    if [[ -n "\${CUDA_VISIBLE_DEVICES:-}" && "\$first_visible" =~ ^[0-9]+$ && "\$gpu_count" =~ ^[0-9]+$ && \$gpu_count -gt 0 && \$first_visible -ge \$gpu_count ]]; then
        echo "Warning: CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES is out of range for this node (gpu_count=\$gpu_count). Falling back to CUDA_VISIBLE_DEVICES=0"
        export CUDA_VISIBLE_DEVICES=0
    fi
fi

echo "Effective CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-unset}"
MODE=$q_mode
MAXFRAMES=$q_maxframes
WINDOW_SIZE=$q_window_size
CONTEXT_FRAMES=$q_context_frames
STRIDE=$q_stride
VIDEO=$q_video
export MODE MAXFRAMES WINDOW_SIZE CONTEXT_FRAMES STRIDE VIDEO

./test_vith.sh
EOF
)

    printf 'Submitted %s as job %s\n' "$job_name" "$job_id"
}