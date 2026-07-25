#!/usr/bin/env bash
set -euo pipefail

REFERENCE_PATH="${REFERENCE_PATH:-/nfs/data/datasets/imagenet/imagenet1k/imagenet-val}"
REFERENCE_SET="${REFERENCE_SET:-imagenet1k_val}"

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CLEAN_FID_ROOT="${CLEAN_FID_ROOT:-$ROOT_DIR/clean-fid}"
# REFERENCE_SET="${REFERENCE_SET:-imagenet_custom}"
FID_MODE="${FID_MODE:-clean}"
FID_MODEL="${FID_MODEL:-inception_v3}"
FID_DEVICE="${FID_DEVICE:-cuda}"
FID_FRAME_STRIDE="${FID_FRAME_STRIDE:-25}"
FID_MAX_FRAMES_PER_VIDEO="${FID_MAX_FRAMES_PER_VIDEO:-32}"
FID_MAX_FRAMES="${FID_MAX_FRAMES:-}"
FID_NUM_WORKERS="${FID_NUM_WORKERS:-16}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-32}"
FID_LIMIT_VIDEOS="${FID_LIMIT_VIDEOS:-}"
FID_CACHE_ROOT="${FID_CACHE_ROOT:-$ROOT_DIR/cache/clean-fid}"
TYPE="${TYPE:-fid}"

SLURM_PARTITION="${SLURM_PARTITION:-debug}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-4}"
SLURM_GPUS_PER_TASK="${SLURM_GPUS_PER_TASK:-1}"
SLURM_MEM="${SLURM_MEM:-24G}"

export TMPDIR=./tmp
mkdir -p "$TMPDIR"

submit_fid_job() {
    local video_list="$1"

    if [[ -z "${REFERENCE_PATH:-}" ]]; then
        echo "REFERENCE_PATH must be set to an ImageNet image folder before submitting FID jobs." >&2
        return 1
    fi

    local dataset_name
    dataset_name="$(basename "$video_list")"
    dataset_name="${dataset_name%.txt}"

    local job_name="fid_${dataset_name}"
    local slurm_log_dir="$ROOT_DIR/logs/clean-fid/slurm"
    local result_dir
    if [[ -n "$FID_MAX_FRAMES" ]]; then
        result_dir="$ROOT_DIR/output/cleanfid/${TYPE}/$FID_MAX_FRAMES/$REFERENCE_SET"
    else
        result_dir="$ROOT_DIR/output/cleanfid/${TYPE}/all/$REFERENCE_SET"
    fi
    local result_path="$result_dir/${dataset_name}.json"

    if [[ -f "$result_path" ]]; then
        printf 'Skipping %s; result already exists at %s\n' "$job_name" "$result_path"
        return 0
    fi

    mkdir -p "$slurm_log_dir" "$result_dir"

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

cmd=(
  python "$ROOT_DIR/scripts/clean-fid/compute_id_from_videos.py"
  --type "$TYPE"
  --video-list "$ROOT_DIR/$video_list"
  --clean-fid-root "$CLEAN_FID_ROOT"
  --reference-path "$REFERENCE_PATH"
  --reference-set "$REFERENCE_SET"
  --frames-root "$FID_CACHE_ROOT/frames"
  --lock-dir "$FID_CACHE_ROOT/locks"
  --output-path "$result_path"
  --mode "$FID_MODE"
  --model-name "$FID_MODEL"
  --device "$FID_DEVICE"
  --num-workers "$FID_NUM_WORKERS"
  --batch-size "$FID_BATCH_SIZE"
  --frame-stride "$FID_FRAME_STRIDE"
)

if [[ -n "$FID_MAX_FRAMES_PER_VIDEO" ]]; then
  cmd+=(--max-frames-per-video "$FID_MAX_FRAMES_PER_VIDEO")
fi
if [[ -n "$FID_MAX_FRAMES" ]]; then
    cmd+=(--fid-max-frames "$FID_MAX_FRAMES")
fi
if [[ -n "$FID_LIMIT_VIDEOS" ]]; then
  cmd+=(--limit-videos "$FID_LIMIT_VIDEOS")
fi

# Print a shell-escaped command line for exact reproducibility in logs.
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"\${cmd[@]}"
EOF
)

    printf 'Submitted %s as job %s\n' "$job_name" "$job_id"
}