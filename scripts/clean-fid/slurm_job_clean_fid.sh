#!/usr/bin/env bash
set -euo pipefail

REFERENCE_PATH="${REFERENCE_PATH-}"
REFERENCE_SET="${REFERENCE_SET-}"

if [[ -n "$REFERENCE_PATH" && -z "$REFERENCE_SET" ]] || [[ -z "$REFERENCE_PATH" && -n "$REFERENCE_SET" ]]; then
    echo "REFERENCE_PATH and REFERENCE_SET must be provided together, or both omitted." >&2
    exit 1
fi

if [[ -z "$REFERENCE_PATH" && -z "$REFERENCE_SET" ]]; then
    REFERENCE_PATH="/nfs/data/datasets/imagenet/imagenet1k/imagenet-val"
    REFERENCE_SET="imagenet1k_val"
fi

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$ROOT_DIR/scripts/job_hub.sh"

CLEAN_FID_ROOT="${CLEAN_FID_ROOT:-$ROOT_DIR/third_party/clean-fid}"
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
TMPDIR="${TMPDIR:-$ROOT_DIR/tmp}"

submit_fid_job() {
    local video_list="$1"

    if [[ -z "${REFERENCE_PATH:-}" ]]; then
        echo "REFERENCE_PATH must be set to an ImageNet image folder before submitting FID/KID jobs." >&2
        return 1
    fi

    local dataset_name
    dataset_name="$(basename "$video_list")"
    dataset_name="${dataset_name%.txt}"

    local job_name="${TYPE}_${dataset_name}"
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

        local q_root_dir q_type q_video_list q_clean_fid_root q_reference_path
        local q_reference_set q_fid_cache_root q_result_path q_fid_mode q_fid_model
        local q_fid_device q_fid_num_workers q_fid_batch_size q_fid_frame_stride
        local q_fid_max_frames_per_video q_fid_max_frames q_fid_limit_videos q_tmpdir

        printf -v q_root_dir '%q' "$ROOT_DIR"
        printf -v q_type '%q' "$TYPE"
        printf -v q_video_list '%q' "$video_list"
        printf -v q_clean_fid_root '%q' "$CLEAN_FID_ROOT"
        printf -v q_reference_path '%q' "$REFERENCE_PATH"
        printf -v q_reference_set '%q' "$REFERENCE_SET"
        printf -v q_fid_cache_root '%q' "$FID_CACHE_ROOT"
        printf -v q_result_path '%q' "$result_path"
        printf -v q_fid_mode '%q' "$FID_MODE"
        printf -v q_fid_model '%q' "$FID_MODEL"
        printf -v q_fid_device '%q' "$FID_DEVICE"
        printf -v q_fid_num_workers '%q' "$FID_NUM_WORKERS"
        printf -v q_fid_batch_size '%q' "$FID_BATCH_SIZE"
        printf -v q_fid_frame_stride '%q' "$FID_FRAME_STRIDE"
        printf -v q_fid_max_frames_per_video '%q' "$FID_MAX_FRAMES_PER_VIDEO"
        printf -v q_fid_max_frames '%q' "$FID_MAX_FRAMES"
        printf -v q_fid_limit_videos '%q' "$FID_LIMIT_VIDEOS"
        printf -v q_tmpdir '%q' "$TMPDIR"

        fid_runner_body() {
                cat <<EOF
ROOT_DIR=$q_root_dir
TYPE=$q_type
VIDEO_LIST=$q_video_list
CLEAN_FID_ROOT=$q_clean_fid_root
REFERENCE_PATH=$q_reference_path
REFERENCE_SET=$q_reference_set
FID_CACHE_ROOT=$q_fid_cache_root
RESULT_PATH=$q_result_path
FID_MODE=$q_fid_mode
FID_MODEL=$q_fid_model
FID_DEVICE=$q_fid_device
FID_NUM_WORKERS=$q_fid_num_workers
FID_BATCH_SIZE=$q_fid_batch_size
FID_FRAME_STRIDE=$q_fid_frame_stride
FID_MAX_FRAMES_PER_VIDEO=$q_fid_max_frames_per_video
FID_MAX_FRAMES=$q_fid_max_frames
FID_LIMIT_VIDEOS=$q_fid_limit_videos
TMPDIR=$q_tmpdir
export ROOT_DIR TYPE VIDEO_LIST CLEAN_FID_ROOT REFERENCE_PATH REFERENCE_SET
export FID_CACHE_ROOT RESULT_PATH FID_MODE FID_MODEL FID_DEVICE FID_NUM_WORKERS
export FID_BATCH_SIZE FID_FRAME_STRIDE FID_MAX_FRAMES_PER_VIDEO FID_MAX_FRAMES
export FID_LIMIT_VIDEOS TMPDIR

mkdir -p "\$TMPDIR"

cmd=(
    python "\$ROOT_DIR/scripts/clean-fid/compute_id_from_videos.py"
    --type "\$TYPE"
    --video-list "\$VIDEO_LIST"
    --clean-fid-root "\$CLEAN_FID_ROOT"
    --reference-path "\$REFERENCE_PATH"
    --reference-set "\$REFERENCE_SET"
    --frames-root "\$FID_CACHE_ROOT/frames"
    --lock-dir "\$FID_CACHE_ROOT/locks"
    --output-path "\$RESULT_PATH"
    --mode "\$FID_MODE"
    --model-name "\$FID_MODEL"
    --device "\$FID_DEVICE"
    --num-workers "\$FID_NUM_WORKERS"
    --batch-size "\$FID_BATCH_SIZE"
    --frame-stride "\$FID_FRAME_STRIDE"
)

if [[ -n "\$FID_MAX_FRAMES_PER_VIDEO" ]]; then
    cmd+=(--max-frames-per-video "\$FID_MAX_FRAMES_PER_VIDEO")
fi
if [[ -n "\$FID_MAX_FRAMES" ]]; then
    cmd+=(--fid-max-frames "\$FID_MAX_FRAMES")
fi
if [[ -n "\$FID_LIMIT_VIDEOS" ]]; then
    cmd+=(--limit-videos "\$FID_LIMIT_VIDEOS")
fi

printf 'Command:'
printf ' %q' "\${cmd[@]}"
printf '\n'

"\${cmd[@]}"
EOF
        }

        job_submit "$job_name" "$slurm_log_dir" "$ROOT_DIR" fid_runner_body
}