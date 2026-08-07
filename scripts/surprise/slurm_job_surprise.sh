#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODE="${MODE:-mean}"
MAXFRAMES="${MAXFRAMES:-250}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-32}"
STRIDE="${STRIDE:-16}"

# shellcheck source=/dev/null
source "$ROOT_DIR/scripts/job_hub.sh"

submit_surprise_job() {
    local video_list="$1"
    local dataset_name
    dataset_name="$(basename "$video_list")"
    dataset_name="${dataset_name%.txt}"

    local job_name="s_$dataset_name"
    local slurm_log_dir="$ROOT_DIR/logs/surprise"

    local q_mode q_maxframes q_window_size q_context_frames q_stride q_video
    printf -v q_mode '%q' "$MODE"
    printf -v q_maxframes '%q' "$MAXFRAMES"
    printf -v q_window_size '%q' "$WINDOW_SIZE"
    printf -v q_context_frames '%q' "$CONTEXT_FRAMES"
    printf -v q_stride '%q' "$STRIDE"
    printf -v q_video '%q' "$video_list"

    surprise_runner_body() {
        cat <<EOF
MODE=$q_mode
MAXFRAMES=$q_maxframes
WINDOW_SIZE=$q_window_size
CONTEXT_FRAMES=$q_context_frames
STRIDE=$q_stride
VIDEO=$q_video
export MODE MAXFRAMES WINDOW_SIZE CONTEXT_FRAMES STRIDE VIDEO

./test_vith.sh
EOF
    }

    job_submit "$job_name" "$slurm_log_dir" "$ROOT_DIR/third_party/WMReward" surprise_runner_body
}
