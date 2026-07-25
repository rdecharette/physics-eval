#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <video_path|list.txt>" >&2
  exit 1
}

[[ $# -eq 1 ]] || usage
input="$1"

command -v ffprobe >/dev/null 2>&1 || {
  echo "Error: ffprobe not found in PATH" >&2
  exit 1
}

format_fps() {
  local rate="$1"
  awk -v r="$rate" 'BEGIN {
    n=split(r,a,"/");
    if (n==2 && a[2] != 0) {
      printf "%.6f", a[1]/a[2];
    } else {
      printf "%s", r;
    }
  }'
}

get_video_line() {
  local video_path="$1"

  if [[ ! -f "$video_path" ]]; then
    echo "$video_path: ERROR=file not found"
    return 0
  fi

  local res fps_raw fps frames nb_read_frames nb_frames duration

  res="$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height \
    -of csv=p=0:s=x "$video_path" | head -n 1 || true)"

  fps_raw="$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=avg_frame_rate \
    -of default=nokey=1:noprint_wrappers=1 "$video_path" | head -n 1 || true)"

  nb_read_frames="$(ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=nb_read_frames \
    -of default=nokey=1:noprint_wrappers=1 "$video_path" | head -n 1 || true)"

  nb_frames="$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=nb_frames \
    -of default=nokey=1:noprint_wrappers=1 "$video_path" | head -n 1 || true)"

  duration="$(ffprobe -v error \
    -show_entries format=duration \
    -of default=nokey=1:noprint_wrappers=1 "$video_path" | head -n 1 || true)"

  if [[ -z "$res" || "$res" == "N/A" ]]; then
    res="N/A"
  fi

  fps="$(format_fps "$fps_raw")"

  if [[ -n "$nb_read_frames" && "$nb_read_frames" != "N/A" ]]; then
    frames="$nb_read_frames"
  elif [[ -n "$nb_frames" && "$nb_frames" != "N/A" ]]; then
    frames="$nb_frames"
  elif [[ -n "$duration" && "$duration" != "N/A" && -n "$fps" && "$fps" != "N/A" ]]; then
    frames="$(awk -v f="$fps" -v d="$duration" 'BEGIN { printf "%d", (f*d)+0.5 }')"
  else
    frames="N/A"
  fi

  echo "$video_path: Res=${res}, F=${frames}, FPS=${fps}"
}

if [[ -f "$input" && "${input##*.}" == "txt" ]]; then
  while IFS= read -r line; do
    line="${line%%$'\r'}"
    [[ -z "$line" ]] && continue
    get_video_line "$line"
  done < "$input"
else
  get_video_line "$input"
fi
