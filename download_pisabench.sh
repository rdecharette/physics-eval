#!/usr/bin/env sh

set -e

ZIP_URL="https://huggingface.co/datasets/nyu-visionx/pisa-experiments/blob/main/pisabench/real.zip"
TARGET_DIR="datasets/pisabench"
ZIP_PATH="pisabench.zip"

mkdir -p "$TARGET_DIR"
TARGET_DIR_ABS=$(cd "$TARGET_DIR" && pwd)
TARGET_PARENT_ABS=$(cd "$TARGET_DIR/.." && pwd)
REAL_DIR="$TARGET_DIR_ABS/real"
REAL_VIDEOS_DIR="$TARGET_DIR_ABS/real_sync"
WORKERS="${WORKERS:-32}"

if [ -d "$REAL_DIR" ]; then
	echo "[+] Found existing extracted PisaBench directory: $REAL_DIR"
	echo "[+] Skipping download and extraction"
else
	echo "[+] Downloading PisaBench via huggingface_hub Python utility..."

	# Ensure the hub utility is installed locally
	pip install -q "huggingface_hub[cli]"

	# Use python inline to cleanly stream the massive 11.5GB dataset split
	python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
repo_id='nyu-visionx/pisa-experiments',
allow_patterns='pisabench/real.zip',
local_dir='$TARGET_PARENT_ABS',
repo_type='dataset'
)
"

	echo "[+] Extracting valid zip archive into $TARGET_DIR_ABS ..."
	# The utility saves the file into the exact repository path layout: pisabench/real.zip
	unzip -q "$TARGET_DIR_ABS/real.zip" -d "$TARGET_DIR_ABS"

	# Optional cleanup of the zip archive to preserve disk space
	rm -f "$TARGET_DIR_ABS/real.zip"
fi

rm -rf "$REAL_VIDEOS_DIR"
mkdir -p "$REAL_VIDEOS_DIR"
echo "[+] Converting movie.mp4 to 120 FPS to match the recorded speed and storing $REAL_VIDEOS_DIR ..."

job_count=0

for seq_dir in "$REAL_DIR"/*; do
	[ -d "$seq_dir" ] || continue

	input_video="$seq_dir/movie.mp4"
	[ -f "$input_video" ] || continue

	seq_name=$(basename "$seq_dir")
	output_video="$REAL_VIDEOS_DIR/$seq_name.mp4"

	echo "[+] Reinterpreting $input_video at 120 FPS -> $output_video"
	ffmpeg -y \
        -loglevel error \
		-r 120 \
		-i "$input_video" \
		-an \
		-c:v libx264 \
		-pix_fmt yuv420p \
		-r 120 \
		"$output_video" &

	job_count=$((job_count + 1))
	if [ $((job_count % WORKERS)) -eq 0 ]; then
		echo "[+] Waiting for batch of $WORKERS jobs to finish..."
		wait
	fi
done

wait


echo "========================================================"
echo "[+] Pipeline complete. Target folder: $TARGET_DIR_ABS"
echo "========================================================"
