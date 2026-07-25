#!/usr/bin/env sh

set -e

ZIP_URL="https://huggingface.co/datasets/nyu-visionx/pisa-experiments/blob/main/pisabench/real.zip"
TARGET_DIR="datasets/pisabench"
ZIP_PATH="pisabench.zip"

mkdir -p "$TARGET_DIR"
TARGET_DIR_ABS=$(cd "$TARGET_DIR" && pwd)

echo "[+] Downloading PisaBench via huggingface_hub Python utility..."

# Ensure the hub utility is installed locally
pip install -q "huggingface_hub[cli]"

# Use python inline to cleanly stream the massive 11.5GB dataset split
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
repo_id='nyu-visionx/pisa-experiments',
allow_patterns='pisabench/real.zip',
local_dir='$TARGET_DIR_ABS',
repo_type='dataset'
)
"

echo "[+] Extracting valid zip archive into $TARGET_DIR_ABS ..."
# The utility saves the file into the exact repository path layout: pisabench/real.zip
unzip -q "$TARGET_DIR_ABS/pisabench/real.zip" -d "$TARGET_DIR_ABS"

# Optional cleanup of the zip archive to preserve disk space
rm -f "$TARGET_DIR_ABS/pisabench/real.zip"


echo "========================================================"
echo "[+] Pipeline complete. Target folder: $TARGET_DIR_ABS"
echo "========================================================"
