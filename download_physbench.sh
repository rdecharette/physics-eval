#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default target directory if none is provided
TARGET_DIR="./datasets/PhysBench"
REPO_URL="https://huggingface.co/datasets/USC-PSI-Lab/PhysBench"

# Display usage instructions
usage() {
    echo "Usage: $0 [-d target_directory]"
    echo "  -d : Specify the target directory where the repository will be cloned and video.zip extracted (Default: $TARGET_DIR)"
    exit 1
}

# Parse command-line flags
while getopts "d:h" opt; do
    case "$opt" in
        d) TARGET_DIR="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo "========================================================"
echo "Target Directory: $TARGET_DIR"
echo "========================================================"

# 1. Validate required tools exist
if ! command -v git > /dev/null 2>&1; then
    echo "[-] Error: 'git' is not found in PATH."
    exit 1
fi

if ! command -v unzip > /dev/null 2>&1; then
    echo "[-] Error: 'unzip' utility is not found in PATH."
    exit 1
fi

# 2. Setup absolute target path
mkdir -p "$TARGET_DIR"
TARGET_DIR_ABS=$(cd "$TARGET_DIR" && pwd)

echo "[+] Preparing PhysBench repository in $TARGET_DIR_ABS ..."

# If target already has a repo, refresh it; otherwise clone.
if [ -d "$TARGET_DIR_ABS/.git" ]; then
    echo "[*] Existing git repository detected at target. Pulling latest changes..."
    git -C "$TARGET_DIR_ABS" pull --ff-only
else
    if [ "$(find "$TARGET_DIR_ABS" -mindepth 1 -maxdepth 1 | head -n 1)" ]; then
        echo "[-] Error: target directory is not empty and is not a git repository: $TARGET_DIR_ABS"
        echo "    Use an empty directory or a directory that already contains the PhysBench git repo."
        exit 1
    fi
    git clone "$REPO_URL" "$TARGET_DIR_ABS"
fi

# Pull large files if git-lfs is available.
if command -v git-lfs > /dev/null 2>&1; then
    echo "[*] Running git lfs pull..."
    git -C "$TARGET_DIR_ABS" lfs pull
else
    echo "[!] 'git-lfs' not found. If video.zip is a pointer file, install git-lfs and rerun."
fi

ZIP_PATH="$TARGET_DIR_ABS/video.zip"

if [ ! -f "$ZIP_PATH" ]; then
    echo "[-] Error: video.zip not found at $ZIP_PATH"
    exit 1
fi

echo "[+] Unzipping video.zip in $TARGET_DIR_ABS ..."
unzip -oq "$ZIP_PATH" -d "$TARGET_DIR_ABS"

echo "========================================================"
echo "[+] Pipeline complete. Target folder: $TARGET_DIR_ABS"
echo "========================================================"
