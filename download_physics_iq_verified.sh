#!/usr/bin/env sh

set -e

mkdir -p ./tmp/download
export TMPDIR=./tmp/download
export TMP=./tmp/download
export TEMP=./tmp/download


REPO_ID="Anates-Labs-Research/Physics-IQ-Verified"
REPO_URL="https://huggingface.co/datasets/$REPO_ID"
TARGET_DIR="datasets/physics-iq-verified"

usage() {
    echo "Usage: $0"
    exit 1
}

if [ "$#" -ne 0 ]; then
    usage
fi

if ! command -v git >/dev/null 2>&1; then
    echo "[-] Error: 'git' is not available in PATH."
    exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "[-] Error: 'git-lfs' is not available in PATH."
    echo "    Install it first, then run: git lfs install"
    exit 1
fi

if ! git lfs env >/dev/null 2>&1; then
    echo "[-] Error: git-lfs is installed but not initialized for this user."
    echo "    Run: git lfs install"
    exit 1
fi

TARGET_PARENT=$(dirname "$TARGET_DIR")
mkdir -p "$TARGET_PARENT"
TARGET_PARENT_ABS=$(cd "$TARGET_PARENT" && pwd)
TARGET_DIR_ABS="$TARGET_PARENT_ABS/$(basename "$TARGET_DIR")"
LFS_STORAGE_DIR="${LFS_STORAGE_DIR:-${TMPDIR:-/tmp}/physics-iq-verified-lfs}"

mkdir -p "$LFS_STORAGE_DIR"

printf '%s\n' "========================================================"
printf '%s\n' "Target Directory: $TARGET_DIR_ABS"
printf '%s\n' "LFS Storage Dir: $LFS_STORAGE_DIR"
printf '%s\n' "========================================================"

if [ -d "$TARGET_DIR_ABS/.git" ]; then
    echo "[+] Found existing git repo at $TARGET_DIR_ABS. Resuming download..."
elif [ -d "$TARGET_DIR_ABS" ] && [ "$(ls -A "$TARGET_DIR_ABS" 2>/dev/null | wc -l | tr -d ' ')" -gt 0 ]; then
    echo "[-] Error: target directory '$TARGET_DIR' already exists and is not empty."
    echo "    Remove it or empty it before cloning."
    exit 1
else
    echo "[+] Cloning Physics-IQ Verified dataset (LFS smudge disabled during clone)..."
    GIT_LFS_SKIP_SMUDGE=1 git clone "$REPO_URL" "$TARGET_DIR_ABS"
fi

echo "[+] Configuring safer LFS transfer settings for network filesystems..."
git -C "$TARGET_DIR_ABS" config lfs.storage "$LFS_STORAGE_DIR"
git -C "$TARGET_DIR_ABS" config lfs.concurrenttransfers 1
git -C "$TARGET_DIR_ABS" config lfs.basictransfersonly true

echo "[+] Restoring worktree metadata before LFS checkout (safe to ignore if clean)..."
git -C "$TARGET_DIR_ABS" restore --source=HEAD :/ || true

echo "[+] Pulling LFS files..."
git -C "$TARGET_DIR_ABS" lfs pull


echo "[+] Cleaning cache..."
rm -R "$TMPDIR"

echo "========================================================"
echo "[+] Pipeline complete. Target folder: $TARGET_DIR_ABS"
echo "========================================================"
