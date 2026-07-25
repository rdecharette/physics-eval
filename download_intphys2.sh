#!/usr/bin/env sh

set -e

ZIP_URL="https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip"
TARGET_DIR="datasets/intphys2"
ZIP_PATH="IntPhys2.zip"

usage() {
    echo "Usage: $0"
    exit 1
}

if ! command -v unzip >/dev/null 2>&1; then
    echo "[-] Error: 'unzip' is not found in PATH."
    exit 1
fi

if command -v curl >/dev/null 2>&1; then
    DOWNLOAD_TOOL="curl"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOAD_TOOL="wget"
else
    echo "[-] Error: neither 'curl' nor 'wget' is available in PATH."
    exit 1
fi

echo "========================================================"
echo "Target Directory: $TARGET_DIR"
echo "========================================================"

mkdir -p "$TARGET_DIR"
TARGET_DIR_ABS=$(cd "$TARGET_DIR" && pwd)

if [ "$(ls -A "$TARGET_DIR_ABS" 2>/dev/null | wc -l | tr -d ' ')" -gt 0 ]; then
    echo "[+] Dataset folder is not empty at $TARGET_DIR_ABS"
    echo "[+] Skipping download/extraction."
    exit 0
fi

echo "[+] Downloading IntPhys2 archive..."
if [ "$DOWNLOAD_TOOL" = "curl" ]; then
    curl -L "$ZIP_URL" -o "$ZIP_PATH"
else
    wget -O "$ZIP_PATH" "$ZIP_URL"
fi

echo "[+] Extracting archive into $TARGET_DIR_ABS ..."
unzip -oq "$ZIP_PATH" -d "$TARGET_DIR_ABS"

rm -f "$ZIP_PATH"

echo "========================================================"
echo "[+] Pipeline complete. Target folder: $TARGET_DIR_ABS"
echo "========================================================"
