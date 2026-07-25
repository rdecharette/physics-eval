#!/usr/bin/env sh

set -e

ZIP_URL="https://download-intphys.cognitive-ml.fr/dev.tar.gz"
TARGET_DIR="datasets/intphys"
ZIP_PATH="dev.tar.gz"

usage() {
    echo "Usage: $0"
    exit 1
}

if ! command -v tar >/dev/null 2>&1; then
    echo "[-] Error: 'tar' is not found in PATH."
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[-] Error: 'ffmpeg' is not found in PATH."
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

if [ -d "$TARGET_DIR" ]; then
    TARGET_DIR_ABS=$(cd "$TARGET_DIR" && pwd)
    echo "[+] Dataset folder already exists at $TARGET_DIR_ABS"
    echo "[+] Skipping download/extraction and proceeding to video generation..."
elif [ -e "$TARGET_DIR" ]; then
    echo "[-] Error: $TARGET_DIR exists but is not a directory."
    exit 1
else
    mkdir -p "$TARGET_DIR"
    TARGET_DIR_ABS=$(cd "$TARGET_DIR" && pwd)
    echo "[+] Downloading IntPhys2 archive..."
    if [ "$DOWNLOAD_TOOL" = "curl" ]; then
        curl -L "$ZIP_URL" -o "$ZIP_PATH"
    else
        wget -O "$ZIP_PATH" "$ZIP_URL"
    fi

    echo "[+] Extracting archive into $TARGET_DIR_ABS ..."
    tar -xzf "$ZIP_PATH" -C "$TARGET_DIR_ABS"
    rm -f "$ZIP_PATH"
fi

echo "[+] Creating 25 FPS videos from scene_*.png files..."
SCENE_COUNT=0
CREATED_COUNT=0
SCENE_LIST="$TARGET_DIR_ABS/.scene_dirs.list"

find "$TARGET_DIR_ABS" -type f -name "scene_25fps.mp4" -delete

find "$TARGET_DIR_ABS" -type d -name "scene" > "$SCENE_LIST"

while IFS= read -r SCENE_DIR; do
    SCENE_COUNT=$((SCENE_COUNT + 1))
    if [ ! -d "$SCENE_DIR" ]; then
        echo "[!] Skipping missing directory: $SCENE_DIR"
        continue
    fi

    OLD_OUTPUT_VIDEO="$SCENE_DIR/scene_25fps.mp4"
    OUTPUT_VIDEO="$SCENE_DIR/_scene_25fps.mp4"

    if [ -f "$OLD_OUTPUT_VIDEO" ]; then
        rm -f "$OLD_OUTPUT_VIDEO"
        echo "[+] Deleted existing: $OLD_OUTPUT_VIDEO"
    fi

    FIRST_PNG=$(find "$SCENE_DIR" -maxdepth 1 -type f -name "scene_*.png" 2>/dev/null | head -n 1)

    if [ -z "$FIRST_PNG" ]; then
        echo "[!] Skipping $SCENE_DIR (no scene_*.png files found)"
        continue
    fi

    ffmpeg -y -hide_banner -loglevel error \
        -nostdin \
        -framerate 25 \
        -pattern_type glob -i "$SCENE_DIR/scene_*.png" \
        -c:v libx264 -pix_fmt yuv420p \
        "$OUTPUT_VIDEO"

    CREATED_COUNT=$((CREATED_COUNT + 1))
    echo "[+] Created: $OUTPUT_VIDEO"
done < "$SCENE_LIST"

rm -f "$SCENE_LIST"

if [ "$SCENE_COUNT" -eq 0 ]; then
    echo "[!] No folders named 'scene' found in $TARGET_DIR_ABS"
else
    echo "[+] Video generation complete. Created $CREATED_COUNT file(s)."
fi

echo "========================================================"
echo "[+] Pipeline complete. Target folder: $TARGET_DIR_ABS"
echo "========================================================"
