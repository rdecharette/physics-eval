#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

WORKERS=${WORKERS:-32}

TARGET_FPS=${TARGET_FPS:-30}
TARGET_HEIGHT=${TARGET_HEIGHT:-256}
DATASET_LIST=${DATASET_LIST:-}

IFS=',' read -r -a DATASET_LIST_FILES <<< "$DATASET_LIST"


CACHE_ROOT="cache/datasets_variants"
if [ -n "$TARGET_FPS" ] && [ -n "$TARGET_HEIGHT" ]; then
	CACHE_DIR="${CACHE_ROOT}/${TARGET_HEIGHT}p_${TARGET_FPS}fps"
elif [ -n "$TARGET_FPS" ]; then
	CACHE_DIR="${CACHE_ROOT}/${TARGET_FPS}fps"
elif [ -n "$TARGET_HEIGHT" ]; then
	CACHE_DIR="${CACHE_ROOT}/${TARGET_HEIGHT}p"
else
	echo "Error: Either TARGET_FPS or TARGET_HEIGHT must be specified."
	exit 1
fi

TOTAL_CONVERTED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0
TOTAL_VIDEOS=0

for list_file in "${DATASET_LIST_FILES[@]}"; do

	echo -e "\nConverting: $list_file with TARGET_HEIGHT=${TARGET_HEIGHT} and TARGET_FPS=${TARGET_FPS}"
	log_file=$(mktemp)
	python convert_videos.py \
		--input "$list_file" \
		--height "$TARGET_HEIGHT" \
		--workers "$WORKERS" \
		--fps "$TARGET_FPS" \
		--dst "$CACHE_DIR" | tee "$log_file"

	summary_line=$(grep -E '^Done\. Converted=' "$log_file" | tail -n 1 || true)
	if [[ "$summary_line" =~ Converted=([0-9]+),\ Failed=([0-9]+),\ Skipped=([0-9]+),\ Total=([0-9]+) ]]; then
		TOTAL_CONVERTED=$((TOTAL_CONVERTED + BASH_REMATCH[1]))
		TOTAL_FAILED=$((TOTAL_FAILED + BASH_REMATCH[2]))
		TOTAL_SKIPPED=$((TOTAL_SKIPPED + BASH_REMATCH[3]))
		TOTAL_VIDEOS=$((TOTAL_VIDEOS + BASH_REMATCH[4]))
	fi
	rm -f "$log_file"
done

echo "========================================================"
echo "Recap: Converted=${TOTAL_CONVERTED}, Failed=${TOTAL_FAILED}, Skipped=${TOTAL_SKIPPED}, Total=${TOTAL_VIDEOS}"
echo "========================================================"

echo "Done converting videos to cache"
