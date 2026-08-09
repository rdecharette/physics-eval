#!/usr/bin/env bash
set -uo pipefail

export PYTHONUNBUFFERED=1
HEIGHT="${HEIGHT:-512}"
OUTDIR="cache/datasets_${HEIGHT}p"
WORKERS="${WORKERS:-32}"
FFMPEG_THREADS="${FFMPEG_THREADS:-1}"

DATASET_SUCCESS=0
DATASET_FAILED=0
TOTAL_CONVERTED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0
TOTAL_VIDEOS=0

run_convert() {
	local dataset_name="$1"
	shift

	local log_file
	log_file="$(mktemp)"

	echo "=== Converting ${dataset_name} ==="
	if python convert_videos.py "$@" | tee "$log_file"; then
		DATASET_SUCCESS=$((DATASET_SUCCESS + 1))
	else
		DATASET_FAILED=$((DATASET_FAILED + 1))
		echo "[ERROR] ${dataset_name} conversion command failed"
	fi

	local summary_line
	summary_line="$(grep -E '^Done\. Converted=' "$log_file" | tail -n 1 || true)"
	if [[ -n "$summary_line" ]] && [[ "$summary_line" =~ Converted=([0-9]+),\ Failed=([0-9]+),\ Skipped=([0-9]+),\ Total=([0-9]+) ]]; then
		TOTAL_CONVERTED=$((TOTAL_CONVERTED + BASH_REMATCH[1]))
		TOTAL_FAILED=$((TOTAL_FAILED + BASH_REMATCH[2]))
		TOTAL_SKIPPED=$((TOTAL_SKIPPED + BASH_REMATCH[3]))
		TOTAL_VIDEOS=$((TOTAL_VIDEOS + BASH_REMATCH[4]))
	else
		echo "[WARN] Could not parse summary for ${dataset_name}"
	fi

	rm -f "$log_file"
}

run_convert "ContPhy" --dataset datasets/ContPhy/ --filter "**/output_Full.mp4" --ext ".mp4" --height "$HEIGHT" --workers "$WORKERS" --ffmpeg-threads "$FFMPEG_THREADS" --dst "$OUTDIR/ContPhy/"

run_convert "intphys" --dataset datasets/intphys/ --height "$HEIGHT" --workers "$WORKERS" --ffmpeg-threads "$FFMPEG_THREADS" --dst "$OUTDIR/intphys/"

run_convert "intphys2" --dataset datasets/intphys2/ --height "$HEIGHT" --workers "$WORKERS" --ffmpeg-threads "$FFMPEG_THREADS" --dst "$OUTDIR/intphys2/"

run_convert "PhysBench" --dataset datasets/PhysBench/ --height "$HEIGHT" --workers "$WORKERS" --ffmpeg-threads "$FFMPEG_THREADS" --dst "$OUTDIR/PhysBench/"

run_convert "NewtPhys" --dataset datasets/NewtPhys/ --filter "dl3dv/random/**/_fps-25_render.mp4" --ext ".mp4" --height "$HEIGHT" --workers "$WORKERS" --ffmpeg-threads "$FFMPEG_THREADS" --dst "$OUTDIR/newtphys/dl3dv/random/"

run_convert "physics-iq-verified" --dataset datasets/physics-iq-verified/ --filter "**/full-videos/**/*.mp4" --ext ".mp4" --height "$HEIGHT" --workers "$WORKERS" --ffmpeg-threads "$FFMPEG_THREADS" --dst "$OUTDIR/physics-iq-verified/"

run_convert "physionpp_trim-e200" --dataset datasets/physionpp_trim-e200/ --height "$HEIGHT" --workers "$WORKERS" --ffmpeg-threads "$FFMPEG_THREADS" --dst "$OUTDIR/physionpp_trim-e200/"

run_convert "pisabench" --dataset datasets/pisabench/ --height "$HEIGHT" --workers "$WORKERS" --ffmpeg-threads "$FFMPEG_THREADS" --dst "$OUTDIR/pisabench/"

echo "=== Resize Summary ==="
echo "Datasets: success=${DATASET_SUCCESS}, failed=${DATASET_FAILED}, total=$((DATASET_SUCCESS + DATASET_FAILED))"
echo "Videos: converted=${TOTAL_CONVERTED}, failed=${TOTAL_FAILED}, skipped=${TOTAL_SKIPPED}, total=${TOTAL_VIDEOS}"

if [[ ${DATASET_FAILED} -gt 0 || ${TOTAL_FAILED} -gt 0 ]]; then
	echo "Done with failures."
	exit 1
fi

echo "Done converting videos to cache"
