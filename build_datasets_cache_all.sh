#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

DATASET_LIST_FILES=(
	"contphy.txt"
	"intphys_possible.txt"
	"intphys_impossible.txt"
	"intphys2_possible.txt"
	"intphys2_impossible.txt"
	"PhysBench.txt"
	"newtphys_random_all.txt"
	"physics-iq-verified.txt"
	"physionpp.txt"
	"pisabench.txt"
)

DATASET_LIST="$(IFS=,; echo "${DATASET_LIST_FILES[*]}")"
export DATASET_LIST

# For Vbench
TARGET_FPS=30 TARGET_HEIGHT=512 WORKERS=32 bash build_datasets_cache.sh

# For VJEPA
TARGET_FPS=30 TARGET_HEIGHT=256 WORKERS=32 bash build_datasets_cache.sh


echo "Done converting videos to cache"
