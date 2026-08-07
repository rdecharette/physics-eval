#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <DATASET_NAME>" >&2
    exit 1
fi

if [[ -z "${SCRATCH:-}" ]]; then
    echo "SCRATCH is not set; this script is intended for Jean-Zay." >&2
    exit 1
fi

DATASET_NAME="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$ROOT_DIR/datasets"
REPO_DATASET_PATH="$DATASETS_DIR/$DATASET_NAME"
REPO_TAR_PATH="$DATASETS_DIR/$DATASET_NAME.tar"

if [[ -e "$REPO_DATASET_PATH" || -L "$REPO_DATASET_PATH" ]]; then
    echo "Dataset exists: $REPO_DATASET_PATH"
    exit 0
fi

SCRATCH_DATASET_DIR="$SCRATCH/datasets/$DATASET_NAME"
SCRATCH_DATASET_PARENT="$(dirname "$SCRATCH_DATASET_DIR")"

if [[ -f "$REPO_TAR_PATH" ]]; then
    echo -e "\nFound tar file for dataset: $REPO_TAR_PATH" >&2

    mkdir -p "$SCRATCH_DATASET_PARENT"

    if [[ -e "$SCRATCH_DATASET_DIR" || -L "$SCRATCH_DATASET_DIR" ]]; then
        echo "Removing existing scratch dataset directory: $SCRATCH_DATASET_DIR" >&2
        rm -rf "$SCRATCH_DATASET_DIR"
    fi

    echo "Creating scratch dataset directory: $SCRATCH_DATASET_DIR" >&2
    mkdir -p "$SCRATCH_DATASET_DIR"

    echo -e "\nExtracting dataset tarball to scratch dataset directory: $SCRATCH_DATASET_DIR" >&2

    tar_args=(
        -xf "$REPO_TAR_PATH"
        --checkpoint=1000
        --checkpoint-action=echo='tar: extracted %u records'
        -C "$SCRATCH_DATASET_DIR"
        --strip-components=1
    )

    tar "${tar_args[@]}"
else
    if [[ ! -e "$SCRATCH_DATASET_DIR" ]]; then
        echo "No TAR and Scratch dataset doesn't exist: $SCRATCH_DATASET_DIR"
        exit 0
    fi
fi

echo "Creating symlink from repo dataset path to scratch dataset directory: $REPO_DATASET_PATH -> $SCRATCH_DATASET_DIR" >&2
ln -s "$SCRATCH_DATASET_DIR" "$REPO_DATASET_PATH"

echo "Dataset ready: $REPO_DATASET_PATH -> $SCRATCH_DATASET_DIR"
