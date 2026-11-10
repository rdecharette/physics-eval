#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
RESULTS_DIR="${RESULTS_DIR:-output/geophys}"
MODE="${MODE:-*}"
FORMAT="${FORMAT:-*}"
BACKBONE="${BACKBONE:-*}"

# Discover score-bearing directories recursively, including every scoring mode.
python ./plot_geophys_stats.py --path "$RESULTS_DIR" --sorted \
    --mode "$MODE" --format "$FORMAT" --backbone "$BACKBONE"
