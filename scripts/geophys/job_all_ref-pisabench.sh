#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use the complete PisaBench list at the target variant as a fixed reference.
unset REFERENCE
export REFERENCE_DATASET="pisabench"
export OUTPUT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)/output/geophys/ref-pisabench"

# Purposely omitting physics-iq-verified as it's physical nature differs
bash "$SCRIPT_DIR/job_all_combined.sh"
