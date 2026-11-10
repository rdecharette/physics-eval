#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/geophys/slurm_job_geophys.sh"

export EVAL_MAX="${EVAL_MAX:-500}"
export VARIANT="${VARIANT:-512p_30fps}"
# The runner stages all datasets instead of the profile's single-dataset hook.
unset REQUIRED_DATASET

# Purposely omitting physics-iq-verified as it's physical nature differs

REQUIRED_DATASETS=(ContPhy intphys intphys2 NewtPhys PhysBench physionpp_trim-e200 pisabench)

# One invocation shares percentiles across all selected videos (500 per dataset
# by default), while still writing a separate .score file for each dataset.
submit_geophys_job \
    contphy intphys_possible intphys2_possible NewtPhys PhysBench \
    physionpp pisabench
