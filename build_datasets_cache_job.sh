#!/usr/bin/env bash
set -uo pipefail

# ----------------------------------------------
# Get a CPU node first:
# ----------------------------------------------
#
# salloc --account=pab@cpu --nodes=6 --ntasks-per-node=1 --cpus-per-task=64 --time=06:00:00
# ----------------------------------------------

CPUS=64
LOG_DIR="logs/datasetscache"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

run_step () {
  local name="$1"
  local cmd="$2"
  local log_file="${LOG_DIR}/${RUN_TS}_${name}.log"

  echo "[$(date +%F' '%T)] starting ${name}, log: ${log_file}"

  srun --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --exclusive \
    --job-name="resize_${name}" \
    bash -lc "
      set -euo pipefail
      module load ffmpeg
      module load anaconda-py3/2024.06
      source \"\$(conda info --base)/etc/profile.d/conda.sh\"
      conda activate physics-eval
      cd \"\$WORK/physics-eval\"
      ${cmd}
    " >"$log_file" 2>&1 &
}

run_step "convert" "bash build_datasets_cache_all.sh"

wait
echo "All dataset steps finished"