#!/usr/bin/env bash
set -uo pipefail

# ----------------------------------------------
# Get a CPU node first:
# ----------------------------------------------
#
# salloc --account=pab@cpu --nodes=6 --ntasks-per-node=1 --cpus-per-task=64 --time=06:00:00
# ----------------------------------------------

HEIGHT=512
WORKERS=16
FFMPEG_THREADS=2
CPUS=32
LOG_DIR="logs/datasetsresize"
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

run_step "contphy" "python convert_videos.py --dataset datasets/ContPhy/ --filter '**/output_Full.mp4' --ext .mp4 --height $HEIGHT --workers $WORKERS --ffmpeg-threads $FFMPEG_THREADS --dst cache/datasets_${HEIGHT}p/ContPhy/"
run_step "intphys" "python convert_videos.py --dataset datasets/intphys/ --height $HEIGHT --workers $WORKERS --ffmpeg-threads $FFMPEG_THREADS --dst cache/datasets_${HEIGHT}p/intphys/"
run_step "intphys2" "python convert_videos.py --dataset datasets/intphys2/ --height $HEIGHT --workers $WORKERS --ffmpeg-threads $FFMPEG_THREADS --dst cache/datasets_${HEIGHT}p/intphys2/"
run_step "physbench" "python convert_videos.py --dataset datasets/PhysBench/ --height $HEIGHT --workers $WORKERS --ffmpeg-threads $FFMPEG_THREADS --dst cache/datasets_${HEIGHT}p/PhysBench/"
run_step "newtphys" "python convert_videos.py --dataset datasets/NewtPhys/ --filter 'dl3dv/random/**/_fps-25_render.mp4' --ext .mp4 --height $HEIGHT --workers $WORKERS --ffmpeg-threads $FFMPEG_THREADS --dst cache/datasets_${HEIGHT}p/NewtPhys"
run_step "physics_iq" "python convert_videos.py --dataset datasets/physics-iq-verified/ --filter '**/full-videos/**/*.mp4' --ext .mp4 --height $HEIGHT --workers $WORKERS --ffmpeg-threads $FFMPEG_THREADS --dst cache/datasets_${HEIGHT}p/physics-iq-verified/"
run_step "physionpp" "python convert_videos.py --dataset datasets/physionpp_trim-e200/ --height $HEIGHT --workers $WORKERS --ffmpeg-threads $FFMPEG_THREADS --dst cache/datasets_${HEIGHT}p/physionpp_trim-e200/"
run_step "pisabench" "python convert_videos.py --dataset datasets/pisabench/ --height $HEIGHT --workers $WORKERS --ffmpeg-threads $FFMPEG_THREADS --dst cache/datasets_${HEIGHT}p/pisabench/"

wait
echo "All dataset steps finished"