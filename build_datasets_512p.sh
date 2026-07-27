#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
HEIGHT=512
OUTDIR="cache/datasets_${HEIGHT}p"
WORKERS=32

python convert_videos.py --dataset datasets/ContPhy/ --filter "**/output_Full.mp4" --ext ".mp4" --height "$HEIGHT" --workers "$WORKERS" --dst "$OUTDIR/ContPhy/"

python convert_videos.py --dataset datasets/intphys/ --height "$HEIGHT" --workers "$WORKERS" --dst "$OUTDIR/intphys/"

python convert_videos.py --dataset datasets/intphys2/ --height "$HEIGHT" --workers "$WORKERS" --dst "$OUTDIR/intphys2/"

python convert_videos.py --dataset datasets/PhysBench/ --height "$HEIGHT" --workers "$WORKERS" --dst "$OUTDIR/PhysBench/"

python convert_videos.py --dataset "/nfs/data/workspaces/rdechare/codes/physics-eval/../physics-sim/output/sims/v4_bis/dl3dv/random/" --filter "**/_fps-25_render.mp4" --ext ".mp4" --height "$HEIGHT" --workers "$WORKERS" --dst "$OUTDIR/newtphys/dl3dv/random/"

python convert_videos.py --dataset datasets/physics-iq-verified/ --filter "**/full-videos/**/*.mp4" --ext ".mp4" --height "$HEIGHT" --workers "$WORKERS" --dst "$OUTDIR/physics-iq-verified/"

python convert_videos.py --dataset datasets/physionpp_trim-e200/ --height "$HEIGHT" --workers "$WORKERS" --dst "$OUTDIR/physionpp_trim-e200/"

python convert_videos.py --dataset datasets/pisabench/ --height "$HEIGHT" --workers "$WORKERS" --dst "$OUTDIR/pisabench/"

echo "Done converting videos to cache"
