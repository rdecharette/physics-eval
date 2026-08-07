#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

# python convert_videos.py --dataset datasets/ContPhy/ --filter "**/output_Full.mp4" --ext ".mp4" --height 256 --workers 32 --fps 30 --dst cache/datasets_vjepa-ready/ContPhy/ &&
# python convert_videos.py --dataset datasets/intphys/ --height 256 --workers 32 --fps 30 --dst cache/datasets_vjepa-ready/intphys/ &&
# python convert_videos.py --dataset datasets/intphys2/ --height 256 --workers 32 --fps 30 --dst cache/datasets_vjepa-ready/intphys2/ &&
# python convert_videos.py --dataset datasets/PhysBench/ --height 256 --workers 32 --fps 30 --dst cache/datasets_vjepa-ready/PhysBench/ &&
python convert_videos.py --dataset "./../physics-sim/output/sims/v4_bis/" --filter "**/dl3dv/random/**/_fps-25_render.mp4" --ext ".mp4" --height 256 --workers 32 --fps 30 --dst "cache/datasets_vjepa-ready/newtphys/" &&
# python convert_videos.py --dataset datasets/physics-iq-verified/ --filter "**/full-videos/**/*.mp4" --ext ".mp4" --height 256 --workers 32 --fps 30 --dst cache/datasets_vjepa-ready/physics-iq-verified/ &&
# python convert_videos.py --dataset datasets/physionpp_trim-e200/ --height 256 --workers 32 --fps 30 --dst cache/datasets_vjepa-ready/physionpp_trim-e200/ &&
# python convert_videos.py --dataset datasets/pisabench/ --height 256 --workers 32 --fps 30 --dst cache/datasets_vjepa-ready/pisabench/ &&
echo "Done converting videos to cache"
