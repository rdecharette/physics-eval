#\!/usr/bin/env python3
"""Split IntPhys dev scenarios into possible/impossible lists.

Scans datasets/intphys/dev at exactly 3 directory levels (e.g. O1/02/1),
reads status.json in each level-3 directory, and writes video paths to:
- datasets/intphys/possible.txt
- datasets/intphys/impossible.txt

Each written path is relative to datasets/intphys/dev, e.g.:
O1/02/1/_scene_25fps.mp4
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("datasets/intphys")
DEV_ROOT = ROOT / "dev"
POSSIBLE_OUT = DEV_ROOT / "possible.txt"
IMPOSSIBLE_OUT = DEV_ROOT / "impossible.txt"
VIDEO_NAME = "_scene_25fps.mp4"


def main() -> None:
    possible: list[str] = []
    impossible: list[str] = []

    # Level-3 directories only: dev/<lvl1>/<lvl2>/<lvl3>
    level3_dirs = [p for p in DEV_ROOT.glob("*/*/*") if p.is_dir()]

    for scene_dir in sorted(level3_dirs):
        status_path = scene_dir / "status.json"
        video_path = scene_dir / VIDEO_NAME

        if not status_path.is_file() or not video_path.is_file():
            continue

        with status_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        is_possible = data.get("header", {}).get("is_possible")
        rel_video = str(video_path.relative_to(DEV_ROOT))

        if is_possible is True:
            possible.append(rel_video)
        elif is_possible is False:
            impossible.append(rel_video)

    POSSIBLE_OUT.write_text("\n".join(possible) + ("\n" if possible else ""), encoding="utf-8")
    IMPOSSIBLE_OUT.write_text("\n".join(impossible) + ("\n" if impossible else ""), encoding="utf-8")

    print(f"Wrote {len(possible)} possible paths to {POSSIBLE_OUT}")
    print(f"Wrote {len(impossible)} impossible paths to {IMPOSSIBLE_OUT}")


if __name__ == "__main__":
    main()
