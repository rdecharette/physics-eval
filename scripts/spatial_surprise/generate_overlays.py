#!/usr/bin/env python3
"""Overlay existing spatial surprise maps on original IntPhys frame sequences."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def generate(args):
    if not 0 <= args.alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
    if not args.tags or len(set(args.tags)) != len(args.tags):
        raise ValueError("Expected unique configuration tags")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is required")
    prepared = []
    for tag in args.tags:
        if Path(tag).name != tag or tag in (".", ".."):
            raise ValueError("Expected a configuration directory name")
        maps = sorted((args.output_root / tag / "datasets").rglob("map.png"))
        if not maps:
            raise ValueError(f"No maps found for {tag}")
        for map_path in maps:
            directory = map_path.parent
            relative = directory.relative_to(args.output_root / tag)
            metadata = json.loads((directory / "map.json").read_text())
            manifest_path = args.variants_root / tag / relative / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if metadata["dataset_path"] != relative.as_posix() or manifest["dataset_path"] != relative.as_posix():
                raise ValueError(f"Dataset metadata mismatch: {directory}")
            with Image.open(map_path) as image:
                if image.mode != "RGB" or image.size != (metadata["width"], metadata["height"]):
                    raise ValueError(f"Invalid RGB heatmap: {map_path}")
                heatmap = image.copy()
            if any(d % 2 for d in heatmap.size):
                raise ValueError("YUV420 output requires even dimensions")
            names = manifest["source_frame_names"]
            if not names or len(names) != len(set(names)) or len(names) != manifest["source_frame_count"]:
                raise ValueError("Invalid source frame list")
            frames = []
            for name in names:
                if Path(name).name != name or Path(name).suffix != ".png":
                    raise ValueError(f"Invalid frame name: {name}")
                path = args.source_root / relative / "scene" / name
                with Image.open(path) as image:
                    image.verify()
                frames.append(path)
            source_fps, target_fps = Fraction(manifest["source_fps"]), Fraction(manifest["target_fps"])
            if min(source_fps, target_fps) <= 0:
                raise ValueError("Frame rates must be positive")
            if not args.overwrite and any((directory / name).exists() for name in ("viz.mp4", "viz.json")):
                raise ValueError(f"Overlay exists: {directory}; use --overwrite to regenerate")
            prepared.append((directory, heatmap, frames, source_fps, target_fps, metadata))
    for directory, heatmap, frames, source_fps, target_fps, metadata in prepared:
        output = directory / "viz.mp4"
        temporary = directory / "viz.partial.mp4"
        width, height = heatmap.size
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                   "-framerate", str(source_fps), "-i", "pipe:0", "-an",
                   "-vf", f"fps=fps={target_fps}:round=near,scale=in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p",
                   "-c:v", "libx264", "-profile:v", "high", "-crf", "12", "-preset", "fast",
                   "-threads", "1", "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
                   "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart", str(temporary)]
        try:
            with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
                try:
                    for path in frames:
                        with Image.open(path) as image:
                            original = image.convert("RGB").resize(heatmap.size, Image.Resampling.BILINEAR)
                        process.stdin.write(Image.blend(original, heatmap, args.alpha).tobytes())
                except BaseException:
                    process.kill()
                    raise
                finally:
                    process.stdin.close()
                if process.wait():
                    raise RuntimeError(f"ffmpeg failed for {output}")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        record = {
            "dataset_path": metadata["dataset_path"], "alpha": args.alpha,
            "blend": "(1-alpha)*resized_original_RGB + alpha*map_RGB",
            "source_frame_count": len(frames), "source_fps": str(source_fps), "target_fps": str(target_fps),
            "width": width, "height": height, "display": metadata["display"],
            "map_sha256": hashlib.sha256((directory / "map.png").read_bytes()).hexdigest(),
            "encoding": "libx264 High, CRF12, yuv420p, BT709 limited range",
        }
        (directory / "viz.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Saved {output}", flush=True)
    return len(prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["s4-v0.5"])
    parser.add_argument("--alpha", type=float, default=0.4, help="Heatmap opacity (default: 0.4)")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--variants-root", type=Path, default=ROOT / "cache/datasets_variants/masked")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/spatial-surprise")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        generate(args)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
