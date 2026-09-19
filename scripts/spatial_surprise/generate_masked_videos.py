#!/usr/bin/env python3
"""Generate gray-masked IntPhys videos without storing intermediate frames."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_LIST = ROOT / "scripts/spatial_surprise/videos.txt"


def natural_key(path):
    return tuple((0, int(part)) if part.isdigit() else (1, part)
                 for part in re.split(r"(\d+)", path.name))


def load_masks(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    width, height = manifest["width"], manifest["height"]
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("Mask dimensions must be positive integers")
    records = manifest["masks"]
    if not records or manifest["mask_count"] != len(records):
        raise ValueError("Manifest mask count is empty or inconsistent")
    loaded = []
    names = set()
    coverage = Image.new("L", (width, height), 0)
    for record in records:
        filename = record["filename"]
        if (Path(filename).name != filename or Path(filename).suffix != ".png"
                or filename in names):
            raise ValueError(f"Invalid or duplicate mask filename: {filename}")
        names.add(filename)
        bounds = record["bounds_pixels"]
        x0, y0, x1, y1 = (bounds[key] for key in ("x0", "y0", "x1", "y1"))
        if (any(type(value) is not int for value in (x0, y0, x1, y1))
                or not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height):
            raise ValueError(f"Invalid rectangle bounds: {filename}")
        expected = Image.new("L", (width, height), 0)
        expected.paste(255, (x0, y0, x1, y1))
        with Image.open(directory / filename) as image:
            image.load()
            if (image.mode != "L" or image.size != (width, height)
                    or ImageChops.difference(image, expected).getbbox()):
                raise ValueError(f"Mask does not match binary manifest rectangle: {filename}")
        coverage = ImageChops.lighter(coverage, expected)
        loaded.append((record, expected))
    if coverage.getextrema() != (255, 255):
        raise ValueError("Masks do not cover the entire image")
    return manifest, loaded


def input_videos(source_root, video_list):
    video_list = video_list if video_list is not None else DEFAULT_VIDEO_LIST
    entries = [line.strip() for line in video_list.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if not entries:
        raise ValueError("Video list is empty")
    videos = []
    seen = set()
    for entry in entries:
        relative = Path(entry)
        if (relative.is_absolute() or ".." in relative.parts
                or not relative.parts or relative.parts[0] != "datasets"):
            raise ValueError(f"Expected a repository-relative datasets/... video path: {entry}")
        if relative in seen:
            raise ValueError(f"Duplicate input video: {entry}")
        seen.add(relative)
        frames = sorted((source_root / relative / "scene").glob("*.png"), key=natural_key)
        if not frames:
            raise ValueError(f"No scene/*.png frames found for {entry}")
        sizes = set()
        for frame in frames:
            with Image.open(frame) as image:
                sizes.add(image.size)
                image.verify()
        if len(sizes) != 1:
            raise ValueError(f"Source frame dimensions vary within {entry}")
        videos.append((relative, frames, next(iter(sizes))))
    return videos


def generate(args):
    source_fps, target_fps = Fraction(args.source_fps), Fraction(args.target_fps)
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame rates must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is required on PATH")
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                              capture_output=True, text=True, check=True)
    if "libx264 " not in encoders.stdout:
        raise ValueError("ffmpeg must provide the libx264 encoder")
    mask_dir = args.mask_dir.resolve()
    mask_manifest, masks = load_masks(mask_dir)
    videos = input_videos(args.source_root.resolve(), args.video_list)
    width, height = mask_manifest["width"], mask_manifest["height"]
    if width % 2 or height % 2:
        raise ValueError("YUV420 video output requires even mask dimensions")
    output = args.output_root.resolve() / mask_dir.name
    for relative, _, _ in videos:
        if (output / relative).exists():
            raise ValueError(f"Video output already exists: {output / relative}")
    output.mkdir(parents=True, exist_ok=True)
    background = Image.new("RGB", (width, height), (128, 128, 128))
    all_paths = []
    for relative, frames, original_size in videos:
        resized = []
        for frame in frames:
            with Image.open(frame) as image:
                resized.append(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))
        video_dir = output / relative
        video_dir.mkdir(parents=True)
        variants = []
        for record, mask in masks:
            encoded = video_dir / f"masked_{Path(record['filename']).stem}.mp4"
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                       "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                       "-framerate", str(source_fps), "-i", "pipe:0", "-an",
                       "-vf", f"fps=fps={target_fps}:round=near,scale=in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p",
                       "-c:v", "libx264", "-profile:v", "high",
                       "-crf", "12", "-preset", "fast", "-threads", "1", "-pix_fmt", "yuv420p",
                       "-color_range", "tv", "-colorspace", "bt709",
                       "-color_primaries", "bt709", "-color_trc", "bt709",
                       "-movflags", "+faststart", str(encoded)]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
                try:
                    for image in resized:
                        masked = Image.composite(image, background, mask)
                        process.stdin.write(masked.tobytes())
                except BaseException:
                    process.kill()
                    raise
                finally:
                    process.stdin.close()
                if process.wait():
                    raise RuntimeError(f"ffmpeg failed for {encoded}")
            all_paths.append(str(encoded))
            variants.append({"mask": record["filename"], "bounds_pixels": record["bounds_pixels"],
                             "video": str(encoded)})
            print(f"Generated {encoded}", flush=True)
        manifest = {
            "schema_version": 2, "dataset_path": relative.as_posix(),
            "source_directory": str((args.source_root / relative).resolve()),
            "source_frame_names": [frame.name for frame in frames], "source_frame_count": len(frames),
            "source_size": {"width": original_size[0], "height": original_size[1]},
            "output_size": {"width": width, "height": height},
            "resize": "Pillow RGB conversion then bilinear resize before masking",
            "source_fps": str(source_fps), "target_fps": str(target_fps),
            "source_duration_seconds": float(Fraction(len(frames), 1) / source_fps),
            "frame_mapping_policy": "ffmpeg fps filter, round=near; duplicate/drop frames to preserve duration to target-frame precision",
            "fill_rgb": [128, 128, 128], "mask_manifest": str(mask_dir / "manifest.json"),
            "scale": mask_manifest["scale"], "overlap": mask_manifest["overlap"],
            "coordinate_decimals": mask_manifest.get("coordinate_decimals"),
            "encoding": {"codec": "libx264", "crf": 12, "preset": "fast", "threads": 1,
                         "input_pixel_format": "rgb24", "requested_output_pixel_format": "yuv420p",
                         "profile": "high", "color_space": "bt709", "color_range": "limited",
                         "color_primaries": "bt709", "color_transfer": "bt709",
                         "pixel_preservation": "MP4 uses chroma subsampling and lossy compression; intermediate RGB frames are not stored",
                         "fps_filter": f"fps=fps={target_fps}:round=near"},
            "variants": variants,
        }
        (video_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "videos.txt").write_text("\n".join(all_paths) + "\n")
    print(f"Completed {len(all_paths)} variants. Evaluation list: {output / 'videos.txt'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-list", type=Path, default=DEFAULT_VIDEO_LIST,
                        help="Text file of repository-relative datasets/... directories (default: scripts/spatial_surprise/videos.txt)")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--mask-dir", type=Path, default=ROOT / "data/mask/s4-v0.5")
    parser.add_argument("--output-root", type=Path, default=ROOT / "cache/datasets_variants/masked")
    parser.add_argument("--source-fps", default="25", help="Original PNG sequence rate; accepts fractions")
    parser.add_argument("--target-fps", default="30", help="Encoded video rate; accepts fractions")
    args = parser.parse_args()
    try:
        generate(args)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ZeroDivisionError,
            subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
