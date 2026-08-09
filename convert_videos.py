#!/usr/bin/env python3
"""Convert videos to target FPS and optional target spatial bounds, preserving directory structure."""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


DEFAULT_EXTENSIONS = {".mp4", ".mov", ".avi"}

FFMPEG_FAILURE_PATTERNS = (
    "no frame!",
    "invalid data found when processing input",
    "error while decoding stream",
    "could not find codec parameters",
    "error opening input file",
    "output file #0 does not contain any stream",
)


def looks_like_ffmpeg_failure(stderr: str, returncode: int) -> bool:
    if returncode != 0:
        return True
    if not stderr:
        return False
    lowered_stderr = stderr.lower()
    return any(pattern.lower() in lowered_stderr for pattern in FFMPEG_FAILURE_PATTERNS)


def find_videos(src_root: Path, extensions: set[str]) -> list[Path]:
    normalized_extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    }
    if not normalized_extensions:
        return []

    videos: list[Path] = []
    seen: set[Path] = set()
    for ext in sorted(normalized_extensions):
        print("Looking for videos with extension:", ext)
        pattern = f"*{ext}"
        for path in src_root.rglob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                videos.append(path)

    return sorted(videos)


def normalize_filter_pattern(src_root: Path, pattern: str) -> str:
    normalized = pattern.strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("/"):
        normalized = normalized[1:]

    dataset_prefix = f"{src_root.name}/"
    if normalized.startswith(dataset_prefix):
        normalized = normalized[len(dataset_prefix):]

    return normalized


def _round_to_even(value: float) -> int:
    rounded = int(round(value))
    if rounded % 2 != 0:
        rounded += 1
    return max(2, rounded)


def resolve_target_size(
    sample_video: Path,
    target_width: int | None,
    target_height: int | None,
    ffprobe_bin: str = "ffprobe",
) -> tuple[int, int] | None:
    if target_width is None and target_height is None:
        return None
    if target_width is not None and target_height is not None:
        return target_width, target_height

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(sample_video),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    raw = result.stdout.strip().splitlines()
    if not raw:
        raise RuntimeError(f"ffprobe returned no resolution for {sample_video}")

    src_w_str, src_h_str = raw[0].split("x", 1)
    src_w = int(src_w_str)
    src_h = int(src_h_str)
    if src_w <= 0 or src_h <= 0:
        raise RuntimeError(f"invalid source resolution for {sample_video}: {src_w}x{src_h}")

    if target_width is not None:
        resolved_h = _round_to_even(src_h * target_width / src_w)
        return target_width, resolved_h

    if target_height is None:
        raise RuntimeError("internal error: expected target_height when target_width is None")

    resolved_w = _round_to_even(src_w * target_height / src_h)
    return resolved_w, target_height


def parse_trim(trim_arg: str) -> tuple[int | None, int | None]:
    parts = trim_arg.split(",")
    if len(parts) != 2:
        raise ValueError("--trim must be in the form start,end (for example: 50,150 or 50, or ,150)")

    start_raw = parts[0].strip()
    end_raw = parts[1].strip()

    start_frame: int | None = None
    end_frame: int | None = None

    if start_raw:
        start_frame = int(start_raw)
    if end_raw:
        end_frame = int(end_raw)

    if start_frame is None and end_frame is None:
        raise ValueError("--trim cannot be empty. Provide at least a start or end frame.")
    if start_frame is not None and start_frame < 1:
        raise ValueError("--trim start frame must be >= 1")
    if end_frame is not None and end_frame < 1:
        raise ValueError("--trim end frame must be >= 1")
    if start_frame is not None and end_frame is not None and start_frame > end_frame:
        raise ValueError("--trim start frame must be <= end frame")

    return start_frame, end_frame


def format_trim_suffix(trim_start: int | None, trim_end: int | None) -> str:
    if trim_start is not None and trim_end is not None:
        return f" [trim {trim_start}..{trim_end}]"
    if trim_start is not None:
        return f" [trim {trim_start}..end]"
    if trim_end is not None:
        return f" [trim 1..{trim_end}]"
    return ""


def shorten_path(path: Path, max_parts: int = 4) -> str:
    parts = path.as_posix().split("/")
    if len(parts) <= max_parts:
        return path.as_posix()
    return f".../{'/'.join(parts[-max_parts:])}"


def convert_video(
    src_path: Path,
    dst_path: Path,
    ffmpeg_bin: str,
    ffmpeg_threads: int | None,
    overwrite: bool,
    target_fps: int | None,
    target_width: int | None,
    target_height: int | None,
    trim_start: int | None,
    trim_end: int | None,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    filters: list[str] = []
    if trim_start is not None or trim_end is not None:
        # --trim is 1-indexed, while ffmpeg frame index n is 0-indexed.
        trim_clauses: list[str] = []
        if trim_start is not None:
            trim_clauses.append(f"gte(n\\,{trim_start - 1})")
        if trim_end is not None:
            trim_clauses.append(f"lte(n\\,{trim_end - 1})")
        filters.append(f"select='{ '*'.join(trim_clauses) }'")
        # Rebuild continuous timestamps after frame selection.
        filters.append("setpts=N/FRAME_RATE/TB")

    if target_fps is not None:
        filters.append(f"fps={target_fps}")
    if target_width is not None and target_height is not None:
        # Fit inside the requested box while preserving source aspect ratio.
        filters.append(
            f"scale=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease"
        )
    elif target_width is not None:
        # ffmpeg computes height automatically while preserving aspect ratio.
        filters.append(f"scale=w={target_width}:h=-2")
    elif target_height is not None:
        # ffmpeg computes width automatically while preserving aspect ratio.
        filters.append(f"scale=w=-2:h={target_height}")

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(src_path),
    ]
    if filters:
        cmd.extend(["-vf", ",".join(filters)])

    if ffmpeg_threads is not None:
        cmd.extend(["-threads", str(ffmpeg_threads)])

    cmd.extend([
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(dst_path),
    ])
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    if result.returncode != 0 or looks_like_ffmpeg_failure(stderr, result.returncode):
        cleanup_detail = ""
        if dst_path.exists():
            try:
                dst_path.unlink()
                cleanup_detail = f" Removed incomplete output: {dst_path}"
            except OSError as exc:
                cleanup_detail = f" Could not remove incomplete output {dst_path}: {exc}"
        detail = stderr or stdout or f"ffmpeg exited with code {result.returncode}"
        print(f"FAILED running: {' '.join(cmd)}")
        raise RuntimeError(
            f"ffmpeg conversion failed ({result.returncode}): {detail}{cleanup_detail}"
        ) from None


def convert_one(
    task: tuple[int, int, Path, Path],
    ffmpeg_bin: str,
    ffmpeg_threads: int | None,
    overwrite: bool,
    target_fps: int | None,
    target_width: int | None,
    target_height: int | None,
    trim_start: int | None,
    trim_end: int | None,
) -> tuple[int, int, Path, Path, str | None, float | None]:
    idx, total, src_path, dst_path = task
    try:
        start_time = time.perf_counter()
        convert_video(
            src_path,
            dst_path,
            ffmpeg_bin=ffmpeg_bin,
            ffmpeg_threads=ffmpeg_threads,
            overwrite=overwrite,
            target_fps=target_fps,
            target_width=target_width,
            target_height=target_height,
            trim_start=trim_start,
            trim_end=trim_end,
        )
        elapsed = time.perf_counter() - start_time
        return idx, total, src_path, dst_path, None, elapsed
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        return idx, total, src_path, dst_path, str(exc), None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert all videos in a dataset folder to a target FPS and optional "
            "spatial bounds, and write them to a mirrored output folder with the same "
            "directory structure."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/intphys2"),
        help="Dataset root directory to scan (default: datasets/intphys2)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help=(
            "Optional glob pattern applied to dataset-relative paths. "
            "Examples: '**/*_img*' or '/data_v1/**/*_img*'"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Target output FPS. If omitted, FPS is left unchanged.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Target output width for aspect-ratio-preserving resize.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target output height for aspect-ratio-preserving resize.",
    )
    parser.add_argument(
        "--trim",
        type=str,
        default=None,
        help=(
            "Trim range in original-frame indices (1-indexed), as start,end. "
            "Examples: 50,150 or 50, or ,150"
        ),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help=(
            "Destination root directory. If omitted, defaults to: "
            "<dataset>_<FPS>fps, <dataset>_<W>x<H>, or <dataset>_<W>x<H>_<FPS>fps."
        ),
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=sorted(DEFAULT_EXTENSIONS),
        help="Video file extensions to include (default: common video extensions)",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg binary/command name (default: ffmpeg)",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=None,
        help=(
            "Per-ffmpeg thread limit passed via -threads. "
            "Use a low value (for example 1 or 2) when running many workers. "
            "Default: ffmpeg automatic threading."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without running ffmpeg.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers for ffmpeg conversions (default: 16).",
    )
    args = parser.parse_args()

    src_root = args.dataset.resolve()
    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.ext}

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.fps is not None and args.fps < 1:
        raise SystemExit("--fps must be >= 1")
    if args.width is not None and args.width < 1:
        raise SystemExit("--width must be >= 1")
    if args.height is not None and args.height < 1:
        raise SystemExit("--height must be >= 1")
    if args.ffmpeg_threads is not None and args.ffmpeg_threads < 1:
        raise SystemExit("--ffmpeg-threads must be >= 1")

    trim_start: int | None = None
    trim_end: int | None = None
    if args.trim is not None:
        try:
            trim_start, trim_end = parse_trim(args.trim)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {src_root}")

    print(f"Scanning for videos under {src_root} with extensions: {sorted(extensions)}")
    videos = find_videos(src_root, {e.lower() for e in extensions})
    print(f"Found {len(videos)} videos")
    if args.filter is not None:
        print(f"Filtering videos using pattern: {args.filter}")
        filter_pattern = args.filter.strip()
        if not filter_pattern:
            raise SystemExit("--filter cannot be empty")

        normalized_pattern = normalize_filter_pattern(src_root, filter_pattern)
        matched_paths = {
            p.resolve()
            for p in src_root.glob(normalized_pattern)
            if p.is_file()
        }
        videos = [v for v in videos if v.resolve() in matched_paths]
        print(f"After filtering, {len(videos)} videos remain")

    if not videos:
        raise SystemExit(f"No matching videos found under: {src_root}")

    resolved_size: tuple[int, int] | None = None
    if args.width is not None or args.height is not None:
        print(f"Resolving target size from first video: {videos[0]}")
        try:
            resolved_size = resolve_target_size(videos[0], args.width, args.height)
        except Exception as exc:
            raise SystemExit(
                f"Failed to infer concrete target WxH for naming from {videos[0]}: {exc}"
            ) from exc

        assert resolved_size is not None
        print(f"Resolved target size for naming: {resolved_size[0]}x{resolved_size[1]}")

    if args.dst is not None:
        dst_root = args.dst.resolve()
    else:
        trim_suffix = ""
        if trim_start is not None and trim_end is not None:
            trim_suffix = f"trim-s{trim_start}-e{trim_end}"
        elif trim_start is not None:
            trim_suffix = f"trim-s{trim_start}"
        elif trim_end is not None:
            trim_suffix = f"trim-e{trim_end}"

        if resolved_size is not None:
            size_suffix = f"{resolved_size[0]}x{resolved_size[1]}"
            if args.fps is not None:
                suffix = f"{size_suffix}_{args.fps}fps{trim_suffix}"
            else:
                suffix = f"{size_suffix}_{trim_suffix}"
        elif args.fps is not None:
            suffix = f"{args.fps}fps_{trim_suffix}"
        else:
            suffix = f"{trim_suffix}"
        dst_root = Path(f"{src_root}_{suffix}")

    print(f"Found {len(videos)} videos under {src_root}")
    if args.filter is not None:
        print(f"Relative path filter: {args.filter}")
    if args.fps is not None:
        print(f"Target FPS: {args.fps}")
    else:
        print("Target FPS: unchanged")
    if args.width is not None and args.height is not None:
        print(f"Target spatial bounds: {args.width}x{args.height} (aspect ratio preserved)")
    elif args.width is not None:
        print(f"Target width: {args.width} (aspect ratio preserved)")
    elif args.height is not None:
        print(f"Target height: {args.height} (aspect ratio preserved)")
    if resolved_size is not None:
        print(f"Resolved naming size: {resolved_size[0]}x{resolved_size[1]}")
    if trim_start is not None and trim_end is not None:
        print(f"Trim range (original frames): {trim_start}..{trim_end}")
    elif trim_start is not None:
        print(f"Trim range (original frames): {trim_start}..end")
    elif trim_end is not None:
        print(f"Trim range (original frames): 1..{trim_end}")
    print(f"Output root: {dst_root}")

    trim_progress_suffix = format_trim_suffix(trim_start, trim_end)

    converted = 0
    skipped = 0
    failed = 0
    cumulative_elapsed = 0.0
    cumulative_elapsed = 0.0
    cumulative_elapsed = 0.0

    tasks: list[tuple[int, int, Path, Path]] = []
    total = len(videos)

    for idx, src_path in enumerate(videos, start=1):
        rel_path = src_path.relative_to(src_root)
        dst_path = dst_root / rel_path

        if dst_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{idx}/{total}] Skipping existing: {dst_path}{trim_progress_suffix}")
            continue

        tasks.append((idx, total, src_path, dst_path))

        # if idx % (len(videos)//10) == 0 or idx == total:
        #     subprocess.run(["bash", "video_stats.sh", str(src_path)], check=True)

    if skipped > 0:
        print(f"Skipped {skipped} existing files (use --overwrite to replace)")
    
    if args.dry_run:
        for idx, total, src_path, dst_path in tasks:
            print(f"[{idx}/{total}] Converting: {src_path} -> {dst_path}{trim_progress_suffix}")
        print(
            f"Dry-run complete. To convert={len(tasks)}, Skipped={skipped}, Total={total}"
        )
        return

    if not tasks:
        print(f"Done. Converted=0, Failed=0, Skipped={skipped}, Total={total}")
        return

    print(f"Starting conversion with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(
                convert_one,
                task,
                args.ffmpeg_bin,
                args.ffmpeg_threads,
                args.overwrite,
                args.fps,
                args.width,
                args.height,
                trim_start,
                trim_end,
            ): task
            for task in tasks
        }

        progress = tqdm(total=len(future_to_task), dynamic_ncols=True, leave=True)
        for future in as_completed(future_to_task):
            idx, total, src_path, dst_path, error, elapsed = future.result()
            rel_src = src_path.relative_to(src_root)
            short_src = shorten_path(rel_src, max_parts=4)

            progress.set_description_str(f"done {short_src}")
            if error is None:
                converted += 1
            else:
                failed += 1

            postfix_parts: list[str] = []
            if failed > 0:
                postfix_parts.append(f"{failed} failed")
            if postfix_parts:
                progress.set_postfix_str(", ".join(postfix_parts))
            progress.update(1)
        progress.close()

    print(f"Done. Converted={converted}, Failed={failed}, Skipped={skipped}, Total={total}")


if __name__ == "__main__":
    main()
