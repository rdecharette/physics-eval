#!/usr/bin/env python3
"""Compute ImageNet-referenced FID for a dataset defined by a video-list file."""

from __future__ import annotations

import argparse
import fcntl
import json
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import torch


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
FID_SHUFFLE_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID for a list of videos against ImageNet stats")
    parser.add_argument("--type", type=str, default="fid", choices=["fid", "kid"], help="fid|kid (default: fid)")
    parser.add_argument("--video-list", type=Path, required=True, help="Text file containing one video path per line")
    parser.add_argument("--clean-fid-root", type=Path, required=True, help="Path to the cloned clean-fid repository")
    parser.add_argument("--reference-path", type=Path, required=True, help="Path to the ImageNet image folder")
    parser.add_argument("--reference-set", type=str, default="imagenet_custom", help="Custom stats name used inside clean-fid")
    parser.add_argument("--frames-root", type=Path, required=True, help="Root folder for extracted frame caches")
    parser.add_argument("--lock-dir", type=Path, required=True, help="Folder used for file locks around custom stats generation")
    parser.add_argument("--output-path", type=Path, required=True, help="JSON file receiving the computed FID result")
    parser.add_argument("--mode", type=str, default="clean", help="clean-fid mode (default: clean)")
    parser.add_argument("--model-name", type=str, default="inception_v3", help="clean-fid feature model (default: inception_v3)")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for clean-fid feature extraction")
    parser.add_argument("--num-workers", type=int, default=16, help="clean-fid DataLoader workers")
    parser.add_argument("--batch-size", type=int, default=32, help="clean-fid batch size")
    parser.add_argument("--frame-stride", type=int, default=25, help="Keep every Nth frame from each video (original frame indices)")
    parser.add_argument("--max-frames-per-video", type=int, default=32, help="Maximum sampled frames to keep per video")
    parser.add_argument("--fid-max-frames", type=int, default=None, help="Maximum number of extracted frames to use for FID after shuffling with a fixed seed")
    parser.add_argument("--limit-videos", type=int, default=None, help="Optional limit on how many videos from the list to process")
    parser.add_argument("--overwrite-frames", action="store_true", help="Re-extract frames even if a cache already exists")
    parser.add_argument("--overwrite-stats", action="store_true", help="Rebuild the cached ImageNet custom stats even if present")
    return parser.parse_args()


def load_clean_fid(clean_fid_root: Path):
    sys.path.insert(0, str(clean_fid_root))
    from cleanfid import fid  # type: ignore

    return fid


def count_images(folder: Path) -> int:
    return sum(1 for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def list_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_video_paths(video_list: Path, limit_videos: int | None) -> list[Path]:
    with video_list.open("r", encoding="utf-8") as f:
        paths = [Path(line.strip()) for line in f if line.strip()]
    if limit_videos is not None:
        paths = paths[:limit_videos]
    return paths


def extract_sampled_frames(
    video_paths: list[Path],
    frames_dir: Path,
    frame_stride: int,
    max_frames_per_video: int | None,
    overwrite_frames: bool,
) -> int:
    if overwrite_frames and frames_dir.exists():
        shutil.rmtree(frames_dir)

    if frames_dir.exists():
        existing = count_images(frames_dir)
        if existing > 0:
            print(f"Reusing extracted frames in {frames_dir} ({existing} images)")
            return existing

    frames_dir.mkdir(parents=True, exist_ok=True)

    for index, video_path in enumerate(video_paths, start=1):
        if not video_path.is_file():
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        out_dir = frames_dir / f"{index:06d}_{video_path.stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
        ]

        if frame_stride > 1:
            cmd.extend(["-vf", f"select=not(mod(n\\,{frame_stride}))"])
            cmd.extend(["-vsync", "vfr"])

        if max_frames_per_video is not None:
            cmd.extend(["-frames:v", str(max_frames_per_video)])

        cmd.append(str(out_dir / "frame_%06d.png"))
        print(f"[{index}/{len(video_paths)}] Extracting frames from {video_path}")
        subprocess.run(cmd, check=True)

    extracted = count_images(frames_dir)
    print(f"Extracted {extracted} frames into {frames_dir}")
    return extracted


def ensure_custom_stats(fid_module, args: argparse.Namespace) -> None:
    args.lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.lock_dir / f"{args.reference_set}.lock"

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        stats_exist = fid_module.test_stats_exists(
            args.reference_set,
            args.mode,
            model_name=args.model_name,
            metric="FID",
        )
        if stats_exist and args.overwrite_stats:
            fid_module.remove_custom_stats(
                args.reference_set,
                mode=args.mode,
                model_name=args.model_name,
            )
            stats_exist = False

        if stats_exist:
            print(f"Using existing clean-fid custom stats: {args.reference_set}")
            return

        print(f"Creating clean-fid custom stats '{args.reference_set}' from {args.reference_path}")
        fid_module.make_custom_stats(
            args.reference_set,
            str(args.reference_path),
            mode=args.mode,
            model_name=args.model_name,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            device=torch.device(args.device),
            verbose=True,
        )


def build_fid_subset_dir(frames_dir: Path, fid_max_frames: int | None) -> tuple[Path, int, tempfile.TemporaryDirectory[str] | None]:
    image_paths = list_images(frames_dir)
    if fid_max_frames is None or len(image_paths) <= fid_max_frames:
        return frames_dir, len(image_paths), None

    rng = random.Random(FID_SHUFFLE_SEED)
    selected_paths = list(image_paths)
    rng.shuffle(selected_paths)
    selected_paths = selected_paths[:fid_max_frames]

    temp_dir = tempfile.TemporaryDirectory(prefix="fid-subset-", dir=frames_dir.parent)
    subset_dir = Path(temp_dir.name)
    for index, image_path in enumerate(selected_paths, start=1):
        target_path = subset_dir / f"frame_{index:08d}{image_path.suffix.lower()}"
        target_path.symlink_to(image_path.resolve())

    print(
        f"Using {len(selected_paths)} shuffled frames out of {len(image_paths)} for FID "
        f"(seed={FID_SHUFFLE_SEED})"
    )
    return subset_dir, len(selected_paths), temp_dir


def main() -> None:
    args = parse_args()

    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")
    if args.max_frames_per_video is not None and args.max_frames_per_video < 1:
        raise SystemExit("--max-frames-per-video must be >= 1")
    if args.fid_max_frames is not None and args.fid_max_frames < 1:
        raise SystemExit("--fid-max-frames must be >= 1")
    if not args.video_list.is_file():
        raise SystemExit(f"Video list does not exist: {args.video_list}")
    if not args.reference_path.is_dir():
        raise SystemExit(f"The reference path does not exist or is not a directory: {args.reference_path}")

    fid_module = load_clean_fid(args.clean_fid_root)

    dataset_name = args.video_list.stem
    max_frames_label = "all" if args.max_frames_per_video is None else str(args.max_frames_per_video)
    frames_dir = args.frames_root / dataset_name / f"stride-{args.frame_stride}_max-{max_frames_label}"

    video_paths = read_video_paths(args.video_list, args.limit_videos)
    if not video_paths:
        raise SystemExit(f"No video paths found in {args.video_list}")

    ensure_custom_stats(fid_module, args)
    extracted_frames = extract_sampled_frames(
        video_paths=video_paths,
        frames_dir=frames_dir,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
        overwrite_frames=args.overwrite_frames,
    )

    fid_frames_dir, fid_num_frames, fid_temp_dir = build_fid_subset_dir(frames_dir, args.fid_max_frames)

    try:
        if args.type == "fid":
            print(f"Compute FID")
            score = fid_module.compute_fid(
                str(fid_frames_dir),
                mode=args.mode,
                model_name=args.model_name,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                device=torch.device(args.device),
                dataset_name=args.reference_set,
                dataset_res=0,
                dataset_split="custom",
                verbose=True,
            )
        elif args.type == "kid":
            print(f"Compute KID")
            score = fid_module.compute_kid(
                str(fid_frames_dir),
                mode=args.mode,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                device=torch.device(args.device),
                dataset_name=args.reference_set,
                dataset_res=0,
                dataset_split="custom",
                verbose=True,
            )
    finally:
        if fid_temp_dir is not None:
            fid_temp_dir.cleanup()

    result = {
        "dataset": dataset_name,
        "video_list": str(args.video_list),
        "reference_path": str(args.reference_path),
        "reference_set": args.reference_set,
        "fid": float(score),
        "mode": args.mode,
        "model_name": args.model_name,
        "device": args.device,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "frame_stride": args.frame_stride,
        "max_frames_per_video": args.max_frames_per_video,
        "fid_max_frames": args.fid_max_frames,
        "limit_videos": args.limit_videos,
        "num_videos": len(video_paths),
        "num_frames": fid_num_frames,
        "num_extracted_frames": extracted_frames,
        "frames_dir": str(frames_dir),
        "fid_frames_dir": str(fid_frames_dir),
        "computed_at": datetime.now().astimezone().isoformat(),
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"Saved FID result to: {args.output_path}")
    print(f"FID({dataset_name} vs {args.reference_set}) = {score:.6f}")


if __name__ == "__main__":
    main()