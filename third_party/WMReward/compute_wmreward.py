# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.s

"""
Sample script to compute VJEPA surprise (World Model Reward) for videos.

Usage:
    python compute_wmreward.py --video_path /path/to/video.mp4
"""

import argparse
import os
import builtins
import copy
import csv
import math
from functools import partial
from pathlib import Path
import random
import tempfile

import torch
import torchvision.transforms.functional as F
from decord import VideoReader
from torchvision.transforms.functional import resize

from utils import (
    compute_vjepa_loss_sliding_window,
    get_video,
)

# Ensure Python stdout is flushed on every print so SLURM logs show live progress.
print = partial(builtins.print, flush=True)


def load_vjepa_models(model_name="vitg"):
    """Load VJEPA models from torchhub."""
    img_size = 384 if "384" in model_name else 256
    hub_dir = os.environ.get("VJEPA_HUB_DIR")
    hub_source = hub_dir or "facebookresearch/vjepa2"
    hub_kwargs = {"source": "local"} if hub_dir else {}

    if model_name == "vith":
        encoder, predictor = torch.hub.load(hub_source, "vjepa2_vit_huge", **hub_kwargs)
    elif model_name == "vitg":
        encoder, predictor = torch.hub.load(hub_source, "vjepa2_vit_giant", **hub_kwargs)
    elif model_name == "vitg384":
        encoder, predictor = torch.hub.load(hub_source, "vjepa2_vit_giant_384", **hub_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    target_encoder = copy.deepcopy(encoder)
    return encoder, target_encoder, predictor, img_size


def load_video_as_tensor(video_path, max_frames=49, img_size=256):
    """Load video and convert to tensor [1, C, T, H, W] in range [-1, 1]."""
    video_np = get_video(video_path, max_frames=max_frames)
    video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float()
    perm_video_tensor = video_tensor.permute(1, 0, 2, 3)
    
    # Old resize (deforming)
    # video_tensor = resize(perm_video_tensor, [img_size, img_size])

    # New resize (ratio preserving)
    video_tensor = F.resize(perm_video_tensor, img_size)
    video_tensor = F.center_crop(video_tensor, [img_size, img_size])
    
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    video_tensor = (video_tensor / 127.5) - 1.0
    return video_tensor.unsqueeze(0)


def assert_video_is_30_fps(video_path: str) -> None:
    fps = float(VideoReader(video_path).get_avg_fps())
    if abs(fps - 30.0) > 1e-3:
        raise ValueError(f"Expected 30 FPS, got {fps:.6f} for video: {video_path}")


def compute_vjepa_surprise(
    video_path: str,
    model_name: str = "vitg",
    window_size: int = 16,
    context_frames: int = 8,
    stride: int = 2,
    seed: int = 42,
    max_frames: int = 49,
    mode: str = "mean",
):
    """Compute VJEPA surprise score for a video."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading VJEPA model: {model_name}...")
    encoder, target_encoder, predictor, img_size = load_vjepa_models(model_name)
    encoder = encoder.to(device).eval()
    target_encoder = target_encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    print(f"Loading video: {video_path}")
    assert_video_is_30_fps(video_path)
    video_tensor = load_video_as_tensor(video_path, max_frames=max_frames, img_size=img_size)
    video_tensor = video_tensor.to(device)
    print(f"Video tensor shape: {video_tensor.shape}")

    print("Computing VJEPA surprise...")
    with torch.no_grad():
        loss = compute_vjepa_loss_sliding_window(
            video_tensor=video_tensor,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            img_size=img_size,
            window_size=window_size,
            loss_exp=2,
            masking_mode="causal",
            context_frames=context_frames,
            is_vae_output=True,
            seed=seed,
            stride=stride,
            mode=mode,
        )

    surprise_score = loss.item()
    print(f"\n{'='*50}")
    print(f"VJEPA Surprise Score: {surprise_score:.6f}")
    print(f"VJEPA Similarity Score: {1.0 - surprise_score:.6f}")
    print(f"{'='*50}")

    return surprise_score


def compute_multi_vjepa_surprise(
    videos_paths: list[str],
    model_name: str = "vitg",
    window_size: int = 16,
    context_frames: int = 8,
    stride: int = 2,
    seed: int = 42,
    max_frames: int = 49,
    output_path: str | None = None,
    force_recompute: bool = False,
    mode: str = "mean",
    max_videos: int | None = None,
    direct_paths: bool = False,
):
    """Compute VJEPA surprise score for multiple videos."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading VJEPA model: {model_name}...")
    encoder, target_encoder, predictor, img_size = load_vjepa_models(model_name)
    encoder = encoder.to(device).eval()
    target_encoder = target_encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    processed_videos = set()
    resumed_scores = {}
    resolved_output_path = None
    if output_path is not None:
        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

        if resolved_output_path.exists() and not force_recompute:
            with resolved_output_path.open("r", encoding="utf-8", newline="") as output_file:
                reader = csv.reader(output_file)
                if direct_paths and next(reader, None) != ["video", "surprise"]:
                    raise ValueError(f"Unexpected surprise CSV header: {resolved_output_path}")
                for row in reader:
                    if not row or row[0] == "video":
                        continue
                    if direct_paths:
                        if len(row) != 2:
                            continue
                        try:
                            score = float(row[1])
                        except (IndexError, ValueError):
                            continue
                        if not math.isfinite(score):
                            continue
                        if row[0] in resumed_scores and resumed_scores[row[0]] != score:
                            raise ValueError(f"Conflicting finite surprise scores for {row[0]}")
                        resumed_scores[row[0]] = score
                    processed_videos.add(row[0])
            if direct_paths:
                # Atomically remove invalid and repeated rows before appending repairs.
                temporary_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", encoding="utf-8", newline="",
                        dir=resolved_output_path.parent, delete=False,
                    ) as output_file:
                        temporary_path = Path(output_file.name)
                        writer = csv.writer(output_file)
                        writer.writerow(["video", "surprise"])
                        writer.writerows(resumed_scores.items())
                    os.replace(temporary_path, resolved_output_path)
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
            print(
                f"Found existing surprise file at {resolved_output_path}; "
                f"loaded {len(processed_videos)} processed videos."
            )
        else:
            print(f"Creating surprise file at {resolved_output_path}")
            with resolved_output_path.open("w", encoding="utf-8", newline="") as output_file:
                csv.writer(output_file).writerow(["video", "surprise"])

    surprise_scores = dict(resumed_scores)

    print("Shuffling video paths deterministically...")
    VIDEO_PATHS_SHUFFLE_SEED = 42  # Important to remain the same as the path shuffling for surprises and other per-video metrics. 42 is the answer to life. Hence to the metrics.
    random.Random(VIDEO_PATHS_SHUFFLE_SEED).shuffle(videos_paths)
    if max_videos is None:
        max_videos = len(videos_paths)
    else:
        print(f"Limiting to {max_videos} videos for evaluation.")
    
    # We do not cap the videos paths but rather process up to max_videos videos.
    # This allows for skipping videos that are not decodable or have fewer frames than the window size.
    
    error = 0
    for i, video_path in enumerate(videos_paths):
        video_path = str(video_path)
        if (i-error) >= max_videos:
            print(f"Reached maximum number of videos to process: {max_videos}. Stopping.")
            break

        # Retrieve the vjepa-ready video cache (256p at ~30 FPS)
        if not direct_paths:
            video_path = video_path.replace("/datasets/", "/cache/datasets_variants/256p_30fps/datasets/")
        
        print(f"\nProcessing video {i + 1 - error}/{min(len(videos_paths), max_videos)}: {video_path}")

        if not force_recompute and video_path in processed_videos:
            print(f"Skipping already processed video")
            continue
        
        assert_video_is_30_fps(video_path)
        
        surprise_scores[video_path] = None
        try:
            print(f"Loading video...")
            video_tensor = load_video_as_tensor(video_path, max_frames=max_frames, img_size=img_size)
            video_tensor = video_tensor.to(device)
            print(f"Video tensor shape: {video_tensor.shape}")
            if video_tensor.shape[2] < window_size:
                print(f"Warning: Video has fewer frames ({video_tensor.shape[2]}) than window size ({window_size}); skipping.")
                error += 1
                continue

            print("Computing VJEPA surprise...")
            with torch.no_grad():
                loss = compute_vjepa_loss_sliding_window(
                    video_tensor=video_tensor,
                    encoder=encoder,
                    target_encoder=target_encoder,
                    predictor=predictor,
                    img_size=img_size,
                    window_size=window_size,
                    loss_exp=2,
                    masking_mode="causal",
                    context_frames=context_frames,
                    is_vae_output=True,
                    seed=seed,
                    stride=stride,
                    mode=mode,
                )

            surprise_score = loss.item()
            if direct_paths and not math.isfinite(surprise_score):
                raise ValueError(f"Nonfinite surprise score: {surprise_score}")
            
            print(f"\n{'='*50}")
            print(f"VJEPA Surprise Score: {surprise_score:.6f}")
            print(f"VJEPA Similarity Score: {1.0 - surprise_score:.6f}")
            print(f"{'='*50}")

            if resolved_output_path is not None:
                with resolved_output_path.open("a", encoding="utf-8", newline="") as output_file:
                    csv.writer(output_file).writerow([video_path, surprise_score])
                processed_videos.add(video_path)
            surprise_scores[video_path] = surprise_score
        except Exception as e:
            print(f"\nError processing video {video_path}: {e}")
            error += 1

    if direct_paths:
        missing = [str(path) for path in videos_paths if surprise_scores.get(str(path)) is None]
        if missing:
            raise RuntimeError(
                f"Missing finite surprise scores for {len(missing)} requested videos: "
                + ", ".join(missing)
            )
    return surprise_scores


def main():
    parser = argparse.ArgumentParser(description="Compute VJEPA surprise for videos")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video or list of videos")
    parser.add_argument(
        "--model",
        type=str,
        default="vitg",
        choices=["vith", "vitg", "vitg384", "vitgac"],
        help="VJEPA model variant",
    )
    parser.add_argument("--window_size", type=int, default=16, help="Sliding window size")
    parser.add_argument("--context_frames", type=int, default=8, help="Context frames per window")
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride")
    parser.add_argument("--eval_max", type=int, default=None, help="Maximum number of videos to process")
    parser.add_argument("--max_frames", type=int, default=49, help="Maximum number of frames to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--direct_paths", action="store_true", help="Use list entries without dataset cache rewriting; require finite scores for every entry")
    parser.add_argument("--output_path", type=str, help="Explicit surprise CSV output path")
    parser.add_argument(
        "--mode",
        type=str,
        default="mean",
        help="Reduction mode used for sliding-window surprise (mean, max, or topK-mean)",
    )
    args = parser.parse_args()

    if args.eval_max is not None and args.eval_max <= 0:
        print("Setting eval_max to None to process all videos.")
        args.eval_max = None

    video_path = Path(args.video_path)
    if video_path.suffix.lower() == ".txt":
        video_list_dir = video_path.parent
        with video_path.open("r", encoding="utf-8") as f:
            videos_paths = []
            for line in f:
                entry = line.strip()
                if not entry:
                    continue
                entry_path = Path(entry).expanduser()
                if not entry_path.is_absolute():
                    entry_path = Path(os.path.abspath(video_list_dir / entry_path))
                videos_paths.append(str(entry_path))

        output_dir = video_path.parent / "output" / "surprise" / args.model / args.mode / f"mf-{args.max_frames}_w-{args.window_size}_c-{args.context_frames}_s-{args.stride}"

        output_name = (
            f"{video_path.stem}_surprises.txt"
        )

        compute_multi_vjepa_surprise(
            videos_paths=videos_paths,
            model_name=args.model,
            window_size=args.window_size,
            context_frames=args.context_frames,
            stride=args.stride,
            seed=args.seed,
            max_frames=args.max_frames,
            output_path=args.output_path or str(output_dir / output_name),
            mode=args.mode,
            max_videos=args.eval_max,
            direct_paths=args.direct_paths,
        )
    elif args.direct_paths or args.output_path:
        compute_multi_vjepa_surprise(
            videos_paths=[args.video_path], model_name=args.model,
            window_size=args.window_size, context_frames=args.context_frames,
            stride=args.stride, seed=args.seed, max_frames=args.max_frames,
            output_path=args.output_path, mode=args.mode,
            max_videos=args.eval_max, direct_paths=True,
        )
    else:
        compute_vjepa_surprise(
            video_path=args.video_path,
            model_name=args.model,
            window_size=args.window_size,
            context_frames=args.context_frames,
            stride=args.stride,
            seed=args.seed,
            max_frames=args.max_frames,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
