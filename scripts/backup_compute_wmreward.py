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
import builtins
import copy
import csv
from functools import partial
from pathlib import Path

import torch
import torchvision.transforms.functional as F
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

    if model_name == "vith":
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_huge")
    elif model_name == "vitg":
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant")
    elif model_name == "vitg384":
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant_384")
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
    max_videos: int = -1
):
    """Compute VJEPA surprise score for multiple videos."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading VJEPA model: {model_name}...")
    encoder, target_encoder, predictor, img_size = load_vjepa_models(model_name)
    encoder = encoder.to(device).eval()
    target_encoder = target_encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    processed_videos = set()
    resolved_output_path = None
    if output_path is not None:
        resolved_output_path = Path(output_path)

        if resolved_output_path.exists() and not force_recompute:
            with resolved_output_path.open("r", encoding="utf-8", newline="") as output_file:
                reader = csv.reader(output_file)
                for row in reader:
                    if not row or row[0] == "video":
                        continue
                    processed_videos.add(row[0])
            print(
                f"Found existing surprise file at {resolved_output_path}; "
                f"loaded {len(processed_videos)} processed videos."
            )
        else:
            print(f"Creating surprise file at {resolved_output_path}")
            with resolved_output_path.open("w", encoding="utf-8", newline="") as output_file:
                csv.writer(output_file).writerow(["video", "surprise"])

    surprise_scores = {}
    max_videos = min(max_videos, len(videos_paths)) if max_videos != -1 else len(videos_paths)
    error = 0
    for i, video_path in enumerate(videos_paths):
        if (i-error) >= max_videos:
            print(f"Reached maximum number of videos to process: {max_videos}. Stopping.")
            break
        
        video_path = video_path
        # Retrieve the vjepa-ready video cache (256x256 at ~30 FPS)
        if not video_path.startswith("/cache/"):
            video_path = video_path.replace("/nfs/data/workspaces/rdechare/codes/physics-eval/../physics-sim/output/sims/v4_bis/", "/nfs/data/workspaces/rdechare/codes/physics-eval/cache/datasets_vjepa-ready/newtphys/")
            video_path = video_path.replace("/nfs/data/workspaces/rdechare/codes/physics-eval/datasets/", "/nfs/data/workspaces/rdechare/codes/physics-eval/cache/datasets_vjepa-ready/")
        else:
            raise ValueError(f"Video path {video_path} does not contain '/datasets/' and cannot be cached. Need to implement the 256x256 conversion at approx 4 FPS")
        
        print(f"\nProcessing video {i + 1}/{len(videos_paths)}: {video_path}")

        if not force_recompute and video_path in processed_videos:
            print(f"Skipping already processed video")
            continue
        
        surprise_scores[video_path] = None
        try:
            print(f"Loading video...")
            video_tensor = load_video_as_tensor(video_path, max_frames=max_frames, img_size=img_size)
            video_tensor = video_tensor.to(device)
            print(f"Video tensor shape: {video_tensor.shape}")
            if video_tensor.shape[2] < window_size:
                print(f"Video has fewer frames ({video_tensor.shape[2]}) than window size ({window_size}); skipping.")
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
            surprise_scores[video_path] = surprise_score
            
            print(f"\n{'='*50}")
            print(f"VJEPA Surprise Score: {surprise_score:.6f}")
            print(f"VJEPA Similarity Score: {1.0 - surprise_score:.6f}")
            print(f"{'='*50}")

            if resolved_output_path is not None:
                with resolved_output_path.open("a", encoding="utf-8", newline="") as output_file:
                    csv.writer(output_file).writerow([video_path, surprise_score])
                processed_videos.add(video_path)
        except Exception as e:
            print(f"\nError processing video {video_path}: {e}")
            error += 1


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
    parser.add_argument("--max_frames", type=int, default=49, help="Maximum number of frames to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mode",
        type=str,
        default="mean",
        help="Reduction mode used for sliding-window surprise (mean, max, or topK-mean)",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if video_path.suffix.lower() == ".txt":
        with video_path.open("r", encoding="utf-8") as f:
            videos_paths = [line.strip() for line in f if line.strip()]

        output_dir = video_path.parent / "output" / "surprise" / args.model / args.mode / f"mf-{args.max_frames}_w-{args.window_size}_c-{args.context_frames}_s-{args.stride}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # output_name = (
        #     f"{video_path.stem}_surprises_"
        #     f"m-{args.mode}_mf-{args.max_frames}_"
        #     f"w-{args.window_size}_c-{args.context_frames}_s-{args.stride}.txt"
        # )
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
            output_path=str(output_dir / output_name),
            mode=args.mode,
            max_videos=250
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
