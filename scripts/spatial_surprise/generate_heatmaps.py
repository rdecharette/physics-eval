#!/usr/bin/env python3
"""Aggregate masked-video scores into coverage-normalized spatial heatmaps."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .generate_masked_videos import ROOT, load_masks
    from .verify_scores import verify
except ImportError:
    from generate_masked_videos import ROOT, load_masks
    from verify_scores import verify


SCORE_RELATIVE = Path("scores/vith/mean/mf-150_w-16_c-8_s-8/surprises.csv")


def aggregate(masks, scores):
    """Use binary membership, not the PNG's 255 magnitude, as the weight."""
    if not masks or len(masks) != len(scores):
        raise ValueError("Expected one score for every mask")
    total = np.zeros(np.asarray(masks[0]).shape, dtype=np.float64)
    coverage = np.zeros(total.shape, dtype=np.uint32)
    for mask, score in zip(masks, scores):
        membership = np.asarray(mask) != 0
        if membership.shape != total.shape or not np.isfinite(score):
            raise ValueError("Inconsistent mask dimensions or nonfinite score")
        total += membership * score
        coverage += membership
    if np.any(coverage == 0):
        raise ValueError("Cannot compute a heatmap with uncovered pixels")
    return total / coverage, coverage


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(tag, mask_root, variants_root, output_root, score_relative):
    if Path(tag).name != tag or tag in (".", ".."):
        raise ValueError("Expected a single configuration directory name")
    mask_dir = mask_root / tag
    mask_manifest, masks = load_masks(mask_dir)
    variants_dir = (variants_root / tag).resolve()
    video_list = variants_dir / "videos.txt"
    score_path = output_root / tag / score_relative
    verify(video_list, score_path)
    with score_path.open(newline="") as f:
        scores = {r["video"]: float(r["surprise"]) for r in csv.DictReader(f)}
    groups = {}
    for video, score in scores.items():
        path = Path(video)
        if not path.is_absolute():
            raise ValueError(f"Expected an absolute masked-video path: {video}")
        relative = path.relative_to(variants_dir)
        if ".." in relative.parts or relative.parts[0] != "datasets":
            raise ValueError(f"Unexpected variant path: {video}")
        groups.setdefault(relative.parent, {})[path.name] = score
    expected_names = {f"masked_{Path(r['filename']).stem}.mp4" for r, _ in masks}
    prepared = []
    for relative, values in sorted(groups.items()):
        if set(values) != expected_names:
            raise ValueError(f"Missing or unexpected masks for {relative}")
        source_manifest_path = variants_dir / relative / "manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text())
        if source_manifest["dataset_path"] != relative.as_posix():
            raise ValueError(f"Source manifest dataset mismatch: {relative}")
        records = source_manifest["variants"]
        expected = {str(variants_dir / relative / f"masked_{Path(r['filename']).stem}.mp4"):
                    (r["filename"], r["bounds_pixels"]) for r, _ in masks}
        actual = {r["video"]: (r["mask"], r["bounds_pixels"]) for r in records}
        if len(actual) != len(records) or actual != expected:
            raise ValueError(f"Variant manifest disagrees with masks: {relative}")
        ordered_scores = [values[f"masked_{Path(r['filename']).stem}.mp4"] for r, _ in masks]
        heatmap, coverage = aggregate([mask for _, mask in masks], ordered_scores)
        metadata = {
            "schema_version": 1, "tag": tag, "dataset_path": relative.as_posix(),
            "formula": "sum_k(binary_mask_k * surprise_k) / sum_k(binary_mask_k)",
            "mask_count": len(masks), "height": mask_manifest["height"],
            "width": mask_manifest["width"], "scale": mask_manifest["scale"],
            "overlap": mask_manifest["overlap"],
            "minimum": float(heatmap.min()), "maximum": float(heatmap.max()),
            "mean": float(heatmap.mean()), "coverage_minimum": int(coverage.min()),
            "coverage_maximum": int(coverage.max()),
            "inputs": {str(p.resolve()): digest(p) for p in
                       (mask_dir / "manifest.json", source_manifest_path, video_list, score_path)},
            "raw_map": "map.npy", "coverage": "coverage.npy", "visualization": "map.png",
        }
        prepared.append((output_root / tag / relative, heatmap, coverage, metadata))
    return prepared


def generate(args):
    if not args.tags or len(set(args.tags)) != len(args.tags):
        raise ValueError("Expected unique configuration tags")
    relative = args.score_relative
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Score path must be relative to each configuration output directory")
    prepared = []
    for tag in args.tags:
        prepared.extend(prepare(tag, args.mask_root, args.variants_root,
                                args.output_root, relative))
    # Validate every input before writing any results. One scale for all maps
    # in this invocation lets the viewer compare sources and configurations.
    low = min(h.min() for _, h, _, _ in prepared)
    high = max(h.max() for _, h, _, _ in prepared)
    for directory, _, _, _ in prepared:
        for name in ("map.npy", "coverage.npy", "map.png", "map.json", "legend.png"):
            if (directory / name).exists() and not args.overwrite:
                raise ValueError(f"Output exists (use --overwrite to regenerate): {directory / name}")
    from matplotlib import colormaps
    from PIL import ImageDraw
    cmap = colormaps["viridis"]
    legend = Image.new("RGB", (320, 60), "white")
    ramp = np.repeat(np.linspace(0, 1, 300)[None, :], 18, axis=0)
    legend.paste(Image.fromarray(cmap(ramp, bytes=True)[..., :3]), (10, 8))
    draw = ImageDraw.Draw(legend)
    draw.text((10, 34), f"{low:.6f}  low surprise", fill="black")
    draw.text((190, 34), f"{high:.6f}  high", fill="black")
    for directory, heatmap, coverage, metadata in prepared:
        normalized = (heatmap - low) / (high - low) if high > low else np.full_like(heatmap, 0.5)
        rgb = cmap(np.clip(normalized, 0, 1), bytes=True)[..., :3]
        metadata["display"] = {
            "colormap": "viridis", "vmin": float(low), "vmax": float(high),
            "scope": "all videos across tags in this invocation", "tags": args.tags,
            "constant_map_color": "colormap midpoint when vmin equals vmax",
            "legend": "legend.png",
        }
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "map.npy", heatmap, allow_pickle=False)
        np.save(directory / "coverage.npy", coverage, allow_pickle=False)
        Image.fromarray(rgb).save(directory / "map.png")
        legend.save(directory / "legend.png")
        (directory / "map.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {directory / 'map.png'}; range {heatmap.min():.6f}–{heatmap.max():.6f}")
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["s4-v0.5"])
    parser.add_argument("--mask-root", type=Path, default=ROOT / "data/mask")
    parser.add_argument("--variants-root", type=Path, default=ROOT / "cache/datasets_variants/masked")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/spatial-surprise")
    parser.add_argument("--score-relative", type=Path, default=SCORE_RELATIVE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        generate(args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
