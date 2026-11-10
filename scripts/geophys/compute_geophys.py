"""Run the vendored GeoPhys per-video metric on physics-eval dataset lists."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "third_party" / "GeoPhys"
# Always use the editable vendored copy, even if another GeoPhys is installed.
sys.path.insert(0, str(VENDOR_ROOT))
VIDEO_PATHS_SHUFFLE_SEED = 42


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_video_list(path, variant="original", eval_max=None):
    videos = []
    for line in path.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        video = Path(entry).expanduser()
        if variant != "original":
            if video.is_absolute():
                video = video.relative_to(REPO_ROOT)
            video = REPO_ROOT / "cache/datasets_variants" / variant / video
        elif not video.is_absolute():
            video = path.parent / video
        videos.append(video.resolve())
    # Match surprise and VBench's deterministic subset selection.
    random.Random(VIDEO_PATHS_SHUFFLE_SEED).shuffle(videos)
    if eval_max and eval_max > 0:
        videos = videos[:eval_max]
    if not videos:
        raise ValueError(f"No video paths found in {path}")
    return videos


def source_digest():
    digest = hashlib.sha256()
    for path in sorted((VENDOR_ROOT / "geophys").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(args):
    import numpy as np
    from geophys.backbones import READOUT, get_backbone
    from geophys.score import SIGNALS, list_videos, percentiles, video_signals

    layer = args.layer or READOUT[args.backbone]
    lists = [REPO_ROOT / f"{name}.txt" for name in args.dataset]
    jobs = [(name, video) for name, path in zip(args.dataset, lists)
            for video in read_video_list(path, args.variant, args.eval_max)]
    reference = []
    reference_dataset = getattr(args, "reference_dataset", None)
    if reference_dataset:
        reference = read_video_list(REPO_ROOT / f"{reference_dataset}.txt", args.variant)
    if args.reference:
        ref = args.reference.expanduser().resolve()
        if ref.is_dir():
            reference = [Path(p).resolve() for p in list_videos(str(ref))]
        elif ref.is_file():
            reference = read_video_list(ref)
        if not reference:
            raise ValueError(f"No reference videos found in {ref}")
    if len(set(v for _, v in jobs)) < 2 and not reference:
        raise ValueError("GeoPhys scores are relative: supply at least two videos or --reference.")

    settings = dict(backbone=args.backbone, layer=layer, prep=args.prep,
                    source_sha256=source_digest(), device=args.device)
    cache_dir = args.cache_dir.expanduser().resolve()
    values, errors = {}, {}
    encoder = None
    for video in dict.fromkeys([v for _, v in jobs] + reference):
        try:
            stat = video.stat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"GeoPhys input video does not exist: {video}. "
                f"Check the video list and --variant {args.variant}; "
                "generate the requested variant or use --variant original "
                "to score the original videos."
            ) from exc
        except OSError as exc:
            errors[video] = str(exc)
            continue
        identity = dict(settings, path=str(video), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cached = cache_dir / f"{key}.json"
        signals = None
        if cached.exists() and not args.force:
            try:
                signals = json.loads(cached.read_text())["signals"]
                if not all(math.isfinite(signals[k]) for k in SIGNALS):
                    signals = None
            except (ValueError, KeyError, TypeError):
                signals = None
        if signals is None:
            if encoder is None:
                # Model/dependency/authentication errors are fatal, not bad-video errors.
                encoder = get_backbone(args.backbone, args.device)
                if layer not in encoder.layers:
                    raise ValueError(f"Unknown layer {layer!r}; expected one of {encoder.layers}")
            print(f"Extracting {video}", flush=True)
            try:
                signals = video_signals(encoder, str(video), layer, args.prep, args.device)
                if signals is None:
                    raise ValueError("Video could not be decoded or has fewer than four frames")
                if not all(math.isfinite(signals[k]) for k in SIGNALS):
                    raise ValueError("Non-finite trajectory signals")
            except Exception as exc:
                errors[video] = f"{type(exc).__name__}: {exc}"
                continue
            atomic_write(cached, json.dumps(dict(identity, signals=signals), allow_nan=False))
        values[video] = signals

    # Never silently alter an explicitly requested reference population.
    bad_reference = [v for v in reference if v not in values]
    if bad_reference:
        raise RuntimeError(f"Failed reference videos: {[(str(v), errors[v]) for v in bad_reference]}")
    good = [v for _, v in jobs if v in values]
    if not good or (not reference and len(set(good)) < 2):
        raise RuntimeError(f"Insufficient valid videos for percentile scoring: {errors}")
    matrix = lambda videos: np.array([[values[v][k] for k in SIGNALS] for v in videos])
    scores = dict(zip(good, percentiles(matrix(good), matrix(reference or good)).mean(axis=1)))
    limit = args.eval_max if args.eval_max and args.eval_max > 0 else "All"
    mode = (f"ref-{reference_dataset or args.reference.stem}" if reference_dataset or args.reference
            else "combined" if len(args.dataset) > 1 else "independent")
    output_root = getattr(args, "output_root", None) or REPO_ROOT / "output/geophys" / mode
    output = args.output_dir or (output_root /
                                f"{args.variant}_n{limit}" / f"{args.backbone}_l{layer}_{args.prep}")
    output = output.expanduser().resolve()
    import io
    for name in args.dataset:
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter="\t")
        writer.writerow(["video", "geophys.score", *SIGNALS, "error"])
        for dataset, video in jobs:
            if dataset == name:
                writer.writerow([str(video), format(float(scores[video]), ".10g") if video in scores else "",
                                 *[values.get(video, {}).get(k, "") for k in SIGNALS], errors.get(video, "")])
        destination = output / f"{name}_geophys.score"
        atomic_write(destination, buffer.getvalue())
        print(f"Wrote {destination}")
    atomic_write(output / "run.json", json.dumps(dict(settings, variant=args.variant,
                 datasets=args.dataset, eval_max=args.eval_max,
                 reference=[str(v) for v in reference], videos=[str(v) for _, v in jobs],
                 errors={str(v): e for v, e in errors.items()}), indent=2))
    if errors:
        print(f"Failed to score {len(errors)} video(s); see error columns. Rerun to retry.", file=sys.stderr)
        return 1
    return 0


def require_geophys_environment():
    expected = (REPO_ROOT / ".venv" / "geophys").resolve()
    if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
        command = shlex.join([str(expected / "bin/python"), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise SystemExit(
            f"Error: GeoPhys requires its dedicated virtualenv at {expected}.\n"
            f"Current Python: {sys.executable} (environment: {sys.prefix})\n"
            f"Set up with: bash {shlex.quote(str(REPO_ROOT / 'scripts/geophys/setup.sh'))}\n"
            f"Run with: {command}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="*", default=["intphys2_possible"], help="Names of <dataset>.txt lists; multiple datasets share percentile normalization")
    parser.add_argument("--eval-max", type=int, default=None, help="Maximum videos per dataset; <=0 means all")
    parser.add_argument("--variant", default="512p_30fps")
    references = parser.add_mutually_exclusive_group()
    references.add_argument("--reference", type=Path, help="Fixed reference video folder or path list (original paths)")
    references.add_argument("--reference-dataset", help="Dataset list used in full as reference, using --variant")
    parser.add_argument("--backbone", choices=["dinov3", "dinov2", "cornet", "vonenet"], default="dinov3")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--prep", choices=["direct", "hf"], default="direct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "cache/geophys/signals")
    parser.add_argument("--output-root", type=Path, help="Root above variant and encoder output subdirectories")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true", help="Recompute cached signals")
    args = parser.parse_args()
    require_geophys_environment()
    for name in args.dataset + ([args.reference_dataset] if args.reference_dataset else []):
        if Path(name).name != name or name in (".", ".."):
            parser.error("dataset must be a name, not a path")
    if Path(args.variant).name != args.variant or args.variant in (".", ".."):
        parser.error("variant must be a name, not a path")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
