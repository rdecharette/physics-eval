#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -u - <<'PY'
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_WORKERS = 8
DEFAULT_MAX_SAFE_WORKERS = 8
DEFAULT_CHUNK_SIZE = 32


def is_video_fully_decodable(video_path: Path) -> bool:
	"""Return True if all frames can be decoded, otherwise False."""
	try:
		from decord import VideoReader
	except Exception as exc:
		raise RuntimeError("decord is required for decode validation") from exc

	try:
		reader = VideoReader(str(video_path))
		total_frames = len(reader)
		if total_frames == 0:
			return False

		# Decode in chunks to avoid allocating one giant batch.
		chunk_size = int(os.environ.get("DECODE_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
		if chunk_size < 1:
			chunk_size = DEFAULT_CHUNK_SIZE
		for start in range(0, total_frames, chunk_size):
			end = min(start + chunk_size, total_frames)
			indices = list(range(start, end))
			reader.get_batch(indices)
		return True
	except Exception as exc:
		print(f"Skipping corrupted video {video_path}: {exc}")
		return False


dataset_root = Path("datasets/physics-iq-verified/full-videos")
if not dataset_root.exists() or not dataset_root.is_dir():
	raise SystemExit(f"Dataset directory does not exist: {dataset_root}")

requested_workers = int(os.environ.get("WORKERS", str(DEFAULT_WORKERS)))
max_safe_workers = int(os.environ.get("MAX_SAFE_WORKERS", str(DEFAULT_MAX_SAFE_WORKERS)))
if requested_workers < 1:
	raise SystemExit("WORKERS must be >= 1")
if max_safe_workers < 1:
	raise SystemExit("MAX_SAFE_WORKERS must be >= 1")

workers = min(requested_workers, max_safe_workers)
if requested_workers > max_safe_workers:
	print(
		f"Requested WORKERS={requested_workers} exceeds MAX_SAFE_WORKERS={max_safe_workers}; "
		f"using {workers} workers to avoid OOM."
	)

extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
videos = sorted(p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)

if not videos:
	print(f"No video files found under {dataset_root}")
	raise SystemExit(0)

deleted = 0
kept = 0
decode_chunk_size = int(os.environ.get("DECODE_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
print(f"Checking {len(videos)} videos with {workers} workers (chunk_size={decode_chunk_size})...")
with ThreadPoolExecutor(max_workers=workers) as executor:
	future_to_video = {executor.submit(is_video_fully_decodable, video_path): video_path for video_path in videos}

	for idx, future in enumerate(as_completed(future_to_video), start=1):
		video_path = future_to_video[future]
		try:
			is_decodable = future.result()
		except Exception as exc:
			print(f"[{idx}/{len(videos)}] FAILED decode check: {video_path} ({exc})")
			is_decodable = False

		if is_decodable:
			kept += 1
			print(f"[{idx}/{len(videos)}] OK: {video_path}")
			continue

		try:
			video_path.unlink()
			deleted += 1
			print(f"[{idx}/{len(videos)}] DELETED corrupted: {video_path}")
		except OSError as exc:
			print(f"[{idx}/{len(videos)}] FAILED delete: {video_path} ({exc})")

print(f"Done. Kept={kept}, Deleted={deleted}, Total={len(videos)}")
PY
