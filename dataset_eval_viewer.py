#!/usr/bin/env python3
"""Serve a lightweight dataset video viewer with Surprise and VBench annotations."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import mimetypes
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote, urlencode, urlparse
import webbrowser


# Mirrors plot_vbench_stats.py so the viewer computes VBench scores the same way
# without importing plotting dependencies.
DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

SORT_METRIC_NONE = "none"
SORT_METRIC_SURPRISE = "surprise"
SORT_METRIC_VBENCH = "vbench"
SORT_METRIC_VBENCH_DIM_PREFIX = "vbench_dim:"
SORT_ORDER_ASC = "asc"
SORT_ORDER_DESC = "desc"
VIDEO_PATHS_SHUFFLE_SEED = 42
DEFAULT_VBENCH_EVAL = "512p_30fps_n500"

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dataset Eval Viewer</title>
  <style>
    :root {
      --bg: #0f1218;
      --panel: #171b24;
      --panel-2: #1f2531;
      --fg: #e7edf8;
      --muted: #9eabc1;
      --accent: #4fc3f7;
      --border: #2d3647;
      --selected: #253247;
      --pill: #233047;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
      color: var(--fg);
      background: radial-gradient(circle at 20% -10%, #1a2230 0%, var(--bg) 40%);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 3;
      background: color-mix(in hsl, var(--bg) 82%, black);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
      padding: 12px 16px;
    }
    .title {
      margin: 0;
      font-size: 1rem;
      line-height: 1.3;
      font-weight: 700;
    }
    .sub {
      margin: 4px 0 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .dataset-list {
      margin: 0;
      padding: 0;
      list-style: none;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .dataset-list a {
      display: inline-block;
      padding: 5px 9px;
      border-radius: 999px;
      border: 1px solid var(--border);
      color: var(--fg);
      text-decoration: none;
      background: #121825;
      font-size: 0.82rem;
      line-height: 1.2;
    }
    .dataset-list a:hover {
      border-color: var(--accent);
      color: #f4f8ff;
    }
    .dataset-list a.selected {
      background: var(--selected);
      border-color: #5c7395;
      font-weight: 700;
    }
    .controls {
      margin-top: 10px;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: rgba(18, 24, 37, 0.85);
    }
    .control-form {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: end;
    }
    .control-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 170px;
      font-size: 0.78rem;
      color: var(--muted);
    }
    .control-group select,
    .control-form button {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #121825;
      color: var(--fg);
      padding: 7px 9px;
      font: inherit;
    }
    .control-form button {
      min-width: 88px;
      cursor: pointer;
    }
    .control-form button:hover {
      border-color: var(--accent);
    }
    main {
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 12px;
    }
    .card {
      background: linear-gradient(180deg, var(--panel-2), var(--panel));
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: clip;
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
    }
    .video-wrap {
      position: relative;
      aspect-ratio: 16 / 9;
      background: #0b0e14;
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #0b0e14;
    }
    .badge {
      position: absolute;
      left: 8px;
      top: 8px;
      font-size: 0.72rem;
      color: #d7e3fb;
      border: 1px solid #355070;
      background: rgba(24, 35, 53, 0.75);
      padding: 3px 7px;
      border-radius: 999px;
      letter-spacing: 0.02em;
    }
    .meta {
      padding: 8px 10px 10px;
      border-top: 1px solid var(--border);
    }
    .name {
      margin: 0;
      font-size: 0.87rem;
      font-weight: 600;
      color: #dce8ff;
      word-break: break-word;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .path {
      margin: 4px 0 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.76rem;
      color: var(--muted);
      word-break: break-word;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .metrics {
      margin-top: 8px;
      display: grid;
      gap: 6px;
    }
    .metric-line {
      margin: 0;
      font-size: 0.76rem;
      line-height: 1.35;
      color: #d7e3fb;
    }
    .metric-label {
      font-weight: 600;
      color: #b7d4ff;
      margin-right: 6px;
    }
    .metric-tag {
      display: inline-block;
      margin: 2px 6px 2px 0;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid #355070;
      background: var(--pill);
      white-space: nowrap;
    }
    .details {
      margin: 6px 0 0;
      font-size: 0.76rem;
      color: #b7d4ff;
      line-height: 1.35;
    }
    .empty {
      margin: 12px;
      padding: 14px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #121827;
      color: var(--muted);
      font-size: 0.9rem;
    }
        .body-count {
            margin: 12px 12px 0;
            padding: 0 2px;
            color: #dfe9fa;
            font-size: 0.9rem;
            font-weight: 700;
        }
        .info-block {
            margin: 12px;
            padding: 12px 14px;
            border-radius: 10px;
            border: 1px solid #33506f;
            background: linear-gradient(180deg, rgba(32, 44, 64, 0.95), rgba(22, 31, 45, 0.95));
            color: #aa3333;
            font-size: 0.88rem;
            line-height: 1.45;
        }
  </style>
</head>
<body>
  <header>
    <h1 class="title">Dataset Eval Viewer</h1>
    <p class="sub">{count} videos from {listing_name}</p>
    <ul class="dataset-list">
      {dataset_links}
    </ul>
    {controls}
  </header>
  {body}
  <script>
    const pendingTimers = new Map();
    const inViewport = new Map();

    function ensureLoaded(video) {
      if (video.hasAttribute('src')) {
        return;
      }
      const dataSrc = video.getAttribute('data-src');
      if (!dataSrc) {
        return;
      }
      video.setAttribute('src', dataSrc);
      video.load();
    }

    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const video = entry.target;

        if (entry.isIntersecting) {
          inViewport.set(video, true);
          if (!pendingTimers.has(video) && !video.src) {
            const timerId = setTimeout(() => {
              const stillVisible = inViewport.get(video) === true;
              if (stillVisible) {
                ensureLoaded(video);
                observer.unobserve(video);
              }
              pendingTimers.delete(video);
            }, 200);
            pendingTimers.set(video, timerId);
          }
        } else {
          inViewport.set(video, false);
          const timerId = pendingTimers.get(video);
          if (timerId) {
            clearTimeout(timerId);
            pendingTimers.delete(video);
          }
        }
      }
    }, {
      root: null,
      rootMargin: '0px',
      threshold: 0.01,
    });

    document.querySelectorAll('video[data-src]').forEach((video) => {
      observer.observe(video);

      video.addEventListener('mouseenter', async () => {
        ensureLoaded(video);
        try {
          await video.play();
        } catch (_) {
          // Ignore play interruptions/policy issues.
        }
      });

      document.addEventListener('visibilitychange', () => {
        if (document.hidden) video.pause();
      });
    });
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class VideoMeta:
    duration_s: str
    resolution: str
    fps: str
    codec: str
    frames: str


@dataclass(frozen=True)
class SurpriseEval:
    key: str
    directory: Path


@dataclass(frozen=True)
class VBenchEval:
    key: str
    directory: Path
    dimensions: tuple[str, ...]


@dataclass(frozen=True)
class VBenchVideoMetrics:
    overall_score: float
    dimension_scores: dict[str, float]


class ViewerState:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.video_search_roots = discover_video_search_roots(root_dir)
        self.listings: dict[str, Path] = scan_root_txt_files(root_dir)
        self.video_cache: dict[str, list[Path]] = {}
        self.video_line_index_cache: dict[str, list[int]] = {}
        self.raw_listing_candidate_cache: dict[str, list[Path]] = {}
        self.shuffled_line_pos_cache: dict[str, dict[int, int]] = {}
        self.meta_cache: dict[Path, VideoMeta | None] = {}
        self.surprise_evals = scan_surprise_evals(root_dir)
        self.vbench_evals = scan_vbench_evals(root_dir)
        self.surprise_cache: dict[tuple[str, str], dict[str, float]] = {}
        self.surprise_path_cache: dict[tuple[str, str], dict[str, str]] = {}
        self.vbench_cache: dict[tuple[str, str], dict[str, VBenchVideoMetrics]] = {}
        self.vbench_path_cache: dict[tuple[str, str], dict[str, str]] = {}
        all_dimensions = {
            dimension
            for evaluation in self.vbench_evals.values()
            for dimension in evaluation.dimensions
        }
        self.dimension_abbreviations = build_dimension_abbreviations(all_dimensions)
        self.ffprobe_path = shutil.which("ffprobe")

    def available_datasets(self) -> list[str]:
        return sorted(self.listings.keys(), key=lambda x: x.lower())

    def available_surprise_evals(self) -> list[SurpriseEval]:
        return [self.surprise_evals[key] for key in sorted(self.surprise_evals, key=str.lower)]

    def available_vbench_evals(self) -> list[VBenchEval]:
        return [self.vbench_evals[key] for key in sorted(self.vbench_evals, key=str.lower)]

    def get_listing_for_dataset(self, dataset: str) -> Path | None:
        return self.listings.get(dataset)

    def get_video_paths(self, dataset: str) -> list[Path]:
        if dataset in self.video_cache:
            return self.video_cache[dataset]

        listing_file = self.get_listing_for_dataset(dataset)

        videos, line_indices, raw_candidates = parse_listing(listing_file, self.video_search_roots)
        self.video_cache[dataset] = videos
        self.video_line_index_cache[dataset] = line_indices
        self.raw_listing_candidate_cache[dataset] = raw_candidates

        shuffled_line_indices = list(range(len(raw_candidates)))
        random.Random(VIDEO_PATHS_SHUFFLE_SEED).shuffle(shuffled_line_indices)
        self.shuffled_line_pos_cache[dataset] = {
            line_idx: shuffled_idx
            for shuffled_idx, line_idx in enumerate(shuffled_line_indices)
        }
        return videos

    def get_video_line_index(self, dataset: str, video_idx: int) -> int | None:
        if dataset not in self.video_cache:
            self.get_video_paths(dataset)
        line_indices = self.video_line_index_cache.get(dataset)
        if line_indices is None or video_idx < 0 or video_idx >= len(line_indices):
            return None
        return line_indices[video_idx]

    def get_raw_listing_candidate(self, dataset: str, line_idx: int) -> Path | None:
        if dataset not in self.video_cache:
            self.get_video_paths(dataset)
        candidates = self.raw_listing_candidate_cache.get(dataset)
        if candidates is None or line_idx < 0 or line_idx >= len(candidates):
            return None
        return candidates[line_idx]

    def get_shuffled_position(self, dataset: str, line_idx: int) -> int | None:
        if dataset not in self.video_cache:
            self.get_video_paths(dataset)
        return self.shuffled_line_pos_cache.get(dataset, {}).get(line_idx)

    def get_meta(self, path: Path) -> VideoMeta | None:
        if path in self.meta_cache:
            return self.meta_cache[path]

        meta = probe_video(path, self.ffprobe_path)
        self.meta_cache[path] = meta
        return meta

    def get_surprise_scores(self, eval_key: str, dataset: str) -> dict[str, float]:
        if not eval_key or eval_key not in self.surprise_evals:
            return {}

        dataset_stem = Path(dataset).stem
        cache_key = (eval_key, dataset_stem)

        print("surprise cache key:", cache_key)
        if cache_key in self.surprise_cache:
            return self.surprise_cache[cache_key]

        eval_dir = self.surprise_evals[eval_key].directory
        candidates = sorted(eval_dir.glob(f"{dataset_stem}_surprises.txt"))
        if not candidates:
            self.surprise_cache[cache_key] = {}
            return {}

        exact_name = f"{dataset_stem}_surprises.txt"

        assert(len(candidates) == 1)
        selected_file = candidates[0]

        scores: dict[str, float] = {}
        source_paths: dict[str, str] = {}
        with selected_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row_index, row in enumerate(reader):
                if not row:
                    continue
                if row_index == 0 and row[0].strip().lower() == "video":
                    continue
                if len(row) < 2:
                    raise ValueError(f"Unexpected row in {selected_file}: {row!r}")
                source_video_path = row[0].strip()
                video_key = surprise_path_key(source_video_path)
                scores[video_key] = float(row[1])
                source_paths.setdefault(video_key, source_video_path)

        self.surprise_cache[cache_key] = scores
        self.surprise_path_cache[cache_key] = source_paths
        return scores

    def get_surprise_source_paths(self, eval_key: str, dataset: str) -> dict[str, str]:
        if not eval_key or eval_key not in self.surprise_evals:
            return {}

        dataset_stem = Path(dataset).stem
        cache_key = (eval_key, dataset_stem)
        if cache_key not in self.surprise_cache:
            self.get_surprise_scores(eval_key, dataset)
        return self.surprise_path_cache.get(cache_key, {})

    def get_vbench_scores(self, eval_key: str, dataset: str) -> dict[str, VBenchVideoMetrics]:
        if not eval_key or eval_key not in self.vbench_evals:
            return {}

        dataset_stem = Path(dataset).stem
        cache_key = (eval_key, dataset_stem)
        if cache_key in self.vbench_cache:
            return self.vbench_cache[cache_key]

        eval_info = self.vbench_evals[eval_key]
        per_video_metrics: dict[str, dict[str, float]] = {}
        per_video_source_paths: dict[str, str] = {}

        for dimension in eval_info.dimensions:
            file_path = eval_info.directory / metric_directory_name(dimension) / f"{dataset_stem}_eval_results.json"
            if not file_path.exists():
                continue

            print(f"vbench {dimension}:", file_path)
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if not isinstance(payload, dict):
                raise ValueError(f"Unexpected VBench payload in {file_path}")

            for metric_name_raw, metric_value in payload.items():
                metric_name = normalize_metric_name(metric_name_raw)
                if metric_name not in NORMALIZE_DIC or metric_name not in DIM_WEIGHT:
                    raise ValueError(f"Unexpected VBench metric in {file_path}: {metric_name}")
                if not isinstance(metric_value, list) or len(metric_value) != 2:
                    raise ValueError(f"Unexpected metric shape for {metric_name} in {file_path}")

                _, per_item = metric_value
                if not isinstance(per_item, list):
                    raise ValueError(f"Unexpected per-item payload for {metric_name} in {file_path}")

                for item in per_item:
                    if not isinstance(item, dict) or "video_path" not in item or "video_results" not in item:
                        raise ValueError(f"Unexpected VBench item in {file_path}: {item!r}")
                    raw_value = item["video_results"]
                    if isinstance(raw_value, bool):
                        numeric_value = float(int(raw_value))
                    elif isinstance(raw_value, (int, float)):
                        numeric_value = float(raw_value)
                    else:
                        raise ValueError(
                            f"Unsupported VBench value type for {metric_name} in {file_path}: {type(raw_value)!r}"
                        )

                    if metric_name == "imaging quality":
                        numeric_value /= 100.0

                    source_path = str(item["video_path"])
                    video_key = Path(source_path).name
                    per_video_metrics.setdefault(video_key, {})[metric_name] = numeric_value
                    per_video_source_paths.setdefault(video_key, source_path)

        scores = {
            video_key: VBenchVideoMetrics(
                overall_score=compute_vbench_score(dimension_scores),
                dimension_scores=dimension_scores,
            )
            for video_key, dimension_scores in per_video_metrics.items()
        }
        self.vbench_cache[cache_key] = scores
        self.vbench_path_cache[cache_key] = per_video_source_paths
        return scores

    def get_vbench_source_paths(self, eval_key: str, dataset: str) -> dict[str, str]:
        if not eval_key or eval_key not in self.vbench_evals:
            return {}

        dataset_stem = Path(dataset).stem
        cache_key = (eval_key, dataset_stem)
        if cache_key not in self.vbench_cache:
            self.get_vbench_scores(eval_key, dataset)
        return self.vbench_path_cache.get(cache_key, {})


def scan_root_txt_files(root_dir: Path) -> dict[str, Path]:
    datasets: dict[str, Path] = {}
    for txt_file in sorted(root_dir.glob("*.txt")):
        if txt_file.name.startswith("environment"):
            continue
        datasets[txt_file.name] = txt_file.resolve()
    return datasets


def scan_surprise_evals(root_dir: Path) -> dict[str, SurpriseEval]:
    output_dir = root_dir / "output"
    if not output_dir.exists():
        return {}

    evaluations: dict[str, SurpriseEval] = {}
    for file_path in sorted(output_dir.glob("surprise/**/*_surprises*.txt")):
        rel_parent = file_path.parent.relative_to(output_dir)
        if not rel_parent.parts or not rel_parent.parts[0].startswith("surprise"):
            continue
        key = rel_parent.as_posix()
        evaluations[key] = SurpriseEval(key=key, directory=file_path.parent)
    return evaluations


def scan_vbench_evals(root_dir: Path) -> dict[str, VBenchEval]:
    base_dir = root_dir / "output" / "vbench"
    if not base_dir.exists():
        return {}

    evaluations: dict[str, VBenchEval] = {}
    for eval_dir in sorted(base_dir.iterdir()):
        if not eval_dir.is_dir():
            continue
        dimensions = []
        for child in sorted(eval_dir.iterdir()):
            if not child.is_dir():
                continue
            if any(child.glob("*_eval_results.json")):
                dimensions.append(normalize_metric_name(child.name))
        if dimensions:
            evaluations[eval_dir.name] = VBenchEval(
                key=eval_dir.name,
                directory=eval_dir,
                dimensions=tuple(dimensions),
            )
    return evaluations


def discover_video_search_roots(root_dir: Path) -> tuple[Path, ...]:
    roots = [root_dir]
    for worktree_root in iter_git_worktree_roots(root_dir):
        if worktree_root not in roots:
            roots.append(worktree_root)
    return tuple(roots)


def iter_git_worktree_roots(root_dir: Path) -> list[Path]:
    cmd = ["git", "--no-pager", "worktree", "list", "--porcelain"]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=root_dir,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    roots: list[Path] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line.startswith("worktree "):
            continue
        worktree_root = Path(raw_line.split(" ", 1)[1]).resolve()
        roots.append(worktree_root)
    return roots


def parse_listing(listing_file: Path, base_dirs: Iterable[Path]) -> tuple[list[Path], list[int], list[Path]]:
    video_paths: list[Path] = []
    video_line_indices: list[int] = []
    raw_candidates: list[Path] = []
    search_roots = list(base_dirs)

    caches = list(Path("cache/datasets_variants/").glob(f"256p*")) + list(Path("cache/datasets_variants/").glob(f"512p*"))
    
    with listing_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            candidate = Path(line)
            line_idx = len(raw_candidates)
            raw_candidates.append(candidate)
            candidate_to_resolve = candidate

            # Prioritize smaller video cache versions if they exist, otherwise use the original dataset paths.
            if str(candidate_to_resolve).startswith("datasets/"):
                for c in caches:
                    candidate_cache = c / candidate_to_resolve
                    if candidate_cache.exists():
                        candidate_to_resolve = candidate_cache
                        break
            
            if candidate_to_resolve.is_absolute():
                resolved_candidates = [candidate_to_resolve]
            else:
                resolved_candidates = [(base_dir / candidate_to_resolve).resolve() for base_dir in search_roots]
                resolved_candidates.append((listing_file.parent / candidate_to_resolve).resolve())
            
            for resolved in resolved_candidates:
                if resolved.exists() and resolved.is_file():
                    video_paths.append(resolved)
                    video_line_indices.append(line_idx)
                    break

    return video_paths, video_line_indices, raw_candidates


def parse_fraction(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        left, right = value.split("/", 1)
        try:
            den = float(right)
            if den == 0:
                return None
            return float(left) / den
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def probe_video(path: Path, ffprobe_path: str | None) -> VideoMeta | None:
    if ffprobe_path is None:
        return None

    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    streams = payload.get("streams") or []
    stream0 = streams[0] if streams else {}
    fmt = payload.get("format") or {}

    width = stream0.get("width")
    height = stream0.get("height")
    resolution = "n/a" if width is None or height is None else f"{width}x{height}"

    fps_value = parse_fraction(stream0.get("avg_frame_rate"))
    if fps_value is None:
        fps_value = parse_fraction(stream0.get("r_frame_rate"))
    fps = "n/a" if fps_value is None else f"{fps_value:.3f}".rstrip("0").rstrip(".")

    codec_name = stream0.get("codec_name")
    codec = str(codec_name) if codec_name else "n/a"

    frames_raw = stream0.get("nb_frames")
    frames = str(frames_raw) if frames_raw not in (None, "") else "n/a"

    duration_raw = fmt.get("duration")
    try:
        duration = "n/a" if duration_raw is None else f"{float(duration_raw):.2f}s"
    except (TypeError, ValueError):
        duration = "n/a"

    return VideoMeta(
        duration_s=duration,
        resolution=resolution,
        fps=fps,
        codec=codec,
        frames=frames,
    )


def normalize_metric_name(value: str) -> str:
    return value.replace("_", " ").strip()


def metric_directory_name(metric_name: str) -> str:
    return metric_name.replace(" ", "_")


def canonical_video_key(path_or_name: str | Path) -> str:
    name = Path(path_or_name).name
    match = re.match(r"^\d{6}_(\d{4}_.+)$", name)
    if match:
        return match.group(1)
    # VBench cache symlinks are prefixed with a stable 6-digit index.
    # Some datasets have compact names like "0239.mp4" after this prefix.
    match = re.match(r"^\d{6}_(.+)$", name)
    if match:
        return match.group(1)
    return name


def surprise_path_key(path_or_name: str | Path) -> str:
    raw = str(path_or_name).strip().replace("\\", "/")
    parts = [part for part in raw.split("/") if part and part != "."]
    if not parts:
        return ""

    # Listings may point to datasets/... or cache/.../datasets/...; the
    # Surprise files store dataset-relative paths.
    if "datasets" in parts:
        idx = parts.index("datasets")
        parts = parts[idx + 1 :]

    return "/".join(parts)


def expected_vbench_basename_for_video(
    state: ViewerState,
    dataset_name: str,
    original_idx: int,
    rendered_path: Path,
) -> str | None:
    line_idx = state.get_video_line_index(dataset_name, original_idx)
    if line_idx is None:
        return None

    shuffled_idx = state.get_shuffled_position(dataset_name, line_idx)
    if shuffled_idx is None:
        return None

    raw_candidate = state.get_raw_listing_candidate(dataset_name, line_idx)
    stem = rendered_path.stem if raw_candidate is None else raw_candidate.stem
    suffix = rendered_path.suffix if raw_candidate is None else raw_candidate.suffix
    if not suffix:
        suffix = rendered_path.suffix or ".mp4"

    return f"{shuffled_idx:06d}_{stem}{suffix}"


def lookup_vbench_match(
    state: ViewerState,
    dataset_name: str,
    vbench_scores: dict[str, VBenchVideoMetrics],
    vbench_source_paths: dict[str, str],
    original_idx: int,
    rendered_path: Path,
) -> tuple[VBenchVideoMetrics | None, str | None]:
    expected_name = expected_vbench_basename_for_video(state, dataset_name, original_idx, rendered_path)
    if not expected_name:
        return None, None

    metrics = vbench_scores.get(expected_name)
    if metrics is not None:
        return metrics, vbench_source_paths.get(expected_name)

    expected_stem = Path(expected_name).stem
    for key, entry_metrics in vbench_scores.items():
        if Path(key).stem == expected_stem:
            return entry_metrics, vbench_source_paths.get(key)

    return None, None


def compute_vbench_score(metrics_dict: dict[str, float]) -> float:
    if not metrics_dict:
        raise ValueError("Cannot compute a VBench score without metrics")

    metrics_norm: dict[str, float] = {}
    for metric_name, value in metrics_dict.items():
        if metric_name not in NORMALIZE_DIC or metric_name not in DIM_WEIGHT:
            raise ValueError(f"Unexpected VBench metric: {metric_name}")
        min_val = NORMALIZE_DIC[metric_name]["Min"]
        max_val = NORMALIZE_DIC[metric_name]["Max"]
        metrics_norm[metric_name] = (value - min_val) / (max_val - min_val)
        metrics_norm[metric_name] *= DIM_WEIGHT[metric_name]

    return sum(metrics_norm.values()) / len(metrics_norm)


def build_dimension_abbreviations(dimensions: Iterable[str]) -> dict[str, str]:
    sorted_dimensions = sorted(set(dimensions), key=str.lower)
    by_base: dict[str, list[str]] = {}
    for dimension in sorted_dimensions:
        base = abbreviation_base(dimension)
        by_base.setdefault(base, []).append(dimension)

    abbreviations: dict[str, str] = {}
    for base, names in by_base.items():
        if len(names) == 1:
            abbreviations[names[0]] = base
            continue
        for idx, name in enumerate(sorted(names, key=str.lower), start=1):
            abbreviations[name] = f"{base}{idx}"
    return abbreviations


def abbreviation_base(dimension: str) -> str:
    tokens = [token for token in re.split(r"[\s_-]+", dimension) if token]
    if not tokens:
        return "NA"
    return "".join(token[0].upper() for token in tokens)


def selected_or_first(available_keys: Iterable[str], requested: str | None) -> str:
    keys = list(available_keys)
    if requested == "":
        return ""
    if requested and requested in keys:
        return requested
    return keys[0] if keys else ""


def select_default_surprise_eval(available_keys: Iterable[str], requested: str | None) -> str:
    keys = list(available_keys)
    if requested == "":
        return ""
    if requested and requested in keys:
        return requested

    for key in keys:
        if "topk3" in key.lower():
            return key
    return keys[0] if keys else ""


def select_default_vbench_eval(available_keys: Iterable[str], requested: str | None) -> str:
    keys = list(available_keys)
    if requested == "":
        return ""
    if requested and requested in keys:
        return requested

    if DEFAULT_VBENCH_EVAL in keys:
        return DEFAULT_VBENCH_EVAL
    return keys[0] if keys else ""


def sort_metric_for_vbench_dimension(dimension: str) -> str:
    return f"{SORT_METRIC_VBENCH_DIM_PREFIX}{dimension}"


def parse_vbench_dimension_sort_metric(sort_metric: str | None) -> str | None:
    if not sort_metric:
        return None
    if not sort_metric.startswith(SORT_METRIC_VBENCH_DIM_PREFIX):
        return None
    dimension = sort_metric[len(SORT_METRIC_VBENCH_DIM_PREFIX):]
    return dimension if dimension else None


def sanitize_sort_metric(requested: str | None, allowed_vbench_dimensions: Iterable[str]) -> str:
    if requested in {SORT_METRIC_SURPRISE, SORT_METRIC_VBENCH}:
        return requested

    requested_dimension = parse_vbench_dimension_sort_metric(requested)
    if requested_dimension and requested_dimension in set(allowed_vbench_dimensions):
        return sort_metric_for_vbench_dimension(requested_dimension)

    return SORT_METRIC_NONE


def sanitize_sort_order(requested: str | None) -> str:
    if requested == SORT_ORDER_ASC:
        return SORT_ORDER_ASC
    return SORT_ORDER_DESC


def count_surprise_matches(
    state: ViewerState,
    dataset: str,
    video_paths: list[Path],
    surprise_scores: dict[str, float],
) -> int:
    match_count = 0
    for original_idx, path in enumerate(video_paths):
        line_idx = state.get_video_line_index(dataset, original_idx)
        raw_candidate = None if line_idx is None else state.get_raw_listing_candidate(dataset, line_idx)
        surprise_key = surprise_path_key(raw_candidate if raw_candidate is not None else path)
        if surprise_key in surprise_scores:
            match_count += 1
    return match_count


def count_vbench_matches(
    state: ViewerState,
    dataset: str,
    video_paths: list[Path],
    vbench_scores: dict[str, VBenchVideoMetrics],
    vbench_source_paths: dict[str, str],
) -> int:
    match_count = 0
    for original_idx, path in enumerate(video_paths):
        metrics, _ = lookup_vbench_match(
            state,
            dataset,
            vbench_scores,
            vbench_source_paths,
            original_idx,
            path,
        )
        if metrics is not None:
            match_count += 1
    return match_count


def resolve_video_sort_metric_value(
    state: ViewerState,
    dataset_name: str,
    original_idx: int,
    path: Path,
    surprise_scores: dict[str, float],
    vbench_scores: dict[str, VBenchVideoMetrics],
    vbench_source_paths: dict[str, str],
    sort_metric: str,
) -> float | None:
    line_idx = state.get_video_line_index(dataset_name, original_idx)
    raw_candidate = None if line_idx is None else state.get_raw_listing_candidate(dataset_name, line_idx)
    surprise_key = surprise_path_key(raw_candidate if raw_candidate is not None else path)

    if sort_metric == SORT_METRIC_SURPRISE:
        return surprise_scores.get(surprise_key)

    dimension_sort = parse_vbench_dimension_sort_metric(sort_metric)
    if sort_metric == SORT_METRIC_VBENCH or dimension_sort:
        metrics, _ = lookup_vbench_match(
            state,
            dataset_name,
            vbench_scores,
            vbench_source_paths,
            original_idx,
            path,
        )
        if metrics is None:
            return None
        if sort_metric == SORT_METRIC_VBENCH:
            return metrics.overall_score
        return metrics.dimension_scores.get(dimension_sort) if dimension_sort else None

    return None


def video_has_missing_visible_metrics(
    state: ViewerState,
    dataset_name: str,
    original_idx: int,
    path: Path,
    surprise_scores: dict[str, float],
    vbench_scores: dict[str, VBenchVideoMetrics],
    vbench_source_paths: dict[str, str],
    show_surprise: bool,
    show_vbench: bool,
) -> bool:
    line_idx = state.get_video_line_index(dataset_name, original_idx)
    raw_candidate = None if line_idx is None else state.get_raw_listing_candidate(dataset_name, line_idx)
    surprise_key = surprise_path_key(raw_candidate if raw_candidate is not None else path)

    if show_surprise and surprise_scores.get(surprise_key) is None:
        return True

    if show_vbench:
        metrics, _ = lookup_vbench_match(
            state,
            dataset_name,
            vbench_scores,
            vbench_source_paths,
            original_idx,
            path,
        )
        if metrics is None:
            return True

    return False


def format_body_count(count: int, missing_count: int) -> str:
    if missing_count > 0:
        return f'<div class="body-count">{count} videos <span style="color: #aa3333;">({missing_count} videos with N/A metrics)</span></div>'
    return f'<div class="body-count">{count} videos</div>'


def build_query(
    dataset: str,
    surprise_eval: str,
    vbench_eval: str,
    sort_metric: str,
    sort_order: str,
) -> str:
    params = [("dataset", dataset)]
    if surprise_eval:
        params.append(("surprise", surprise_eval))
    if vbench_eval:
        params.append(("vbench", vbench_eval))
    if sort_metric != SORT_METRIC_NONE:
        params.append(("sort_metric", sort_metric))
    if sort_order != SORT_ORDER_DESC:
        params.append(("sort_order", sort_order))
    return urlencode(params)


def iter_dataset_links(
    dataset_names: Iterable[str],
    selected_dataset: str,
    surprise_eval: str,
    vbench_eval: str,
    sort_metric: str,
    sort_order: str,
) -> str:
    links: list[str] = []
    for name in dataset_names:
        css_class = "selected" if name == selected_dataset else ""
        query = build_query(name, surprise_eval, vbench_eval, sort_metric, sort_order)
        url = f"/?{query}"
        links.append(
            f'<li><a class="{css_class}" href="{html.escape(url)}">{html.escape(name)}</a></li>'
        )
    return "\n".join(links)


def render_controls(
    state: ViewerState,
    dataset: str,
    video_paths: list[Path],
    surprise_evals: list[SurpriseEval],
    selected_surprise: str,
    vbench_evals: list[VBenchEval],
    selected_vbench: str,
    vbench_dimensions: tuple[str, ...],
    sort_metric: str,
    sort_order: str,
) -> str:
    surprise_labels = []
    for item in surprise_evals:
        surprise_scores = state.get_surprise_scores(item.key, dataset)
        match_count = count_surprise_matches(state, dataset, video_paths, surprise_scores)
        surprise_labels.append((item.key, f"{item.key} [{match_count}]") )

    vbench_labels = []
    for item in vbench_evals:
        vbench_scores = state.get_vbench_scores(item.key, dataset)
        vbench_source_paths = state.get_vbench_source_paths(item.key, dataset)
        match_count = count_vbench_matches(state, dataset, video_paths, vbench_scores, vbench_source_paths)
        vbench_labels.append((item.key, f"{item.key} [{match_count}]") )

    surprise_options = render_options(
        [("", "No surprise annotations")] + surprise_labels,
        selected_surprise,
    )
    vbench_options = render_options(
        [("", "No VBench annotations")] + vbench_labels,
        selected_vbench,
    )
    sort_metric_items: list[tuple[str, str]] = [
        (SORT_METRIC_NONE, "Original order"),
        (SORT_METRIC_SURPRISE, "Surprise"),
        (SORT_METRIC_VBENCH, "VBench overall"),
    ]
    for dimension in vbench_dimensions:
        sort_metric_items.append(
            (sort_metric_for_vbench_dimension(dimension), f"VBench: {dimension}")
        )
    sort_metric_options = render_options(sort_metric_items, sort_metric)
    sort_order_options = render_options(
        [
            (SORT_ORDER_DESC, "Descending"),
            (SORT_ORDER_ASC, "Ascending"),
        ],
        sort_order,
    )
    return """
    <div class="controls">
      <form class="control-form" method="get">
        <input type="hidden" name="dataset" value="{dataset}" />
        <label class="control-group">
          <span>Surprise evaluation</span>
          <select name="surprise">{surprise_options}</select>
        </label>
        <label class="control-group">
          <span>VBench evaluation</span>
          <select name="vbench">{vbench_options}</select>
        </label>
        <label class="control-group">
          <span>Sort metric</span>
          <select name="sort_metric">{sort_metric_options}</select>
        </label>
        <label class="control-group">
          <span>Sort order</span>
          <select name="sort_order">{sort_order_options}</select>
        </label>
        <button type="submit">Apply</button>
      </form>
    </div>
    """.format(
        dataset=html.escape(dataset),
        surprise_options=surprise_options,
        vbench_options=vbench_options,
        sort_metric_options=sort_metric_options,
        sort_order_options=sort_order_options,
    )


def render_options(options: list[tuple[str, str]], selected_value: str) -> str:
    rendered = []
    for value, label in options:
        selected_attr = ' selected="selected"' if value == selected_value else ""
        rendered.append(
            f'<option value="{html.escape(value)}"{selected_attr}>{html.escape(label)}</option>'
        )
    return "".join(rendered)


def sort_video_entries(
    video_paths: list[Path],
    state: ViewerState,
    dataset_name: str,
    surprise_scores: dict[str, float],
    vbench_scores: dict[str, VBenchVideoMetrics],
    vbench_source_paths: dict[str, str],
    sort_metric: str,
    sort_order: str,
) -> list[tuple[int, Path]]:
    indexed_paths = list(enumerate(video_paths))
    if sort_metric == SORT_METRIC_NONE:
        return indexed_paths

    def sort_key(item: tuple[int, Path]) -> tuple[int, float, int]:
        original_idx, path = item
        value = resolve_video_sort_metric_value(
            state=state,
            dataset_name=dataset_name,
            original_idx=original_idx,
            path=path,
            surprise_scores=surprise_scores,
            vbench_scores=vbench_scores,
            vbench_source_paths=vbench_source_paths,
            sort_metric=sort_metric,
        )
        if value is None or not math.isfinite(value):
            return (1, 0.0, original_idx)
        sortable = value if sort_order == SORT_ORDER_ASC else -value
        return (0, sortable, original_idx)

    return sorted(indexed_paths, key=sort_key)


def iter_cards(
    dataset_name: str,
    ordered_videos: Iterable[tuple[int, Path]],
    base_dir: Path,
    state: ViewerState,
    surprise_scores: dict[str, float],
    surprise_source_paths: dict[str, str],
    vbench_scores: dict[str, VBenchVideoMetrics],
    vbench_source_paths: dict[str, str],
    show_surprise: bool,
    vbench_dimensions: tuple[str, ...],
) -> str:
    cards: list[str] = []
    for display_idx, (original_idx, path) in enumerate(ordered_videos):
        try:
            short = path.relative_to(base_dir)
            display_path = str(short)
        except ValueError:
            display_path = str(path)

        line_idx = state.get_video_line_index(dataset_name, original_idx)
        raw_candidate = None if line_idx is None else state.get_raw_listing_candidate(dataset_name, line_idx)
        original_path_display = str(raw_candidate) if raw_candidate is not None else str(path)
        searched_pattern = expected_vbench_basename_for_video(state, dataset_name, original_idx, path) or "NONE"

        surprise_key = surprise_path_key(raw_candidate if raw_candidate is not None else path)
        video_key = canonical_video_key(path)
        matched_surprise_path = surprise_source_paths.get(surprise_key)
        # print(
        #     "surprise match "
        #     f"| video={original_path_display} "
        #     f"| surprise_key={surprise_key or 'NONE'} "
        #     f"| result={matched_surprise_path or 'NONE'}"
        # )
        vbench_metrics, matched_vbench_path = lookup_vbench_match(
            state,
            dataset_name,
            vbench_scores,
            vbench_source_paths,
            original_idx,
            path,
        )
        print(
            "vbench match "
            f"| original={original_path_display} "
            f"| pattern={searched_pattern} "
            f"| result={matched_vbench_path or 'NONE'}"
        )
        metrics_html = render_metric_block(
            show_surprise=show_surprise,
            surprise_value=surprise_scores.get(surprise_key) if show_surprise else None,
            vbench_metrics=vbench_metrics,
            vbench_dimensions=vbench_dimensions,
            abbreviations=state.dimension_abbreviations,
        )

        details_html = ""
        if display_idx % 100 == 0:
            meta = state.get_meta(path)
            if meta is None:
                details_text = "Duration: n/a | Resolution: n/a | FPS: n/a | Codec: n/a | Frames: n/a"
            else:
                details_text = (
                    f"Duration: {meta.duration_s} | "
                    f"Resolution: {meta.resolution} | "
                    f"FPS: {meta.fps} | "
                    f"Codec: {meta.codec} | "
                    f"Frames: {meta.frames}"
                )
            details_html = f'<p class="details">{html.escape(details_text)}</p>'

        data_src = f"/video?dataset={quote(dataset_name)}&i={original_idx}"
        cards.append(
            """
            <article class="card">
              <div class="video-wrap">
                <span class="badge">#{num}</span>
                <video
                  preload="none"
                  muted
                  playsinline
                  controls
                  data-src="{data_src}"
                ></video>
              </div>
              <div class="meta">
                <p class="name">{name}</p>
                <p class="path">{path}</p>
                {metrics}
                {details}
              </div>
            </article>
            """.format(
                num=display_idx + 1,
                data_src=html.escape(data_src),
                name=html.escape(path.name),
                path=html.escape(display_path),
                metrics=metrics_html,
                details=details_html,
            )
        )

    return "\n".join(cards)


def render_metric_block(
    show_surprise: bool,
    surprise_value: float | None,
    vbench_metrics: VBenchVideoMetrics | None,
    vbench_dimensions: tuple[str, ...],
    abbreviations: dict[str, str],
) -> str:
    lines: list[str] = []
    if show_surprise:
        lines.append(
            '<p class="metric-line"><span class="metric-label">Surprise</span>'
            f"{html.escape(format_metric_value(surprise_value))}</p>"
        )
    elif not vbench_dimensions:
        return ""

    if vbench_dimensions:
        overall_value = None if vbench_metrics is None else vbench_metrics.overall_score
        lines.append(
            '<p class="metric-line"><span class="metric-label">VBench</span>'
            f"{html.escape(format_metric_value(overall_value))}</p>"
        )
        tags = []
        dimension_scores = {} if vbench_metrics is None else vbench_metrics.dimension_scores
        for dimension in vbench_dimensions:
            value = dimension_scores.get(dimension)
            label = f"{abbreviations.get(dimension, abbreviation_base(dimension))} {format_metric_value(value)}"
            tags.append(f'<span class="metric-tag">{html.escape(label)}</span>')
        lines.append(f'<p class="metric-line">{"".join(tags)}</p>')

    return f'<div class="metrics">{"".join(lines)}</div>'


def format_metric_value(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.4f}"


def content_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    return "application/octet-stream"


def parse_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    if not range_header.startswith("bytes="):
        return None

    value = range_header.split("=", 1)[1].strip()
    if "," in value:
        return None

    start_s, end_s = value.split("-", 1)
    if not start_s and not end_s:
        return None

    if start_s:
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1
    else:
        length = int(end_s)
        if length <= 0:
            return None
        start = max(0, file_size - length)
        end = file_size - 1

    if start < 0 or end < start or start >= file_size:
        return None

    end = min(end, file_size - 1)
    return start, end


def build_page(
    state: ViewerState,
    selected_dataset: str,
    requested_surprise: str | None,
    requested_vbench: str | None,
    requested_sort_metric: str | None,
    requested_sort_order: str | None,
) -> str:
    dataset_names = state.available_datasets()
    listing_file = state.get_listing_for_dataset(selected_dataset)

    surprise_evals = state.available_surprise_evals()
    vbench_evals = state.available_vbench_evals()
    selected_surprise = select_default_surprise_eval((item.key for item in surprise_evals), requested_surprise)
    selected_vbench = select_default_vbench_eval((item.key for item in vbench_evals), requested_vbench)
    vbench_dimensions = state.vbench_evals[selected_vbench].dimensions if selected_vbench else ()
    sort_metric = sanitize_sort_metric(requested_sort_metric, vbench_dimensions)
    sort_order = sanitize_sort_order(requested_sort_order)

    video_paths = state.get_video_paths(selected_dataset)
    info_block = (
        '<div class="info-block">'
        'For surprise to be defined, a minimum of (w) frames is required with w=Window size. Top-k surprise requires a minimum of k*w frames.'
        '</div>'
    )

    controls = render_controls(
        state=state,
        dataset=selected_dataset,
        video_paths=video_paths,
        surprise_evals=surprise_evals,
        selected_surprise=selected_surprise,
        vbench_evals=vbench_evals,
        selected_vbench=selected_vbench,
        vbench_dimensions=vbench_dimensions,
        sort_metric=sort_metric,
        sort_order=sort_order,
    )

    surprise_scores = state.get_surprise_scores(selected_surprise, selected_dataset) if selected_surprise else {}
    surprise_source_paths = state.get_surprise_source_paths(selected_surprise, selected_dataset) if selected_surprise else {}
    vbench_scores = state.get_vbench_scores(selected_vbench, selected_dataset) if selected_vbench else {}
    vbench_source_paths = state.get_vbench_source_paths(selected_vbench, selected_dataset) if selected_vbench else {}

    if listing_file is None:
        listing_name = "(dataset not found)"
        count = 0
        missing_count = 0
        body_header = format_body_count(count, missing_count)
        body = body_header + info_block + '<div class="empty">Selected dataset was not found.</div>'
    elif not video_paths:
        listing_name = listing_file.name
        count = 0
        missing_count = 0
        body_header = format_body_count(count, missing_count)
        body = body_header + info_block + '<div class="empty">No valid video files found in this listing.</div>'
    else:
        ordered_videos = sort_video_entries(
            video_paths=video_paths,
            state=state,
            dataset_name=selected_dataset,
            surprise_scores=surprise_scores,
            vbench_scores=vbench_scores,
            vbench_source_paths=vbench_source_paths,
            sort_metric=sort_metric,
            sort_order=sort_order,
        )
        if sort_metric != SORT_METRIC_NONE:
            ordered_videos = [
                item
                for item in ordered_videos
                if resolve_video_sort_metric_value(
                    state=state,
                    dataset_name=selected_dataset,
                    original_idx=item[0],
                    path=item[1],
                    surprise_scores=surprise_scores,
                    vbench_scores=vbench_scores,
                    vbench_source_paths=vbench_source_paths,
                    sort_metric=sort_metric,
                ) is not None
            ]
        listing_name = listing_file.name
        show_surprise = bool(selected_surprise)
        show_vbench = bool(vbench_dimensions)
        missing_count = sum(
            1
            for original_idx, path in ordered_videos
            if video_has_missing_visible_metrics(
                state=state,
                dataset_name=selected_dataset,
                original_idx=original_idx,
                path=path,
                surprise_scores=surprise_scores,
                vbench_scores=vbench_scores,
                vbench_source_paths=vbench_source_paths,
                show_surprise=show_surprise,
                show_vbench=show_vbench,
            )
        )
        cards = iter_cards(
            dataset_name=selected_dataset,
            ordered_videos=ordered_videos,
            base_dir=state.root_dir,
            state=state,
            surprise_scores=surprise_scores,
            surprise_source_paths=surprise_source_paths,
            vbench_scores=vbench_scores,
            vbench_source_paths=vbench_source_paths,
            show_surprise=show_surprise,
            vbench_dimensions=vbench_dimensions,
        )
        count = len(ordered_videos)
        body_header = format_body_count(count, missing_count)
        body = f'{body_header}{info_block}<main>{cards}</main>'

    page = HTML_TEMPLATE
    page = page.replace("{count}", str(count))
    page = page.replace("{listing_name}", html.escape(listing_name))
    page = page.replace(
        "{dataset_links}",
        iter_dataset_links(
            dataset_names=dataset_names,
            selected_dataset=selected_dataset,
            surprise_eval=selected_surprise,
            vbench_eval=selected_vbench,
            sort_metric=sort_metric,
            sort_order=sort_order,
        ),
    )
    page = page.replace("{controls}", controls)
    page = page.replace("{body}", body)
    return page


def make_handler(state: ViewerState, default_dataset: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DatasetEvalViewer/0.1"

        def do_GET(self) -> None:  # noqa: N802
            self._handle_request(include_body=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._handle_request(include_body=False)

        def _handle_request(self, include_body: bool) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                params = parse_qs(parsed.query, keep_blank_values=True)
                selected = params.get("dataset", [default_dataset])[0]
                if selected not in state.listings:
                    selected = default_dataset
                surprise_eval = params.get("surprise", [None])[0]
                vbench_eval = params.get("vbench", [None])[0]
                sort_metric = params.get("sort_metric", [None])[0]
                sort_order = params.get("sort_order", [None])[0]
                self._send_html(
                    build_page(
                        state=state,
                        selected_dataset=selected,
                        requested_surprise=surprise_eval,
                        requested_vbench=vbench_eval,
                        requested_sort_metric=sort_metric,
                        requested_sort_order=sort_order,
                    ),
                    include_body=include_body,
                )
                return

            if parsed.path == "/video":
                params = parse_qs(parsed.query)
                dataset_values = params.get("dataset")
                idx_values = params.get("i")
                if not dataset_values or not idx_values:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Missing dataset or video index")
                    return

                dataset = dataset_values[0]
                if dataset not in state.listings:
                    self.send_error(HTTPStatus.NOT_FOUND, "Dataset not found")
                    return

                try:
                    idx = int(idx_values[0])
                except ValueError:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid video index")
                    return

                videos = state.get_video_paths(dataset)
                # print(f"Serving video {idx} from dataset {dataset}: {videos[idx] if 0 <= idx < len(videos) else 'N/A'}")
                if idx < 0 or idx >= len(videos):
                    self.send_error(HTTPStatus.NOT_FOUND, "Video index out of range")
                    return

                self._send_video(videos[idx], include_body=include_body)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, fmt: str, *args) -> None:
            return

        def _send_html(self, body: str, include_body: bool) -> None:
            content = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            if include_body:
                self.wfile.write(content)

        def _send_video(self, path: Path, include_body: bool) -> None:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return

            content_type = content_type_for(path)
            range_header = self.headers.get("Range")
            byte_range = parse_range(range_header, size) if range_header else None

            if byte_range is None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if include_body:
                    with path.open("rb") as f:
                        self._copy_file(f, self.wfile)
                return

            start, end = byte_range
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            if include_body:
                with path.open("rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

        @staticmethod
        def _copy_file(src, dst) -> None:
            while True:
                buf = src.read(1024 * 1024)
                if not buf:
                    break
                dst.write(buf)

    return Handler


def choose_default_dataset(state: ViewerState, requested: str | None) -> str:
    names = state.available_datasets()
    if not names:
        raise SystemExit("No root .txt files found in current directory.")

    if requested is None:
        return names[0]

    req_name = Path(requested).name
    if req_name in state.listings:
        return req_name

    if requested in state.listings:
        return requested

    return names[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a local dataset evaluation video viewer")
    parser.add_argument(
        "listing",
        nargs="?",
        help="Optional initial listing file name (e.g., contphy.txt)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8890, help="Port to bind (default: 8890)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    root_dir = Path.cwd().resolve()
    state = ViewerState(root_dir)
    default_dataset = choose_default_dataset(state, args.listing)

    handler = make_handler(state, default_dataset)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/?dataset={quote(default_dataset)}"

    print(f"Found {len(state.listings)} root txt files.")
    print("Available datasets:")
    for name in state.available_datasets():
        print(f"  - {name}")
    print("Video search roots:")
    for path in state.video_search_roots:
        print(f"  - {path}")
    print(f"Found {len(state.available_surprise_evals())} surprise evaluation folders.")
    print(f"Found {len(state.available_vbench_evals())} VBench evaluation folders.")
    print(f"Default dataset: {default_dataset}")
    print(f"Viewer URL: {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
