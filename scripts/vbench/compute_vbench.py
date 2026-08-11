from __future__ import annotations

import argparse
import importlib.resources
import os
import random
import shutil
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from vbench import VBench

DEVICE = "cuda"

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = (REPO_ROOT / "cache" / "vbench").resolve()
DEFAULT_DIMENSION_LIST = [
  "subject_consistency",
  "background_consistency",
  "motion_smoothness",
  "dynamic_degree",
  "aesthetic_quality",
  "imaging_quality",
]
VIDEO_PATHS_SHUFFLE_SEED = 42


def parse_dimension_list(value: str | None) -> list[str]:
  if value is None:
    return list(DEFAULT_DIMENSION_LIST)

  # Accept comma-separated and/or newline-separated values.
  separators_normalized = value.replace("\n", ",")
  parsed = [item.strip() for item in separators_normalized.split(",") if item.strip()]
  return parsed if parsed else list(DEFAULT_DIMENSION_LIST)

LOCK_FILENAME = ".lock"
LOCK_POLL_INTERVAL_SECONDS = 10

FULL_INFO_PATH = importlib.resources.files("vbench").joinpath("VBench_full_info.json")
if not FULL_INFO_PATH.is_file():
  raise FileNotFoundError("Could not locate VBench_full_info.json in installed package 'vbench'.")


@contextmanager
def dataset_cache_lock(cache_dir: Path):
  cache_dir.mkdir(parents=True, exist_ok=True)
  lock_path = cache_dir / LOCK_FILENAME

  while True:
    try:
      lock_path.touch(exist_ok=False)
      break
    except FileExistsError:
      print(f"Lock exists at {lock_path}; waiting...")
      time.sleep(LOCK_POLL_INTERVAL_SECONDS)

  try:
    yield cache_dir
  finally:
    lock_path.unlink(missing_ok=True)

def resolve_path(path: Path, variant: str | None = None) -> Path:
  if variant == "original":
    pass
  else:
    path = Path(f"cache/datasets_variants/{variant}/" + str(path))

  return path


def build_video_cache(cache_dir:Path, video_list_path: Path, dataset: str, eval_max: int | None = None, variant: str = "512p_30fps", shuffle: bool = True) -> Path:
  with video_list_path.open("r", encoding="utf-8") as handle:
    video_paths = []
    for line in handle:
      entry = line.strip()
      if not entry or entry.lstrip().startswith("#"):
        continue

      entry = resolve_path(Path(entry), variant=variant)
      video_paths.append(str(entry))
  
  if not video_paths:
    raise ValueError(f"No video paths found in {video_list_path}")

  # Some datasets are ordered and might have various quality in folders. Shuffle deterministically for reproducibility.
  if shuffle:
    random.Random(VIDEO_PATHS_SHUFFLE_SEED).shuffle(video_paths)

  if eval_max is not None:
    video_paths = video_paths[:eval_max]

  for index, video_path in enumerate(video_paths):
    video_full_path = Path(video_path).expanduser()
    if not video_full_path.is_absolute():
      video_full_path = (video_list_path.parent / video_full_path).resolve()

    if not video_full_path.exists():
      print(f"Video not found: {video_full_path}")
      continue

    link_name = f"{index:06d}_{video_full_path.stem}{video_full_path.suffix}"
    link_path = cache_dir / link_name

    link_path.symlink_to(video_full_path)

  if len(video_paths) == 0:
    raise ValueError(f"No decodable videos available for dataset {dataset}")

  return cache_dir


def run_vbench(dataset: str, eval_max: int | None = None, variant: str = "512p_30fps", skip_existing: bool = True) -> None:
  video_list_path = REPO_ROOT / f"{dataset}.txt"

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  random_suffix = f"{random.SystemRandom().randint(0, 999999):06d}"
  cache_dir = CACHE_ROOT / f"{dataset}_{variant}_{timestamp}_{random_suffix}"
  cache_dir.resolve().relative_to(CACHE_ROOT)
  cache_dir.mkdir(parents=True, exist_ok=False)

  run_name = f"{variant}"
  run_name += f"_n{eval_max if eval_max is not None else 'All'}"

  dimension_list = parse_dimension_list(os.environ.get("DIMENSION_LIST"))
  print("Dimensions to run:", dimension_list)
  failed_dimensions: list[str] = []

  try:
    with dataset_cache_lock(cache_dir):
      cache_dir = build_video_cache(cache_dir, video_list_path, dataset, eval_max=eval_max, variant=variant, shuffle=True)

    for i, dimension_name in enumerate(dimension_list):
      output_dir = REPO_ROOT / "output" / "vbench" / run_name / dimension_name
      output_dir.mkdir(parents=True, exist_ok=True)

      print(f"\n\nVBench dimension {i+1}/{len(dimension_list)}: running {dataset} => {dimension_name}")
      results_json = output_dir / f"{dataset}_eval_results.json"
      if skip_existing and results_json.exists():
        print(f"  Skipping. Results already exist: {results_json}")
        continue

      print(f"  Running evaluation and saving results to: {results_json}")
      try:
        my_vbench = VBench(DEVICE, str(FULL_INFO_PATH), str(output_dir))
        my_vbench.output_path = str(output_dir)
        my_vbench.evaluate(
          videos_path=str(cache_dir),
          mode="custom_input",
          name=dataset,
          dimension_list=[dimension_name],
        )
      except Exception as exc:
        failed_dimensions.append(dimension_name)
        print(f"Dimension '{dimension_name}' failed for dataset '{dataset}': {exc}")
        traceback.print_exc()
        continue
  finally:
    shutil.rmtree(cache_dir, ignore_errors=True)

  if failed_dimensions:
    print(f"Completed with failures for dataset '{dataset}'. Failed dimensions: {failed_dimensions}")
  else:
    print(f"Completed all dimensions successfully for dataset '{dataset}'.")


def main() -> None:
  parser = argparse.ArgumentParser(description="Stage VBench inputs and run evaluation.")
  parser.add_argument("dataset", type=str, help="Dataset name; the input list is read from <dataset>.txt", nargs="?", default="intphys2_possible")
  parser.add_argument("--eval-max", type=int, default=None, help="Maximum number of videos to evaluate; use 0 for all videos")
  parser.add_argument("--variant", type=str, default="512p_30fps", help="Video variant; e.g., original or 512p_30fps. If variant is not 'original' the script will look for videos in cache/datasets_variants/<variant>/")
  args = parser.parse_args()
  
  run_vbench(args.dataset, eval_max=args.eval_max, variant=args.variant)


if __name__ == "__main__":
  main()