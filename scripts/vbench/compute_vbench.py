from __future__ import annotations

import argparse
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from vbench import VBench

DEVICE = "cuda"

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = (REPO_ROOT / "cache" / "vbench").resolve()
FULL_INFO_PATH = Path(
  "/nfs/data/workspaces/rdechare/miniconda3/envs/wmreward-score/lib/python3.10/site-packages/vbench/VBench_full_info.json"
)
DEFAULT_DIMENSION_LIST = [
  "subject_consistency",
  # "background_consistency",
  "motion_smoothness",
  "dynamic_degree",
  "aesthetic_quality",
  "imaging_quality",
]

LOCK_FILENAME = ".lock"
LOCK_POLL_INTERVAL_SECONDS = 10

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

def resolve_path(path: Path, format: str | None = None) -> Path:
  if format == "vjepa":
    if not str(path).startswith("/cache/"):
      path = Path(str(path).replace("/nfs/data/workspaces/rdechare/codes/physics-eval/../physics-sim/output/sims/v4_bis/", "/nfs/data/workspaces/rdechare/codes/physics-eval/cache/datasets_vjepa-ready/newtphys/"))
      path = Path(str(path).replace("/nfs/data/workspaces/rdechare/codes/physics-eval/datasets/", "/nfs/data/workspaces/rdechare/codes/physics-eval/cache/datasets_vjepa-ready/"))
    else:
      raise ValueError(f"Video path {path} does not contain '/datasets/' and cannot be cached. Need to implement the 256x256 conversion")
  elif format == "original":
    pass

  return path


def build_video_cache(video_list_path: Path, dataset: str, eval_max: int | None = None, format: str | None = None) -> Path:
  cache_dir = CACHE_ROOT / dataset
  try:
    cache_dir.resolve().relative_to(CACHE_ROOT)
  except ValueError as exc:
    raise ValueError(f"Cache directory must stay under {CACHE_ROOT}: {cache_dir}") from exc

  cache_dir.mkdir(parents=True, exist_ok=False)

  with video_list_path.open("r", encoding="utf-8") as handle:
    video_paths = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]

  if not video_paths:
    raise ValueError(f"No video paths found in {video_list_path}")

  if eval_max is not None and eval_max > 0:
    video_paths = video_paths[:eval_max]

  for index, video_path in enumerate(video_paths, start=1):
    source_path = Path(video_path)
    if not source_path.is_absolute():
      source_path = (REPO_ROOT / source_path).resolve()

    source_path = resolve_path(source_path, format=format)
    if not source_path.exists():
      raise FileNotFoundError(f"Video not found: {source_path}")

    link_name = f"{index:06d}_{source_path.stem}{source_path.suffix}"
    link_path = cache_dir / link_name
    link_path.symlink_to(source_path)

  return cache_dir


def run_vbench(dataset: str, eval_max: int | None = None, format: str | None = None) -> None:
  video_list_path = REPO_ROOT / f"{dataset}.txt"
  cache_dir = REPO_ROOT / "cache" / "vbench" / dataset

  with dataset_cache_lock(cache_dir):
    shutil.rmtree(cache_dir)
    cache_dir = build_video_cache(video_list_path, dataset, eval_max=eval_max, format=format)

  dims_str = "_".join(["".join([s[0] for s in d.split("_")]) for d in DEFAULT_DIMENSION_LIST])

  output_dir = REPO_ROOT / "output" / "vbench" / f"{format}" / dims_str
  output_dir.mkdir(parents=True, exist_ok=True)

  my_vbench = VBench(DEVICE, str(FULL_INFO_PATH), str(output_dir))
  my_vbench.output_path = str(output_dir)
  my_vbench.evaluate(
    videos_path=str(cache_dir),
    mode="custom_input",
    name=dataset,
    dimension_list=DEFAULT_DIMENSION_LIST,
  )

  shutil.rmtree(cache_dir)


def parse_eval_max(value: str | None) -> int | None:
  if value is None:
    return None

  parsed = int(value)
  return None if parsed == 0 else parsed


def main() -> None:
  parser = argparse.ArgumentParser(description="Stage VBench inputs and run evaluation.")
  parser.add_argument("dataset", type=str, help="Dataset name; the input list is read from <dataset>.txt", nargs="?", default="intphys2_possible")
  parser.add_argument("--eval-max", type=str, default=None, help="Maximum number of videos to evaluate; use 0 for all videos")
  parser.add_argument("--format", type=str, default=None, help="Video format; e.g., vjepa (ie, 256 pixels) or original")
  args = parser.parse_args()

  env_eval_max = os.environ.get("EVAL_MAX")
  eval_max = parse_eval_max(args.eval_max if args.eval_max is not None else env_eval_max)

  run_vbench(args.dataset, eval_max=eval_max, format=args.format)


if __name__ == "__main__":
  main()