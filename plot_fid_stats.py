#!/usr/bin/env python3
"""Aggregate FID json files and plot FID distributions per dataset."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def extract_dataset_info(file_path: Path) -> tuple[str, str]:
    return file_path.stem, file_path.parent.name


def read_fid_info(file_path: Path) -> tuple[float, int, int]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fid = float(data["fid"])
    num_frames = int(data["num_frames"])
    num_videos = int(data["num_videos"])
    return fid, num_frames, num_videos


def find_files(search_root: Path) -> list[Path]:
    return sorted(search_root.glob("**/*.json"))


def plot_stats(dataset_labels: list[str], values_by_dataset: list[list[float]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(dataset_labels)), 5))
    x_positions = list(range(1, len(dataset_labels) + 1))
    fid_values = [values[0] for values in values_by_dataset]
    ax.plot(x_positions, fid_values, linestyle="", marker="o", markersize=8, color="#1f77b4")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dataset_labels)
    
    
    
    ax.set_xlabel("Dataset")
    ax.set_ylabel(output.parent.parent.parent.name.upper() + " score")
    ax.set_title(output.parent.name)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for tick in ax.get_xticklabels():
        is_real = tick.get_text().strip().lower().startswith("physics-iq-verified")
        is_ours = tick.get_text().strip().lower().startswith("newtphys")
        if is_ours:
            tick.set_fontweight("bold")
        if is_real:
            tick.set_color("darkgreen")

    ax.yaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(True, axis="y", linestyle="--", alpha=0.40)
    ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.30)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search a result folder for FID json files, compute FID stats per dataset, and save a boxplot."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("output/fid"),
        help="Result folder to parse (default: output/fid)",
    )
    parser.add_argument(
        "--sorted",
        action="store_true",
        help="Sort datasets by mean FID in ascending order before plotting",
    )
    args = parser.parse_args()

    path = args.path
    files = find_files(path)
    if not files:
        raise SystemExit(f"No matching files found under: {path}")

    dataset_rows: list[tuple[float, str, list[float]]] = []

    print(f"\n\nFound {len(files)} FID files under: {path}")
    for file_path in files:
        # print(f"Processing {file_path}...")
        try:
            dataset, stats_name = extract_dataset_info(file_path)
        except ValueError:
            continue

        try:
            fid, num_frames, num_videos = read_fid_info(file_path)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"Skipping {file_path}: {exc}")
            continue

        label = f"{dataset} (f={num_frames}, v={num_videos})"
        dataset_rows.append((fid, label, [fid]))
        print(f"{label} [{stats_name}]: fid={fid:.4f}")

    if not dataset_rows:
        raise SystemExit("No valid FID datasets found after parsing.")

    if args.sorted:
        dataset_rows.sort(key=lambda row: row[0])

    dataset_labels = [row[1] for row in dataset_rows]
    values_by_dataset = [row[2] for row in dataset_rows]

    output_name = "plot_sorted.png" if args.sorted else "plot.png"
    output_path = path / output_name
    plot_stats(dataset_labels, values_by_dataset, output_path)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()