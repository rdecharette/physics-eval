#!/usr/bin/env python3
"""Aggregate surprise txt files and plot surprise distributions per dataset."""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import numpy as np
import pandas as pd


def extract_dataset_info(file_path: Path) -> tuple[str, str]:
    stem = file_path.stem
    marker = "_surprises"
    if marker not in stem:
        raise ValueError(f"File does not match expected pattern: {stem}")

    parts = stem.split("_surprises", 1)
    if len(parts) == 2:
        dataset_name, params = parts
    else:
        dataset_name = parts[0]
        params = ""

    return dataset_name, params


def read_surprises(file_path: Path, mask: list[bool] | None = None) -> list[float]:
    values: list[float] = []
    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue  # Skip header
            if mask is None or mask[i-1]:
                values.append(float(row[1]))
    return values


def find_files(search_root: Path) -> list[Path]:
    # Recursive by default so mode subfolders (e.g., output/mean) are covered.
    return sorted(search_root.glob("**/*_surprises*.txt"))


def plot_stats(dataset_labels: list[str], values_by_dataset: list[list[float]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(dataset_labels)), 5))

    boxplot = ax.boxplot(
        values_by_dataset,
        labels=dataset_labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    )

    for box in boxplot["boxes"]:
        box.set(facecolor="#9ecae1", alpha=0.8)

    for median in boxplot["medians"]:
        median.set(color="#1f1f1f", linewidth=1.5)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("VJEPA Surprise")
    ax.set_title("Dataset Surprise Distribution")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for tick in ax.get_xticklabels():
        text = tick.get_text().strip().lower()
        if text.startswith("(ours)"):
            tick.set_fontweight("bold")
        if text.startswith("(real)"):
            tick.set_color("darkgreen")

    ax.yaxis.set_minor_locator(MultipleLocator(0.025))

    ax.grid(True, axis="y", linestyle="--", alpha=0.40)
    ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.30)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search a result folder for surprise txt files, compute surprise stats per dataset, "
            "and save a boxplot."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("output/surprise"),
        help="Result folder to parse (default: output/surprise)",
    )
    parser.add_argument(
        "--sorted",
        action="store_true",
        help="Sort datasets by mean surprise in descending order before plotting",
    )
    args = parser.parse_args()

    path = args.path
    files = find_files(path)
    files = [f for f in files if not re.match(r"^newtphys_random_[0-9]+(?:_.*)?$", f.stem)]
    if not files:
        raise SystemExit(f"No matching files found under: {path}")

    dataset_rows: list[tuple[float, str, list[float]]] = []

    print(f"Found {len(files)} surprise files under: {path}")
    for file_path in files:
        # print(f"Processing {file_path}...")
        try:
            dataset, params = extract_dataset_info(file_path)
        except ValueError as e:
            print(f"{file_path} baddata")
            print(e)
            continue
        
        mask = None
        # if dataset.lower().startswith("physics-iq-verified"):
        #     dataset = f"{dataset} (Solid Mechanics)"

        #     # Filter categories
        #     physiq_dat = pd.read_csv("datasets/physics-iq-verified/descriptions_base.csv", sep=",", header=0)

        #     mask: list[bool] = []
        #     with file_path.open("r", encoding="utf-8", newline="") as f:
        #         reader = csv.reader(f)
        #         for i, row in enumerate(reader):
        #             if i == 0:
        #                 continue  # Skip header
                    
        #             fname = Path(row[0]).name
        #             fname = re.sub(r"_[0-9]+FPS_", "_", fname)
        #             fname = fname.replace("_full-videos_", "_")
        #             mask.append(physiq_dat.loc[physiq_dat["scenario"] == fname, "category"].eq("Solid Mechanics").all())

        values = read_surprises(file_path, mask=mask)

        
        if not values:
            print(f"Skipping {file_path}: no valid surprise values.")
            continue

        is_real = dataset.lower().startswith("physics-iq-verified")
        is_ours = dataset.lower().startswith("newtphys")

        num_entries = len(values)
        values_np = np.array(values)
        mean = values_np.mean()
        std = values_np.std(ddof=1)
        display_dataset = f"(real) {dataset}" if is_real else (f"(ours) {dataset}" if is_ours else dataset)
        label = f"{display_dataset} ({num_entries})\n[{params}]"

        dataset_rows.append((mean, label, values))
        print(f"{label}: n={len(values)} mean={mean:.4f} std={std:.4f}")

    if not dataset_rows:
        raise SystemExit("No valid surprise datasets found after parsing.")

    if args.sorted:
        dataset_rows.sort(key=lambda row: row[0], reverse=True)

    dataset_labels = [row[1] for row in dataset_rows]
    values_by_dataset = [row[2] for row in dataset_rows]

    output_name = "surprise_plot_sorted.png" if args.sorted else "surprise_plot.png"
    output_path = path / output_name
    plot_stats(dataset_labels, values_by_dataset, output_path)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
