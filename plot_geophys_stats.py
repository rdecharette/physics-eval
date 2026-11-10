#!/usr/bin/env python3
"""Plot per-dataset GeoPhys score distributions, separately for each configuration."""

import argparse
import csv
from fnmatch import fnmatchcase
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def read_scores(path: Path) -> tuple[list[float], int]:
    values = []
    skipped = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "geophys.score" not in reader.fieldnames:
            raise ValueError(f"Missing geophys.score column in {path}")
        for row in reader:
            try:
                score = float(row["geophys.score"])
            except (ValueError, TypeError):
                skipped += 1
                continue
            if row.get("error") or not math.isfinite(score) or not 0 <= score <= 1:
                skipped += 1
                continue
            values.append(score)
    return values, skipped


def plot_directory(directory: Path, files: list[Path], sort: bool) -> bool:
    rows = []
    print(f"\nConfiguration: {directory}")
    for path in files:
        dataset = path.name.removesuffix("_geophys.score")
        values, skipped = read_scores(path)
        if not values:
            print(f"Skipping {dataset}: no valid scores ({skipped} invalid rows)")
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else float("nan")
        print(f"{dataset}: n={len(values)}, skipped={skipped}, mean={mean:.6f}, "
              f"std={std:.6f}, median={statistics.median(values):.6f}")
        rows.append((mean, dataset, values))
    if not rows:
        return False
    if sort:
        rows.sort(key=lambda row: (row[0], row[1]))
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(rows)), 6))
    boxes = ax.boxplot([row[2] for row in rows], patch_artist=True, showmeans=True,
                      meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
    for box in boxes["boxes"]:
        box.set(facecolor="#9ecae1", alpha=0.8)
    ax.set_xticks(range(1, len(rows) + 1))
    ax.set_xticklabels([f"{name} (n={len(values)})\nmean={mean:.4f}" for mean, name, values in rows], rotation=45, ha="right")
    for tick, (_, name, _) in zip(ax.get_xticklabels(), rows):
        if name.lower().startswith("newtphys"):
            tick.set_fontweight("bold")
        if name.lower().startswith("physics-iq-verified"):
            tick.set_color("darkgreen")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("GeoPhys score (lower = more plausible relative to reference)")
    configuration = " / ".join(directory.parts[-3:])
    ax.set_title(f"GeoPhys score distributions\n{configuration}")
    ax.set_ylim(0, 1)
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.text(0.5, 0.01, "Scores are relative to the normalization population; independent runs need not be comparable.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output = directory / ("geophys_plot_sorted.png" if sort else "geophys_plot.png")
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {output}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("output/geophys"),
                        help="Results root or configuration directory (searched recursively)")
    parser.add_argument("--sorted", action="store_true", help="Sort datasets by ascending mean score")
    parser.add_argument("--mode", default="*", help="Scoring mode glob (e.g. combined or ref-pisabench)")
    parser.add_argument("--format", default="*", help="Video variant/limit directory glob (e.g. 512p_30fps_n*)")
    parser.add_argument("--backbone", default="*", help="Encoder directory glob (e.g. dinov3_*)")
    args = parser.parse_args()
    groups = {}
    for path in sorted(args.path.rglob("*_geophys.score")):
        directory = path.parent
        if not (fnmatchcase(directory.parent.parent.name, args.mode)
                and fnmatchcase(directory.parent.name, args.format)
                and fnmatchcase(directory.name, args.backbone)):
            continue
        groups.setdefault(directory, []).append(path)
    if not groups:
        raise SystemExit(f"No GeoPhys score files matching the filters found under: {args.path}")
    plotted = 0
    for directory, files in groups.items():
        plotted += plot_directory(directory, files, args.sorted)
    if not plotted:
        raise SystemExit("No valid GeoPhys scores found.")


if __name__ == "__main__":
    main()
