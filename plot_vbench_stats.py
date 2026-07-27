#!/usr/bin/env python3
"""Aggregate VBench evaluation JSON files and plot a radar chart of mean metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_vbench_results(file_path: Path) -> dict[str, tuple[float, list[float]]]:
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    parsed: dict[str, tuple[float, list[float]]] = {}
    for metric_name, metric_value in payload.items():
        if not isinstance(metric_value, list) or len(metric_value) != 2:
            raise ValueError(f"Unexpected metric shape for {metric_name} in {file_path}")

        summary_value, per_item = metric_value
        values: list[float] = []
        for item in per_item:
            if not isinstance(item, dict) or "video_results" not in item:
                raise ValueError(f"Unexpected per-item entry for {metric_name} in {file_path}")
            raw_value = item["video_results"]
            if isinstance(raw_value, bool):
                values.append(float(int(raw_value)))
            elif isinstance(raw_value, (int, float)):
                values.append(float(raw_value))
            else:
                raise ValueError(f"Unsupported metric payload type for {metric_name}: {type(raw_value)!r}")

        if not values:
            raise ValueError(f"No per-item values found for {metric_name} in {file_path}")

        parsed[metric_name] = (float(summary_value), values)

    return parsed


def collect_metric_stats(search_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    files = sorted(search_root.glob("*_eval_results.json"))
    if not files:
        raise FileNotFoundError(f"No *_eval_results.json files found under {search_root}")

    dataset_stats: list[dict[str, Any]] = []
    metric_names: list[str] | None = None

    for file_path in files:
        dataset = file_path.stem.replace("_eval_results", "")
        parsed = parse_vbench_results(file_path)
        metric_names_for_file = sorted(parsed)
        if metric_names is None:
            metric_names = metric_names_for_file
        else:
            metric_names = [name for name in metric_names if name in metric_names_for_file]

        dataset_metrics: dict[str, dict[str, float]] = {}
        for metric_name, (summary_value, values) in parsed.items():
            if metric_name not in metric_names:
                continue
            values_array = np.array(values, dtype=float)
            if metric_name == "imaging_quality":
                values_array /= 100.

            mean_value = float(values_array.mean())
            std_value = float(values_array.std(ddof=1)) if len(values_array) > 1 else 0.0
            dataset_metrics[metric_name] = {
                "mean": mean_value,
                "std": std_value,
                "summary": summary_value,
            }
            print(f"{dataset}, {metric_name}: mean={mean_value:.3f}, std={std_value:.3f}")

        dataset_stats.append(
            {
                "name": dataset,
                "metrics": dataset_metrics,
            }
        )

    if metric_names is None or not metric_names:
        raise ValueError(f"No usable metrics found in {search_root}")

    return dataset_stats, metric_names


def normalize_metric_values(dataset_stats: list[dict[str, Any]], metric_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(dataset_stats), len(metric_names)), dtype=float)
    stds = np.zeros((len(dataset_stats), len(metric_names)), dtype=float)

    for dataset_idx, dataset in enumerate(dataset_stats):
        for metric_idx, metric_name in enumerate(metric_names):
            if metric_name in dataset["metrics"]:
                values[dataset_idx, metric_idx] = dataset["metrics"][metric_name]["mean"]
                stds[dataset_idx, metric_idx] = dataset["metrics"][metric_name]["std"]

    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    ranges = maxs - mins
    normalized = np.zeros_like(values)
    for idx, span in enumerate(ranges):
        if span > 0:
            normalized[:, idx] = (values[:, idx] - mins[idx]) / span
        else:
            normalized[:, idx] = 0.0

    return normalized, stds


def print_dataset_metrics(dataset_stats: list[dict[str, Any]], metric_names: list[str]) -> None:
    for dataset in dataset_stats:
        mean = np.array([dataset["metrics"][name]["mean"] for name in metric_names if name in dataset["metrics"]], dtype=float)
        std = np.array([dataset["metrics"][name]["std"] for name in metric_names if name in dataset["metrics"]], dtype=float)
        print(f"{dataset['name']}: mean={mean.mean():.3f}, std={std.mean():.3f}")
        for metric_name in metric_names:
            metric = dataset["metrics"].get(metric_name)
            if metric is None:
                continue
            print(f"  {metric_name}: {metric['mean']:.3f} ({metric['std']:.3f})")
        print()


def plot_radar_chart(dataset_stats: list[dict[str, Any]], metric_names: list[str], output_path: Path) -> None:
    normalized, stds = normalize_metric_values(dataset_stats, metric_names)

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.grid(color="0.7", linestyle="--", linewidth=0.8)

    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_stats)))
    for dataset_idx, dataset in enumerate(dataset_stats):
        values_closed = np.r_[normalized[dataset_idx], normalized[dataset_idx, 0]]
        metric_mean = float(np.mean(normalized[dataset_idx]))
        label = f"{dataset['name']} ({metric_mean:.2f})"
        ax.plot(angles, values_closed, color=colors[dataset_idx], linewidth=2, label=label)
        ax.fill(angles, values_closed, color=colors[dataset_idx], alpha=0.15)

    ax.set_title("VBench metric radar chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # print(f"Computed per-axis standard deviations: {stds.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a radar plot from VBench *_eval_results.json files")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("output/vbench"),
        help="Directory containing VBench *_eval_results.json files",
    )
    args = parser.parse_args()

    search_root = args.path.resolve()
    dataset_stats, metric_names = collect_metric_stats(search_root)
    print_dataset_metrics(dataset_stats, metric_names)
    output_path = search_root / "star_plot.png"
    plot_radar_chart(dataset_stats, metric_names, output_path)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
