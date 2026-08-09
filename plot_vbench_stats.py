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


# ---------------
# From https://github.com/Vchitect/VBench/blob/master/scripts/constant.py
DIM_WEIGHT = {
    "subject consistency":1,
    "background consistency":1,
    "temporal flickering":1,
    "motion smoothness":1,
    "aesthetic quality":1,
    "imaging quality":1,
    "dynamic degree":0.5,
    "object class":1,
    "multiple objects":1,
    "human action":1,
    "color":1,
    "spatial relationship":1,
    "scene":1,
    "appearance style":1,
    "temporal style":1,
    "overall consistency":1
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
    "overall consistency": {"Min": 0.0, "Max": 0.364}
}

TASK_INFO = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency"
    ]

def get_normalized_score(upload_data):
    # get the normalize score
    normalized_score = {}
    for key in upload_data.keys():
        min_val = NORMALIZE_DIC[key]['Min']
        max_val = NORMALIZE_DIC[key]['Max']
        normalized_score[key] = (upload_data[key] - min_val) / (max_val - min_val)
        normalized_score[key] = normalized_score[key] * DIM_WEIGHT[key]
    return normalized_score

# ---------------

def parse_vbench_results(file_path: Path) -> dict[str, tuple[float, list[float]]]:
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    parsed: dict[str, tuple[float, list[float]]] = {}
    for metric_name, metric_value in payload.items():
        metric_name = metric_name.replace("_", " ")
        if metric_name not in TASK_INFO:
            raise ValueError(f"Unexpected metric name in {file_path}: {metric_name}")
        
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


def collect_metric_stats(search_root: Path, plot_missing_dimension: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    dim_dirs = sorted([path for path in search_root.iterdir() if path.is_dir()])
    if not dim_dirs:
        raise FileNotFoundError(f"No dimension subfolders found under {search_root}")

    dataset_stats_by_name: dict[str, dict[str, Any]] = {}
    metric_names_seen: set[str] = set()

    for dim_dir in dim_dirs:
        files = sorted(dim_dir.glob("*_eval_results.json"))
        if not files:
            continue

        for file_path in files:
            dataset = file_path.stem.replace("_eval_results", "")
            parsed = parse_vbench_results(file_path)

            if dataset not in dataset_stats_by_name:
                dataset_stats_by_name[dataset] = {"name": dataset, "metrics": {}}

            for metric_name, (summary_value, values) in parsed.items():
                values_array = np.array(values, dtype=float)
                if metric_name == "imaging quality":
                    values_array /= 100.0

                mean_value = float(values_array.mean())
                std_value = float(values_array.std(ddof=1)) if len(values_array) > 1 else 0.0
                result_count = int(len(values_array))

                dataset_stats_by_name[dataset]["metrics"][metric_name] = {
                    "mean": mean_value,
                    "std": std_value,
                    "summary": summary_value,
                    "count": result_count,
                    "values": values_array.tolist(),
                }
                metric_names_seen.add(metric_name)

    if not dataset_stats_by_name:
        raise FileNotFoundError(f"No *_eval_results.json files found under dimension folders in {search_root}")

    dataset_stats = [dataset_stats_by_name[name] for name in sorted(dataset_stats_by_name)]

    common_metric_names: set[str] | None = None
    for dataset in dataset_stats:
        dataset_metric_names = set(dataset["metrics"].keys())
        if common_metric_names is None:
            common_metric_names = dataset_metric_names
        else:
            common_metric_names &= dataset_metric_names

    if common_metric_names is None:
        raise ValueError(f"No dimensions found across datasets in {search_root}")

    dropped_metrics = sorted(metric_names_seen - common_metric_names)
    if plot_missing_dimension:
        metric_names = sorted(metric_names_seen)
        if dropped_metrics:
            print(f"Keeping dimensions with missing dataset values: {dropped_metrics}")
    else:
        if not common_metric_names:
            raise ValueError(f"No common dimensions found across datasets in {search_root}")
        if dropped_metrics:
            print(f"Skipping dimensions missing in some datasets: {dropped_metrics}")
        metric_names = sorted(common_metric_names)

    if not plot_missing_dimension:
        for dataset in dataset_stats:
            dataset["metrics"] = {name: dataset["metrics"][name] for name in metric_names}

    return dataset_stats, metric_names


def rel_metrics(dataset_stats: list[dict[str, Any]], metric_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(dataset_stats), len(metric_names)), dtype=float)
    stds = np.zeros((len(dataset_stats), len(metric_names)), dtype=float)

    for dataset_idx, dataset in enumerate(dataset_stats):
        for metric_idx, metric_name in enumerate(metric_names):
            if metric_name in dataset["metrics"]:
                values[dataset_idx, metric_idx] = dataset["metrics"][metric_name]["mean"]
                stds[dataset_idx, metric_idx] = dataset["metrics"][metric_name]["std"]

    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    # maxs = np.maximum(maxs, 1.0)
    ranges = maxs - mins
    normalized = np.zeros_like(values)
    for idx, span in enumerate(ranges):
        if span > 0:
            normalized[:, idx] = (values[:, idx] - mins[idx]) / span
        else:
            normalized[:, idx] = 0.0

    return normalized, stds


def normalized_metrics(dataset_stats: list[dict[str, Any]], metric_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    normalized_means = np.zeros((len(dataset_stats), len(metric_names)), dtype=float)
    normalized_stds = np.zeros((len(dataset_stats), len(metric_names)), dtype=float)

    for dataset_idx, dataset in enumerate(dataset_stats):
        for metric_idx, metric_name in enumerate(metric_names):
            if metric_name not in dataset["metrics"]:
                continue
            metric = dataset["metrics"][metric_name]

            values_normalized = [
                get_normalized_score({metric_name: float(value)})[metric_name]
                for value in metric["values"]
            ]
            normalized_values = np.array(values_normalized, dtype=float)

            normalized_means[dataset_idx, metric_idx] = float(normalized_values.mean())
            normalized_stds[dataset_idx, metric_idx] = (
                float(normalized_values.std(ddof=1)) if len(normalized_values) > 1 else 0.0
            )

    return normalized_means, normalized_stds


def print_dataset_metrics(dataset_stats: list[dict[str, Any]], metric_names: list[str]) -> None:
    for dataset in dataset_stats:
        mean = np.array([dataset["metrics"][name]["mean"] for name in metric_names if name in dataset["metrics"]], dtype=float)
        std = np.array([dataset["metrics"][name]["std"] for name in metric_names if name in dataset["metrics"]], dtype=float)

        metrics_dict = {name: dataset["metrics"][name]["mean"] for name in metric_names if name in dataset["metrics"]}

        # Following VBench recommendation to compute the score: computing the normalized metrics and applying dimension weights
        metrics_norm = get_normalized_score(metrics_dict)
        dataset_score = np.mean([v * DIM_WEIGHT[k] for k, v in metrics_norm.items()])

        print(f"{dataset['name']}: mean={mean.mean():.3f}, std={std.mean():.3f} => score={dataset_score:.4f}")
        for m in metric_names:
            metric = dataset["metrics"].get(m)
            if metric is None:
                print(f"  {m}: MISSING")
                continue
            print(f"  {m}: {metric['mean']:.3f} ({metric['std']:.3f}) / norm={metrics_norm[m]:.3f} [{metric['count']}]")
        print()


def plot_radar_chart(dataset_stats: list[dict[str, Any]], metric_names: list[str], output_path: Path) -> None:
    means, stds = rel_metrics(dataset_stats, metric_names)
    # means, stds = normalized_metrics(dataset_stats, metric_names)

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(-0.02, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.grid(color="0.7", linestyle="--", linewidth=0.8)

    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_stats)))
    for dataset_idx, dataset in enumerate(dataset_stats):
        values_closed = np.r_[means[dataset_idx], means[dataset_idx, 0]]
        metric_mean = float(np.mean(means[dataset_idx]))
        missing_dims = sum(1 for metric_name in metric_names if metric_name not in dataset["metrics"])

        # Following VBench recommendation to compute the score: computing the normalized metrics and applying dimension weights
        metrics_dict = {name: dataset["metrics"][name]["mean"] for name in metric_names if name in dataset["metrics"]}
        metrics_norm = get_normalized_score(metrics_dict)
        dataset_score = np.mean([v * DIM_WEIGHT[k] for k, v in metrics_norm.items()])

        label = f"{dataset['name']} (rel={metric_mean:.3f}, score={dataset_score:.4f})"
        if missing_dims > 0:
            label += f" **MISSING {missing_dims}**"
        ax.plot(angles, values_closed, color=colors[dataset_idx], linewidth=2, label=label, marker='s')
        ax.fill(angles, values_closed, color=colors[dataset_idx], alpha=0.15)

    ax.set_title(f"VBench ({len(metric_names)} metrics)", fontsize=14, fontweight="bold", pad=20)
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
    parser.add_argument(
        "--missing-dimension",
        action="store_true",
        help="Plot all dimensions even if some datasets are missing them",
    )
    parser.add_argument(
        "--ignore-dimension",
        type=str,
        default=None,
        help="Comma-separated list of dimensions to ignore",
    )
    args = parser.parse_args()

    search_root = args.path.resolve()
    dataset_stats, metric_names = collect_metric_stats(search_root, plot_missing_dimension=args.missing_dimension)
    ignored_dimensions: set[str] = set()

    if args.ignore_dimension:
        ignored_dimensions = {item.strip().replace("_", " ") for item in args.ignore_dimension.split(",") if item.strip()}
        metric_names = [metric_name for metric_name in metric_names if metric_name not in ignored_dimensions]
        for dataset in dataset_stats:
            dataset["metrics"] = {name: metric for name, metric in dataset["metrics"].items() if name not in ignored_dimensions}

    print_dataset_metrics(dataset_stats, metric_names)
    
    output_name = "star_plot"
    if args.ignore_dimension:
        ignored_suffix = "_".join(
            "".join(word[0] for word in dimension.replace("_", " ").strip().split() if word)
            for dimension in sorted(ignored_dimensions)
        )
        output_name += "_ignore-" + ignored_suffix
    if args.missing_dimension:
        output_name += "_missing"

    output_path = search_root / f"{output_name}.png"
    plot_radar_chart(dataset_stats, metric_names, output_path)
    print(f"Saved plot to: {output_path}\n\n")


if __name__ == "__main__":
    main()
