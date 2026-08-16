#!/usr/bin/env python3
"""Small diagnostic plots for the revised HOT3D hand bottleneck experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "runs/hot3d_hand_diagnosis"


def _safe(value: float | None) -> float:
    return value if value is not None and math.isfinite(float(value)) else math.nan


def _domain_plot() -> None:
    data = json.loads((OUTPUT_ROOT / "eval_domain.json").read_text(encoding="utf-8"))
    items = data["items"]
    labels = [f"{item['clip'].replace('.tar', '')}\n{item['selected_side']}" for item in items]
    left = [item["gt_visible_left_count"] for item in items]
    right = [item["gt_visible_right_count"] for item in items]
    both = [item["gt_visible_both_count"] for item in items]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - width, left, width, label="GT-visible left")
    ax.bar(x, right, width, label="GT-visible right")
    ax.bar(x + width, both, width, label="GT-visible both")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("frames")
    ax.set_title("Experiment 0: frozen GT-visible domain")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_ROOT / "exp0_domain_vis.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def _exp1_plot() -> None:
    data = json.loads(
        (OUTPUT_ROOT / "exp1_2d_localization_summary.json").read_text(encoding="utf-8")
    )
    items = data["items"]
    labels: list[str] = []
    wilor_left: list[float] = []
    wilor_right: list[float] = []
    hamer_left: list[float] = []
    hamer_right: list[float] = []
    wilor_2d: list[float] = []
    hamer_2d: list[float] = []
    for item in items:
        if item.get("status") != "evaluated":
            continue
        labels.append(f"{item['clip'].replace('.tar', '')}\n{item['selected_side']}")
        cov = item.get("coverage", {})
        wilor_left.append(_safe(cov.get("left", {}).get("detection_rate")))
        wilor_right.append(_safe(cov.get("right", {}).get("detection_rate")))
        # HaMeR is the next item in the sorted sequence; collect below in second pass.
    model_rows = [item for item in items if item.get("status") == "evaluated"]
    wilor_rows = model_rows[::2]
    hamer_rows = model_rows[1::2]
    x = np.arange(len(wilor_rows))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    ax.bar(
        x - 1.5 * width,
        [item.get("coverage", {}).get("left", {}).get("detection_rate", math.nan) for item in wilor_rows],
        width,
        label="WiLoR left",
    )
    ax.bar(
        x - 0.5 * width,
        [item.get("coverage", {}).get("right", {}).get("detection_rate", math.nan) for item in wilor_rows],
        width,
        label="WiLoR right",
    )
    ax.bar(
        x + 0.5 * width,
        [item.get("coverage", {}).get("left", {}).get("detection_rate", math.nan) for item in hamer_rows],
        width,
        label="HaMeR left",
    )
    ax.bar(
        x + 1.5 * width,
        [item.get("coverage", {}).get("right", {}).get("detection_rate", math.nan) for item in hamer_rows],
        width,
        label="HaMeR right",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{item['clip'].replace('.tar', '')}\n{item['selected_side']}" for item in wilor_rows],
        fontsize=8,
    )
    ax.set_ylabel("detection rate in GT-visible domain")
    ax.set_title("Experiment 1: coverage")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    categories = [item["clip"].replace(".tar", "") for item in wilor_rows]
    med_wilor = [
        item.get("localization", {})
        .get("left_only_left_view", {})
        .get("keypoint_error_px", {})
        .get("median", math.nan)
        for item in wilor_rows
    ]
    med_hamer = [
        item.get("localization", {})
        .get("left_only_left_view", {})
        .get("keypoint_error_px", {})
        .get("median", math.nan)
        for item in hamer_rows
    ]
    ax.plot(categories, med_wilor, marker="o", label="WiLoR")
    ax.plot(categories, med_hamer, marker="s", label="HaMeR")
    ax.set_ylabel("median 2D keypoint error (px)")
    ax.set_title("Experiment 1: localization (evaluable views)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_ROOT / "exp1_2d_localization_vis.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def _exp3_plot() -> None:
    data = json.loads(
        (OUTPUT_ROOT / "exp3_depth_scale_summary.json").read_text(encoding="utf-8")
    )
    items = data["items"]
    variants = ["D0", "D1", "D2", "D3"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_keys = [
        ("mpjpe_mm", "MPJPE (mm)"),
        ("root_aligned_mpjpe_mm", "root-aligned MPJPE (mm)"),
        ("pa_mpjpe_mm", "PA-MPJPE (mm)"),
    ]
    for ax, (key, title) in zip(axes, metric_keys):
        for item in items:
            if not item.get("variants"):
                continue
            label = f"{item['clip'].replace('.tar', '')}\n{item['model']}"
            values = [
                item["variants"][variant].get(key, {}).get("mean", math.nan)
                for variant in variants
            ]
            ax.plot(variants, values, marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("oracle variant")
        ax.set_ylabel("mm")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = OUTPUT_ROOT / "exp3_depth_scale_vis.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def main() -> int:
    _domain_plot()
    _exp1_plot()
    _exp3_plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
