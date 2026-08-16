#!/usr/bin/env python3
"""P1 rectification valid-region audit and fair hand-observation metrics."""

from __future__ import annotations

import csv
import json
import math
import sys
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
HOT3D_TOOLKIT_REPO = Path(
    "/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo"
)
if str(HOT3D_TOOLKIT_REPO) not in sys.path:
    sys.path.insert(0, str(HOT3D_TOOLKIT_REPO))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_tracking_toolkit import camera
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel

from scripts.evaluate_hot3d_fov_ablation import (
    ABLATION_RUNS,
    CLIPS_ROOT,
    MANO_DIR,
    REQUIRED,
    FINGERTIPS,
    _build_gt_world,
    _load_model_artifacts,
    _model_observation,
    _pinhole_visibility,
)


OUTPUT_ROOT = REPO_ROOT / "runs/hot3d_hand_diagnosis/p1_valid_region_audit"
P1_MANIFEST = ABLATION_RUNS / "P1" / "manifest.json"
WIDTH = 640
HEIGHT = 480
VALID_THRESHOLD = 0.5


def _p1_camera(
    T: np.ndarray,
) -> camera.PinholePlaneCameraModel:
    return camera.PinholePlaneCameraModel(
        width=WIDTH,
        height=HEIGHT,
        f=[160.0, 160.0],
        c=(WIDTH / 2.0, HEIGHT / 2.0),
        distort_coeffs=[],
        T_world_from_eye=np.asarray(T, dtype=np.float64).copy(),
    )


def _valid_mask(
    src_camera: camera.CameraModel,
    dst_camera: camera.PinholePlaneCameraModel,
) -> np.ndarray:
    width, height = WIDTH, HEIGHT
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    win = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    dst_eye = dst_camera.window_to_eye(win)
    world = dst_camera.eye_to_world(dst_eye)
    src_eye = src_camera.world_to_eye(world)
    src_win = src_camera.eye_to_window(src_eye)
    inside = (
        (src_win[:, 0] >= -0.5)
        & (src_win[:, 0] < src_camera.width - 0.5)
        & (src_win[:, 1] >= -0.5)
        & (src_win[:, 1] < src_camera.height - 0.5)
    )
    front = src_eye[:, 2] > 0
    return (inside & front).reshape(height, width).astype(np.uint8)


def _bbox_from_uv(uv: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    finite = np.isfinite(uv).all(axis=-1)
    inside = (
        finite
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    if not inside.any():
        return None
    points = uv[inside]
    x0 = int(math.floor(float(np.min(points[:, 0]))))
    y0 = int(math.floor(float(np.min(points[:, 1]))))
    x1 = int(math.ceil(float(np.max(points[:, 0]))))
    y1 = int(math.ceil(float(np.max(points[:, 1]))))
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _mask_ratio_at_points(
    mask: np.ndarray,
    uv: np.ndarray,
    width: int,
    height: int,
) -> float:
    finite = np.isfinite(uv).all(axis=-1)
    inside = (
        finite
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    if not inside.any():
        return math.nan
    coords = np.rint(uv[inside]).astype(np.int64)
    coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
    values = mask[coords[:, 1], coords[:, 0]]
    return float(values.mean())


def _min_valid_distance(
    mask: np.ndarray,
    uv: np.ndarray,
    width: int,
    height: int,
) -> float:
    finite = np.isfinite(uv).all(axis=-1)
    inside = (
        finite
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    if not inside.any():
        return -1.0
    coords = np.rint(uv[inside]).astype(np.int64)
    coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    values = distances[coords[:, 1], coords[:, 0]]
    return float(values.min())


def _classify_failure(row: dict[str, Any]) -> str:
    if row["hand_roi_valid_ratio"] < VALID_THRESHOLD:
        return "A_invalid_remap_dominated"
    if row["hand_bbox_diagonal_px"] < 40.0:
        return "B_very_small_hand"
    if (
        row["distance_to_invalid_boundary_px"] >= 0
        and row["distance_to_invalid_boundary_px"] < 10.0
    ):
        return "C_near_valid_boundary"
    if row["roi_mean_intensity"] is not None and row["roi_mean_intensity"] < 40.0:
        return "D_low_illumination"
    if row["bbox_touches_image_boundary"]:
        return "E_truncated_or_occluded"
    return "F_visually_clear_model_miss"


def _draw_example_panel(
    image: np.ndarray,
    uv: np.ndarray,
    title: str,
) -> np.ndarray:
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for index in range(21):
        u, v = float(uv[index, 0]), float(uv[index, 1])
        if u < 0 or v < 0 or u >= out.shape[1] or v >= out.shape[0]:
            continue
        color = (0, 0, 255) if index == 0 else (255, 200, 0)
        if index in FINGERTIPS:
            color = (0, 165, 255)
        cv2.circle(out, (int(round(u)), int(round(v))), 3, color, -1)
    cv2.putText(
        out,
        title,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mano_model = MANOHandModel(str(MANO_DIR))
    manifest = json.loads(P1_MANIFEST.read_text(encoding="utf-8"))["items"]
    rows: list[dict[str, Any]] = []
    representative_masks: dict[str, np.ndarray] = {}

    for entry in manifest:
        run_dir = Path(entry["run_dir"]).resolve()
        clip = entry["clip"]
        side = entry["selected_side"]
        side_index = 0 if side == "left" else 1
        frame_numbers, world = _build_gt_world(run_dir, mano_model)
        pin_left, pin_right, uv_left, uv_right = _pinhole_visibility(run_dir, world)
        K_left = np.load(run_dir / "calibration/intrinsics.npy").astype(np.float64)
        K_right = np.load(run_dir / "calibration/intrinsics_right.npy").astype(np.float64)
        T_left_all = np.load(run_dir / "calibration/T_world_camera.npy").astype(np.float64)
        T_right_all = np.load(run_dir / "calibration/T_world_camera_right.npy").astype(np.float64)
        artifacts = {
            model: _load_model_artifacts(run_dir, model)
            for model in ("wilor", "hamer")
        }
        model_p = {
            model: {
                "left": _model_observation(artifacts[model]["left"], side_index, K_left)[2],
                "right": _model_observation(artifacts[model]["right"], side_index, K_right)[2],
            }
            for model in ("wilor", "hamer")
            if "left" in artifacts[model] and "right" in artifacts[model]
        }
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            for t, frame_number in enumerate(frame_numbers):
                cams = json.load(tar.extractfile(f"{int(frame_number):06d}.cameras.json"))
                raw_left = camera.from_json(cams["1201-1"])
                raw_right = camera.from_json(cams["1201-2"])
                dst_left = _p1_camera(T_left_all[t])
                dst_right = _p1_camera(T_right_all[t])
                masks = {
                    "left": _valid_mask(raw_left, dst_left),
                    "right": _valid_mask(raw_right, dst_right),
                }
                if clip == "clip-001852.tar" and t == 0:
                    representative_masks["left"] = masks["left"].copy()
                    representative_masks["right"] = masks["right"].copy()
                for view_name in ("left", "right"):
                    gt_uv = uv_left if view_name == "left" else uv_right
                    image_visible = bool((pin_left if view_name == "left" else pin_right)[t])
                    bbox = _bbox_from_uv(gt_uv[t], WIDTH, HEIGHT)
                    mask = masks[view_name]
                    required_ratio = _mask_ratio_at_points(
                        mask, gt_uv[t, REQUIRED], WIDTH, HEIGHT
                    )
                    roi_ratio = math.nan
                    bbox_area = 0
                    bbox_width = 0
                    bbox_height = 0
                    bbox_diag = 0.0
                    dist = -1.0
                    roi_mean = None
                    touches_boundary = False
                    if bbox is not None:
                        x0, y0, x1, y1 = bbox
                        bbox_width = x1 - x0 + 1
                        bbox_height = y1 - y0 + 1
                        bbox_area = int(bbox_width * bbox_height)
                        bbox_diag = float(math.hypot(bbox_width, bbox_height))
                        roi = mask[y0 : y1 + 1, x0 : x1 + 1]
                        roi_ratio = float(roi.mean())
                        image_path = (
                            run_dir / "frames/rgb" / f"{t:06d}.png"
                            if view_name == "left"
                            else run_dir / "frames/right" / f"{t:06d}.png"
                        )
                        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            roi_img = image[y0 : y1 + 1, x0 : x1 + 1]
                            roi_mean = float(roi_img.mean())
                        touches_boundary = bool(
                            x0 <= 2 or y0 <= 2 or x1 >= WIDTH - 3 or y1 >= HEIGHT - 3
                        )
                    dist = _min_valid_distance(mask, gt_uv[t], WIDTH, HEIGHT)
                    gt_visible_image = image_visible
                    gt_visible_valid = bool(
                        gt_visible_image
                        and math.isfinite(required_ratio)
                        and math.isfinite(roi_ratio)
                        and required_ratio >= VALID_THRESHOLD
                        and roi_ratio >= VALID_THRESHOLD
                    )
                    row = {
                        "clip": clip,
                        "frame": int(frame_number),
                        "frame_index": t,
                        "selected_hand_side": side,
                        "camera_side": view_name,
                        "gt_visible_image": gt_visible_image,
                        "gt_visible_valid": gt_visible_valid,
                        "required_joint_valid_ratio": required_ratio,
                        "hand_roi_valid_ratio": roi_ratio,
                        "hand_bbox_area_px": bbox_area,
                        "hand_bbox_width": bbox_width,
                        "hand_bbox_height": bbox_height,
                        "hand_bbox_diagonal_px": bbox_diag,
                        "distance_to_invalid_boundary_px": dist,
                        "full_valid_pixel_ratio": float(mask.mean()),
                        "roi_mean_intensity": roi_mean,
                        "bbox_touches_image_boundary": touches_boundary,
                        "left_image": str(run_dir / "frames/rgb" / f"{t:06d}.png"),
                        "right_image": str(run_dir / "frames/right" / f"{t:06d}.png"),
                    }
                    for model in ("wilor", "hamer"):
                        if model in model_p:
                            row[f"{model}_detected"] = bool(
                                model_p[model][view_name][t]
                            )
                        else:
                            row[f"{model}_detected"] = False
                    rows.append(row)

    # Representative masks.
    if representative_masks:
        cv2.imwrite(
            str(OUTPUT_ROOT / "p1_valid_mask_left.png"),
            representative_masks["left"] * 255,
        )
        cv2.imwrite(
            str(OUTPUT_ROOT / "p1_valid_mask_right.png"),
            representative_masks["right"] * 255,
        )

    # Distributions.
    image_rows = [r for r in rows if r["gt_visible_image"]]
    dist_buckets = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]

    def bucket(value: float) -> str | None:
        if not math.isfinite(value):
            return None
        if value < 0.25:
            return dist_buckets[0]
        if value < 0.5:
            return dist_buckets[1]
        if value < 0.75:
            return dist_buckets[2]
        return dist_buckets[3]

    valid_region_summary: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol": "P1",
        "valid_pixel_definition": (
            "P1 pixel inverse-mapped to raw fisheye; source coordinate inside "
            "raw sensor and source ray z>0"
        ),
        "threshold_candidates": {
            "roi_valid_gte_0.5": VALID_THRESHOLD,
            "roi_valid_gte_0.75": 0.75,
        },
        "valid_pixel_ratio": {
            "left": float(np.mean([r["full_valid_pixel_ratio"] for r in rows if r["camera_side"] == "left"])),
            "right": float(np.mean([r["full_valid_pixel_ratio"] for r in rows if r["camera_side"] == "right"])),
        },
        "required_joint_valid_ratio_distribution": {
            b: sum(1 for r in image_rows if bucket(r["required_joint_valid_ratio"]) == b)
            for b in dist_buckets
        },
        "hand_roi_valid_ratio_distribution": {
            b: sum(1 for r in image_rows if bucket(r["hand_roi_valid_ratio"]) == b)
            for b in dist_buckets
        },
        "gt_visible_image_count": len(image_rows),
        "gt_visible_valid_count": sum(1 for r in image_rows if r["gt_visible_valid"]),
    }
    (OUTPUT_ROOT / "valid_region_summary.json").write_text(
        json.dumps(valid_region_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Frame/view matrix.
    fieldnames = list(rows[0].keys())
    with (OUTPUT_ROOT / "frame_view_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Detection old vs new.
    detection_rows: list[dict[str, Any]] = []
    for model in ("wilor", "hamer"):
        for domain_name, domain_key in (("old_image", "gt_visible_image"), ("new_valid", "gt_visible_valid")):
            agg = {"protocol": "P1", "model": model, "domain": domain_name}
            for view_name in ("left", "right", "both"):
                if view_name == "both":
                    denom = [
                        r
                        for r in rows
                        if r[domain_key]
                        and r["camera_side"] == "left"
                        and any(
                            x["clip"] == r["clip"] and x["frame"] == r["frame"]
                            and x["camera_side"] == "right"
                            and x[domain_key]
                            for x in rows
                        )
                    ]
                    detected = [
                        r
                        for r in rows
                        if r[domain_key]
                        and r["camera_side"] == "left"
                        and r[f"{model}_detected"]
                        and any(
                            x["clip"] == r["clip"] and x["frame"] == r["frame"]
                            and x["camera_side"] == "right"
                            and x[domain_key]
                            and x[f"{model}_detected"]
                            for x in rows
                        )
                    ]
                    agg[f"{view_name}_denominator"] = len(denom)
                    agg[f"{view_name}_recall"] = (
                        len(detected) / len(denom) if denom else math.nan
                    )
                else:
                    denom = [r for r in rows if r["camera_side"] == view_name and r[domain_key]]
                    detected = [
                        r
                        for r in denom
                        if r[f"{model}_detected"]
                    ]
                    agg[f"{view_name}_denominator"] = len(denom)
                    agg[f"{view_name}_recall"] = (
                        len(detected) / len(denom) if denom else math.nan
                    )
            detection_rows.append(agg)
    (OUTPUT_ROOT / "detection_old_vs_new.json").write_text(
        json.dumps({"rows": detection_rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT_ROOT / "detection_old_vs_new.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["protocol", "model", "domain", "left_denom", "left_recall", "right_denom", "right_recall", "both_denom", "both_recall"]
        )
        for row in detection_rows:
            writer.writerow(
                [
                    row["protocol"], row["model"], row["domain"],
                    row["left_denominator"], row["left_recall"],
                    row["right_denominator"], row["right_recall"],
                    row["both_denominator"], row["both_recall"],
                ]
            )

    # Side x camera summary.
    side_summary_rows: list[dict[str, Any]] = []
    for clip in sorted({r["clip"] for r in rows}):
        clip_rows = [r for r in rows if r["clip"] == clip]
        for camera_side in ("left", "right"):
            sub = [r for r in clip_rows if r["camera_side"] == camera_side]
            side_summary_rows.append(
                {
                    "clip": clip,
                    "selected_hand_side": sub[0]["selected_hand_side"] if sub else "",
                    "camera_side": camera_side,
                    "image_visible": sum(1 for r in sub if r["gt_visible_image"]),
                    "valid_visible": sum(1 for r in sub if r["gt_visible_valid"]),
                    "wilor_detected": sum(1 for r in sub if r["wilor_detected"]),
                    "hamer_detected": sum(1 for r in sub if r["hamer_detected"]),
                    "mean_required_joint_valid_ratio": float(
                        np.nanmean([r["required_joint_valid_ratio"] for r in sub])
                    ),
                    "mean_hand_roi_valid_ratio": float(
                        np.nanmean([r["hand_roi_valid_ratio"] for r in sub])
                    ),
                    "mean_bbox_diagonal_px": float(
                        np.nanmean([r["hand_bbox_diagonal_px"] for r in sub])
                    ),
                    "mean_distance_to_invalid_boundary_px": float(
                        np.nanmean([r["distance_to_invalid_boundary_px"] for r in sub])
                    ),
                }
            )
    with (OUTPUT_ROOT / "hand_side_camera_side_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(side_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(side_summary_rows)

    # Localization on valid domain.
    localization_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row["gt_visible_valid"]:
            continue
        for model in ("wilor", "hamer"):
            if not row[f"{model}_detected"]:
                continue
            view_name = row["camera_side"]
            run_dir = next(
                Path(item["run_dir"])
                for item in manifest
                if item["clip"] == row["clip"]
            )
            K = np.load(
                run_dir / ("calibration/intrinsics.npy" if view_name == "left" else "calibration/intrinsics_right.npy")
            ).astype(np.float64)
            artifacts = _load_model_artifacts(run_dir, model)
            side_index = 0 if row["selected_hand_side"] == "left" else 1
            pred_uv = _model_observation(
                artifacts[view_name], side_index, K
            )[0][row["frame_index"]]
            gt_uv = uv_left[row["frame_index"]] if view_name == "left" else uv_right[row["frame_index"]]
            errors = np.linalg.norm(pred_uv - gt_uv, axis=-1)
            wrist = float(np.linalg.norm(pred_uv[0] - gt_uv[0]))
            fingertip = float(
                np.mean(
                    np.linalg.norm(
                        pred_uv[FINGERTIPS] - gt_uv[FINGERTIPS],
                        axis=-1,
                    )
                )
            )
            denom = max(row["hand_bbox_diagonal_px"], 1.0)
            localization_rows.append(
                {
                    "clip": row["clip"],
                    "frame": row["frame"],
                    "selected_hand_side": row["selected_hand_side"],
                    "camera_side": view_name,
                    "model": model,
                    "keypoint_error_mean_px": float(np.mean(errors)),
                    "keypoint_error_median_px": float(np.median(errors)),
                    "wrist_error_px": wrist,
                    "fingertip_error_px": fingertip,
                    "pck5": float(np.mean(errors <= 5)),
                    "pck10": float(np.mean(errors <= 10)),
                    "pck20": float(np.mean(errors <= 20)),
                    "nme_bbox_diagonal": float(np.mean(errors) / denom),
                }
            )
    with (OUTPUT_ROOT / "localization_valid_domain.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(localization_rows[0].keys()))
        writer.writeheader()
        writer.writerows(localization_rows)
    localization_summary: dict[str, Any] = {"rows": localization_rows}
    if localization_rows:
        for model in ("wilor", "hamer"):
            sub = [r for r in localization_rows if r["model"] == model]
            localization_summary[model] = {
                "count": len(sub),
                "keypoint_error_mean_px": float(np.mean([r["keypoint_error_mean_px"] for r in sub])),
                "keypoint_error_median_of_medians_px": float(np.median([r["keypoint_error_median_px"] for r in sub])),
                "wrist_error_mean_px": float(np.mean([r["wrist_error_px"] for r in sub])),
                "fingertip_error_mean_px": float(np.mean([r["fingertip_error_px"] for r in sub])),
                "mean_pck5": float(np.mean([r["pck5"] for r in sub])),
                "mean_pck10": float(np.mean([r["pck10"] for r in sub])),
                "mean_pck20": float(np.mean([r["pck20"] for r in sub])),
                "mean_nme_bbox_diagonal": float(np.mean([r["nme_bbox_diagonal"] for r in sub])),
            }
    (OUTPUT_ROOT / "localization_valid_domain.json").write_text(
        json.dumps(localization_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Failure taxonomy examples for WiLoR.
    failure_rows = [
        r
        for r in rows
        if r["gt_visible_valid"] and not r["wilor_detected"]
    ]
    for row in failure_rows:
        row["failure_class"] = _classify_failure(row)
    classes = [
        "A_invalid_remap_dominated",
        "B_very_small_hand",
        "C_near_valid_boundary",
        "D_low_illumination",
        "E_truncated_or_occluded",
        "F_visually_clear_model_miss",
    ]
    panels: list[np.ndarray] = []
    titles: list[str] = []
    for class_name in classes:
        examples = [r for r in failure_rows if r["failure_class"] == class_name][:10]
        for row in examples:
            view_name = row["camera_side"]
            image_path = Path(
                row["left_image"] if view_name == "left" else row["right_image"]
            )
            image = cv2.imread(str(image_path))
            if image is None:
                image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            run_dir = next(
                Path(item["run_dir"])
                for item in manifest
                if item["clip"] == row["clip"]
            )
            gt_uv = uv_left[row["frame_index"]] if view_name == "left" else uv_right[row["frame_index"]]
            panels.append(
                _draw_example_panel(
                    image,
                    gt_uv,
                    f"{class_name[:1]}:{row['clip'].replace('.tar','')}/{view_name[0]}/f{row['frame_index']}",
                )
            )
            titles.append(class_name)
    if panels:
        target_w, target_h = 220, 165
        resized = [cv2.resize(p, (target_w, target_h)) for p in panels]
        per_row = 10
        rows_out = []
        for start in range(0, len(resized), per_row):
            chunk = resized[start : start + per_row]
            if len(chunk) < per_row:
                chunk.extend(
                    [np.zeros_like(chunk[0]) for _ in range(per_row - len(chunk))]
                )
            rows_out.append(np.hstack(chunk))
        cv2.imwrite(
            str(OUTPUT_ROOT / "failure_examples.png"),
            np.vstack(rows_out),
        )

    # Config.
    config = {
        "schema_version": "1.0",
        "protocol": "P1",
        "valid_pixel_definition": valid_region_summary["valid_pixel_definition"],
        "valid_domain_threshold": {
            "required_joint_valid_ratio_gte": VALID_THRESHOLD,
            "hand_roi_valid_ratio_gte": VALID_THRESHOLD,
        },
    }
    (OUTPUT_ROOT / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Summary markdown.
    old_wilor = next(r for r in detection_rows if r["model"] == "wilor" and r["domain"] == "old_image")
    new_wilor = next(r for r in detection_rows if r["model"] == "wilor" and r["domain"] == "new_valid")
    old_hamer = next(r for r in detection_rows if r["model"] == "hamer" and r["domain"] == "old_image")
    new_hamer = next(r for r in detection_rows if r["model"] == "hamer" and r["domain"] == "new_valid")
    md = f"""# P1 Valid-Region Audit

## Q1
GT_VISIBLE_IMAGE frames/views: {valid_region_summary['gt_visible_image_count']}
GT_VISIBLE_VALID frames/views at ROI/required valid >= {VALID_THRESHOLD}: {valid_region_summary['gt_visible_valid_count']}

## Q2
Fair both-view denominator (new valid): {new_wilor['both_denominator']}

## Q3
- WiLoR new both recall: {new_wilor['both_recall']:.4f}
- HaMeR new both recall: {new_hamer['both_recall']:.4f}

## Old vs new detection
| model | domain | left | right | both |
|---|---|---|---:|---:|---:|
| WiLoR | old | {old_wilor['left_recall']:.4f} | {old_wilor['right_recall']:.4f} | {old_wilor['both_recall']:.4f} |
| WiLoR | new | {new_wilor['left_recall']:.4f} | {new_wilor['right_recall']:.4f} | {new_wilor['both_recall']:.4f} |
| HaMeR | old | {old_hamer['left_recall']:.4f} | {old_hamer['right_recall']:.4f} | {old_hamer['both_recall']:.4f} |
| HaMeR | new | {new_hamer['left_recall']:.4f} | {new_hamer['right_recall']:.4f} | {new_hamer['both_recall']:.4f} |

## Q4/Q5
See `hand_side_camera_side_summary.csv` and `failure_examples.png`.
"""
    (OUTPUT_ROOT / "summary.md").write_text(md, encoding="utf-8")

    print("audit outputs ->", OUTPUT_ROOT)
    print(json.dumps(valid_region_summary, indent=2, ensure_ascii=False))
    print(json.dumps(detection_rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
