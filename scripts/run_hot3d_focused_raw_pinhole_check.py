#!/usr/bin/env python3
"""Focused raw-fisheye vs virtual-pinhole sanity check.

Only for clip-001852 and clip-001891. This script does not alter production
pipeline artifacts; it reads HOT3D tars directly and writes diagnostics.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
HOT3D_TOOLKIT_REPO = Path(
    "/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo"
)
if str(HOT3D_TOOLKIT_REPO) not in sys.path:
    sys.path.insert(0, str(HOT3D_TOOLKIT_REPO))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_tracking_toolkit import camera
from hand_tracking_toolkit.dataset import warp_image
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel


CLIPS_ROOT = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
OUTPUT_ROOT = REPO_ROOT / "runs/hot3d_hand_diagnosis/raw_vs_pinhole_visibility"
MANO_DIR = REPO_ROOT / "third_party/POEM-v2/assets/mano_v1_2/models"
LEFT_STREAM = "1201-1"
RIGHT_STREAM = "1201-2"
REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
FRAMES = [0, 5, 10, 15, 20]
CLIP_SIDES = {
    "clip-001852.tar": "left",
    "clip-001891.tar": "right",
}


def _gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.ndim == 3:
        return image[..., 0]
    return image


def _bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[-1] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)


def _make_pinhole(raw: camera.CameraModel) -> camera.PinholePlaneCameraModel:
    return camera.PinholePlaneCameraModel(
        width=raw.width,
        height=raw.height,
        f=[raw.f[0], raw.f[1]],
        c=raw.c,
        distort_coeffs=[],
        T_world_from_eye=raw.T_world_from_eye.copy(),
    )


def _visible(cam: camera.CameraModel, world: np.ndarray) -> tuple[bool, int, int]:
    eye = cam.world_to_eye(world)
    front = int((eye[REQUIRED, 2] > 0).sum())
    uv = cam.world_to_window(world)
    inside = int(cam.w_visible(uv[REQUIRED]).sum())
    visible = bool(front > 0 and inside > 0)
    return visible, front, inside


def _roi_stats(image: np.ndarray, box: list[float] | None) -> dict[str, Any]:
    gray = _gray(image)
    height, width = gray.shape
    if box is None:
        return {"box_exists": False, "mean": None, "median": None, "pixels": 0}
    x0 = int(max(0, min(width - 1, round(float(box[0])))))
    y0 = int(max(0, min(height - 1, round(float(box[1])))))
    x1 = int(max(0, min(width - 1, round(float(box[2])))))
    y1 = int(max(0, min(height - 1, round(float(box[3])))))
    if x1 <= x0 or y1 <= y0:
        return {"box_exists": True, "mean": None, "median": None, "pixels": 0}
    roi = gray[y0 : y1 + 1, x0 : x1 + 1]
    return {
        "box_exists": True,
        "mean": float(np.mean(roi)),
        "median": float(np.median(roi)),
        "pixels": int(roi.size),
    }


def _annotate(
    image: np.ndarray,
    cam: camera.CameraModel,
    world: np.ndarray,
    box: list[float] | None,
    visible: bool,
    view_name: str,
) -> np.ndarray:
    out = _bgr(image)
    height, width = out.shape[:2]
    if box is not None:
        x0 = int(round(float(box[0])))
        y0 = int(round(float(box[1])))
        x1 = int(round(float(box[2])))
        y1 = int(round(float(box[3])))
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 0), 2)
    uv = cam.world_to_window(world)
    for index in range(21):
        if not np.isfinite(uv[index]).all():
            continue
        u, v = float(uv[index, 0]), float(uv[index, 1])
        if u < -20 or v < -20 or u > width + 20 or v > height + 20:
            continue
        color = (0, 0, 255) if index == 0 else (255, 200, 0)
        if index in FINGERTIPS:
            color = (0, 165, 255)
        cv2.circle(out, (int(round(u)), int(round(v))), 4, color, -1)
    cv2.putText(
        out,
        f"{view_name} visible={str(visible).upper()}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _world_points(
    mano_model: MANOHandModel,
    shape_beta: np.ndarray,
    hand_payload: dict[str, Any],
    side: str,
) -> np.ndarray:
    beta = (
        torch.from_numpy(shape_beta.astype(np.float32))
        .float()
        .expand(1, -1)
    )
    theta = torch.from_numpy(
        np.asarray(hand_payload["mano_pose"]["thetas"], dtype=np.float32)
    ).float()[None]
    wrist = torch.from_numpy(
        np.asarray(hand_payload["mano_pose"]["wrist_xform"], dtype=np.float32)
    ).float()[None]
    hand_side = torch.tensor([0 if side == "left" else 1], dtype=torch.long)
    with torch.no_grad():
        _, landmarks = mano_model(
            beta,
            theta,
            wrist,
            is_right_hand=hand_side.bool(),
        )
    return landmarks.detach().cpu().numpy()[0].astype(np.float64)


def _process_clip(
    clip_name: str,
    side: str,
    mano_model: MANOHandModel,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[str]]:
    tar_path = CLIPS_ROOT / clip_name
    rows: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []
    titles: list[str] = []
    with tarfile.open(tar_path, "r") as tar:
        shape_payload = json.load(tar.extractfile("__hand_shapes.json__"))
        shape_beta = np.asarray(shape_payload["mano"], dtype=np.float32)
        for frame_number in FRAMES:
            hands = json.load(tar.extractfile(f"{frame_number:06d}.hands.json"))
            hand = hands.get(side)
            if not hand or "mano_pose" not in hand:
                rows.append(
                    {
                        "clip": clip_name,
                        "selected_hand": side,
                        "frame": frame_number,
                        "status": "missing_mano",
                    }
                )
                continue
            world = _world_points(mano_model, shape_beta, hand, side)
            cameras = json.load(tar.extractfile(f"{frame_number:06d}.cameras.json"))
            raw_left = camera.from_json(cameras[LEFT_STREAM])
            raw_right = camera.from_json(cameras[RIGHT_STREAM])
            pin_left = _make_pinhole(raw_left)
            pin_right = _make_pinhole(raw_right)
            raw_img_left = imageio.imread(
                tar.extractfile(f"{frame_number:06d}.image_{LEFT_STREAM}.jpg")
            )
            raw_img_right = imageio.imread(
                tar.extractfile(f"{frame_number:06d}.image_{RIGHT_STREAM}.jpg")
            )
            pin_img_left = warp_image(
                src_camera=raw_left,
                dst_camera=pin_left,
                src_image=np.asarray(raw_img_left),
                interpolation=cv2.INTER_LINEAR,
                depth_check=True,
            )
            pin_img_right = warp_image(
                src_camera=raw_right,
                dst_camera=pin_right,
                src_image=np.asarray(raw_img_right),
                interpolation=cv2.INTER_LINEAR,
                depth_check=True,
            )
            boxes = hand.get("boxes_amodal", {})
            box_left = boxes.get(LEFT_STREAM)
            box_right = boxes.get(RIGHT_STREAM)

            raw_left_visible, raw_left_front, raw_left_inside = _visible(
                raw_left, world
            )
            raw_right_visible, raw_right_front, raw_right_inside = _visible(
                raw_right, world
            )
            pin_left_visible, pin_left_front, pin_left_inside = _visible(
                pin_left, world
            )
            pin_right_visible, pin_right_front, pin_right_inside = _visible(
                pin_right, world
            )
            raw_left_roi = _roi_stats(raw_img_left, box_left)
            raw_right_roi = _roi_stats(raw_img_right, box_right)
            rows.append(
                {
                    "clip": clip_name,
                    "selected_hand": side,
                    "frame": int(frame_number),
                    "views": {
                        "left": {
                            "stream_id": LEFT_STREAM,
                            "metadata_box_exists": box_left is not None,
                            "raw_fisheye_visible": bool(raw_left_visible),
                            "raw_front_required_joints": raw_left_front,
                            "raw_inside_required_joints": raw_left_inside,
                            "pinhole_visible": bool(pin_left_visible),
                            "pinhole_front_required_joints": pin_left_front,
                            "pinhole_inside_required_joints": pin_left_inside,
                            "raw_roi_intensity": raw_left_roi,
                        },
                        "right": {
                            "stream_id": RIGHT_STREAM,
                            "metadata_box_exists": box_right is not None,
                            "raw_fisheye_visible": bool(raw_right_visible),
                            "raw_front_required_joints": raw_right_front,
                            "raw_inside_required_joints": raw_right_inside,
                            "pinhole_visible": bool(pin_right_visible),
                            "pinhole_front_required_joints": pin_right_front,
                            "pinhole_inside_required_joints": pin_right_inside,
                            "raw_roi_intensity": raw_right_roi,
                        },
                    },
                }
            )
            panels.extend(
                [
                    _annotate(raw_img_left, raw_left, world, box_left, raw_left_visible, "RAW LEFT"),
                    _annotate(raw_img_right, raw_right, world, box_right, raw_right_visible, "RAW RIGHT"),
                    _annotate(pin_img_left, pin_left, world, box_left, pin_left_visible, "PINHOLE LEFT"),
                    _annotate(pin_img_right, pin_right, world, box_right, pin_right_visible, "PINHOLE RIGHT"),
                ]
            )
            titles.extend(
                [
                    f"f{frame_number}\nraw L",
                    f"f{frame_number}\nraw R",
                    f"f{frame_number}\npin L",
                    f"f{frame_number}\npin R",
                ]
            )
    return rows, panels, titles


def _draw_panel_grid(
    panels: list[np.ndarray],
    titles: list[str],
    ncols: int,
    nrows: int,
) -> np.ndarray:
    target_w = 320
    target_h = 240
    rendered: list[np.ndarray] = []
    for panel, title in zip(panels, titles):
        resized = cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_AREA)
        label = np.full((36, target_w, 3), 20, dtype=np.uint8)
        cv2.putText(
            label,
            title.replace("\n", " "),
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rendered.append(np.vstack([resized, label]))
    rows_out: list[np.ndarray] = []
    for r in range(nrows):
        start = r * ncols
        end = start + ncols
        if end > len(rendered):
            end = len(rendered)
        chunk = rendered[start:end]
        if len(chunk) < ncols:
            chunk.extend(
                [np.zeros_like(chunk[0]) for _ in range(ncols - len(chunk))]
            )
        rows_out.append(np.hstack(chunk))
    return np.vstack(rows_out)


def _save_clip_image(
    clip_name: str,
    panels: list[np.ndarray],
    titles: list[str],
) -> Path:
    ncols = 4
    nrows = len(FRAMES)
    grid = _draw_panel_grid(panels, titles, ncols, nrows)
    path = OUTPUT_ROOT / f"{clip_name.replace('.tar', '')}_raw_vs_pinhole.png"
    cv2.imwrite(str(path), grid)
    return path


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mano_model = MANOHandModel(str(MANO_DIR))
    all_rows: list[dict[str, Any]] = []
    overview_panels: list[np.ndarray] = []
    overview_titles: list[str] = []
    for clip_name, side in CLIP_SIDES.items():
        print(f"processing {clip_name}", flush=True)
        rows, panels, titles = _process_clip(clip_name, side, mano_model)
        all_rows.extend(rows)
        _save_clip_image(clip_name, panels, titles)
        # First frame for the combined contact sheet.
        overview_panels.extend(panels[:4])
        overview_titles.extend(
            [
                f"{clip_name}\nraw L",
                f"{clip_name}\nraw R",
                f"{clip_name}\npin L",
                f"{clip_name}\npin R",
            ]
        )
    combined = _draw_panel_grid(overview_panels, overview_titles, 4, 2)
    combined_path = OUTPUT_ROOT / "raw_vs_pinhole_contact_sheet.png"
    cv2.imwrite(str(combined_path), combined)
    summary = {
        "schema_version": "1.0",
        "clips": list(CLIP_SIDES),
        "frames": FRAMES,
        "camera_protocol": "same_size_pinhole_from_source_fisheye",
        "raw_projection": "HOT3D official CameraModel.world_to_window on source fisheye",
        "rows": all_rows,
        "images": {
            "clip_001852": str(
                OUTPUT_ROOT / "clip-001852_raw_vs_pinhole.png"
            ),
            "clip_001891": str(
                OUTPUT_ROOT / "clip-001891_raw_vs_pinhole.png"
            ),
            "combined": str(combined_path),
        },
    }
    summary_path = OUTPUT_ROOT / "focused_raw_pinhole_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"summary -> {summary_path}")
    print(f"combined -> {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
