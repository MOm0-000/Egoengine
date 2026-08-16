#!/usr/bin/env python3
"""Generate raw/P0/P1 comparison sheets for the FOV ablation."""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
HOT3D_TOOLKIT_REPO = Path(
    "/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo"
)
if str(HOT3D_TOOLKIT_REPO) not in sys.path:
    sys.path.insert(0, str(HOT3D_TOOLKIT_REPO))

from hand_tracking_toolkit import camera
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel


CLIPS_ROOT = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
P0_MANIFEST = REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
P1_RUNS = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/fov_rectification_ablation/runs/P1"
)
OUTPUT_ROOT = (
    REPO_ROOT / "runs/hot3d_hand_diagnosis/fov_rectification_ablation"
)
MANO_DIR = REPO_ROOT / "third_party/POEM-v2/assets/mano_v1_2/models"
LEFT_STREAM = "1201-1"
RIGHT_STREAM = "1201-2"
REQUIRED = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
SAMPLES = {
    "clip-001852.tar": "left",
    "clip-001891.tar": "right",
    "clip-002974.tar": "left",
}


def _bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[-1] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)


def _invert(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    translation = np.asarray(transform[:3, 3], dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.einsum("ij,...j->...i", transform[:3, :3], points) + transform[:3, 3]


def _project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    projected = np.einsum("ij,...j->...i", K, points)
    return projected[..., :2] / projected[..., 2:3]


def _gt_world(mano: MANOHandModel, clip: str, side: str, frame: int) -> np.ndarray:
    with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
        shape = json.load(tar.extractfile("__hand_shapes.json__"))
        beta_np = np.asarray(shape["mano"], dtype=np.float32)
        hand = json.load(tar.extractfile(f"{frame:06d}.hands.json"))[side]
        beta = torch.from_numpy(beta_np).float().expand(1, -1)
        theta = torch.from_numpy(
            np.asarray(hand["mano_pose"]["thetas"], dtype=np.float32)
        ).float()[None]
        wrist = torch.from_numpy(
            np.asarray(hand["mano_pose"]["wrist_xform"], dtype=np.float32)
        ).float()[None]
        hand_side = torch.tensor([0 if side == "left" else 1], dtype=torch.long)
        with torch.no_grad():
            _, landmarks = mano(
                beta,
                theta,
                wrist,
                is_right_hand=hand_side.bool(),
            )
        return landmarks.detach().cpu().numpy()[0].astype(np.float64)


def _annotate(
    image: np.ndarray,
    uv: np.ndarray,
    title: str,
) -> np.ndarray:
    out = _bgr(image)
    height, width = out.shape[:2]
    cv2.rectangle(out, (0, 0), (width - 1, height - 1), (180, 180, 180), 2)
    for index in range(21):
        u, v = float(uv[index, 0]), float(uv[index, 1])
        if u < -30 or v < -30 or u > width + 30 or v > height + 30:
            continue
        color = (0, 0, 255) if index == 0 else (255, 200, 0)
        if index in FINGERTIPS:
            color = (0, 165, 255)
        cv2.circle(out, (int(round(u)), int(round(v))), 4, color, -1)
    cv2.putText(
        out,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _resize(panel: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    return cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_AREA)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    mano = MANOHandModel(str(MANO_DIR))
    manifest = json.loads(P0_MANIFEST.read_text(encoding="utf-8"))["items"]
    by_clip = {item["clip"]: item for item in manifest}
    rows_out: list[np.ndarray] = []
    for clip, side in SAMPLES.items():
        frame = 0
        entry = by_clip[clip]
        p0_run = Path(entry["run_dir"])
        p1_run = P1_RUNS / p0_run.name
        world = _gt_world(mano, clip, side, frame)
        panels: list[np.ndarray] = []
        with tarfile.open(CLIPS_ROOT / clip, "r") as tar:
            raw_left_img = imageio.imread(
                tar.extractfile(f"{frame:06d}.image_{LEFT_STREAM}.jpg")
            )
            raw_right_img = imageio.imread(
                tar.extractfile(f"{frame:06d}.image_{RIGHT_STREAM}.jpg")
            )
            cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
            raw_left = camera.from_json(cams[LEFT_STREAM])
            raw_right = camera.from_json(cams[RIGHT_STREAM])
        panels.append(
            _annotate(
                raw_left_img,
                raw_left.world_to_window(world),
                "RAW LEFT",
            )
        )
        panels.append(
            _annotate(
                raw_right_img,
                raw_right.world_to_window(world),
                "RAW RIGHT",
            )
        )

        for run, label in ((p0_run, "P0"), (p1_run, "BEST P1")):
            left_img = cv2.imread(str(run / "frames/rgb/000000.png"))
            right_img = cv2.imread(str(run / "frames/right/000000.png"))
            K_left = np.load(run / "calibration/intrinsics.npy").astype(np.float64)
            K_right = np.load(
                run / "calibration/intrinsics_right.npy"
            ).astype(np.float64)
            T_left = np.load(run / "calibration/T_world_camera.npy").astype(np.float64)
            T_right = np.load(
                run / "calibration/T_world_camera_right.npy"
            ).astype(np.float64)
            left_cam = _transform_points(_invert(T_left[0]), world)
            right_cam = _transform_points(_invert(T_right[0]), world)
            left_uv = _project(K_left, left_cam)
            right_uv = _project(K_right, right_cam)
            panels.append(_annotate(left_img, left_uv, f"{label} LEFT"))
            panels.append(_annotate(right_img, right_uv, f"{label} RIGHT"))

        target_w, target_h = 360, 270
        panels = [_resize(p, target_w, target_h) for p in panels]
        rows_out.append(np.hstack(panels))

    comparison = np.vstack(rows_out)
    path = OUTPUT_ROOT / "comparison.png"
    cv2.imwrite(str(path), comparison)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
