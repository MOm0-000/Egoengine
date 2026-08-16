#!/usr/bin/env python3
"""EgoForce E1 full official-path inference on the fixed v9 hardest subset.

This script intentionally avoids GT hand boxes, GT crop cameras, GT
handedness, GT depth, and temporal tracking. Inputs are the RAW fisheye RGB
frame and HOT3D/Aria fisheye calibration only. The official EgoForce YOLO
hand detector proposes the hand crop; EgoForce HALO then regresses the hand.

Run inside the v2s-egoforce environment:

    conda run -n v2s-egoforce python \
        scripts/run_egoforce_v10_full.py \
        --out runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tarfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EGOFORCE_ROOT = REPO_ROOT / "third_party/EgoForce"
if str(EGOFORCE_ROOT) not in sys.path:
    sys.path.insert(0, str(EGOFORCE_ROOT))
if str(EGOFORCE_ROOT / "demo") not in sys.path:
    sys.path.insert(0, str(EGOFORCE_ROOT / "demo"))

from settings import config as cfg
from camera_models import OVR624CameraModel
from models import HALO
from models.limb_model import LimbModel
from core import compute_camera_space_mesh, get_limb
from demo.demo_hand_arm_loader import DemoHandArmLoader
from ultralytics import YOLO
from mmdet.apis import DetInferencer


CLIPS_ROOT = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
SUBSET_MANIFEST = (
    REPO_ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
)
GAP_CSV = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
REQUIRED_JOINTS = np.asarray([0, 4, 8, 12, 16, 20], dtype=np.int64)
DEFAULT_OUT = REPO_ROOT / "runs/hot3d_hand_diagnosis/hand_backbone_benchmark_v10"


def create_square_bbox(keypoints, image_width, image_height, padding_factor):
    keypoints = np.asarray(keypoints, dtype=np.float32)[:, :2]
    if keypoints.size == 0:
        return (0, 0, image_width, image_height)
    mean = np.mean(keypoints, axis=0)
    centered = keypoints - mean
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    rotation_matrix = np.array(
        [
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)],
        ]
    )
    rotated = centered @ rotation_matrix.T
    x_min, y_min = np.min(rotated, axis=0)
    x_max, y_max = np.max(rotated, axis=0)
    width = x_max - x_min
    height = y_max - y_min
    side_length = max(width, height)
    side_length += 2 * side_length * padding_factor
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_min_padded = x_center - side_length / 2
    y_min_padded = y_center - side_length / 2
    corners_rotated = np.array(
        [
            [x_min_padded, y_min_padded],
            [x_min_padded + side_length, y_min_padded],
            [x_min_padded + side_length, y_min_padded + side_length],
            [x_min_padded, y_min_padded + side_length],
        ]
    )
    rotation_matrix_inv = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    corners = corners_rotated @ rotation_matrix_inv.T + mean
    x1 = float(np.min(corners[:, 0]))
    y1 = float(np.min(corners[:, 1]))
    x2 = float(np.max(corners[:, 0]))
    y2 = float(np.max(corners[:, 1]))
    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    return (x1, y1, x2, y2)


def compute_bbox(j2d, img_w, img_h, type="hand"):
    valid_joints = []
    for x, y in np.asarray(j2d, dtype=np.float32)[:, :2]:
        valid_joints.append(not (x < 0 or x >= img_w or y < 0 or y >= img_h))
    valid_ratio = sum(valid_joints) / max(1, len(valid_joints))
    if valid_ratio < 0.5:
        return False, np.array([-1, -1, -1, -1])
    bbox = create_square_bbox(
        np.asarray(j2d, dtype=np.float32)[:, :2],
        img_w,
        img_h,
        padding_factor=0.05,
    )
    return True, np.asarray(bbox, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    return parser.parse_args()


def long_no_seed_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gap"]) > 5:
                rows.append(
                    {
                        "clip": row["clip"],
                        "frame": int(row["frame"]),
                    }
                )
    return rows


def make_camera(calib_json: dict) -> OVR624CameraModel:
    calib = calib_json["calibration"]
    pp = np.asarray(calib["projection_params"], dtype=np.float32)
    assert len(pp) == 15 and "FISHEYE624" in calib["projection_model_type"]
    fx = fy = float(pp[0])
    cx = float(pp[1])
    cy = float(pp[2])
    coeffs = pp[3:].astype(np.float32)
    return OVR624CameraModel(
        f=np.asarray([fx, fy], dtype=np.float32),
        c=np.asarray([cx, cy], dtype=np.float32),
        params=coeffs,
        width=int(calib["image_width"]),
        height=int(calib["image_height"]),
    )


def yolo_candidates(yolo_model, rgb: np.ndarray, conf: float) -> list[dict]:
    result = yolo_model.predict(
        rgb,
        verbose=False,
        conf=conf,
        device=str(yolo_model.device) if hasattr(yolo_model, "device") else "cuda",
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int64)
    if result.keypoints is None:
        return []
    kpxy = result.keypoints.xy.cpu().numpy().astype(np.float32)

    h, w = rgb.shape[:2]
    candidates: list[dict] = []
    for i in range(len(xyxy)):
        cls_id = int(classes[i])
        handedness = "left" if cls_id in (0, 2) else "right"
        kp = np.asarray(kpxy[i], dtype=np.float32)
        ok, bbox = compute_bbox(kp, w, h, type="hand")
        if not ok:
            bbox = xyxy[i].copy()
        candidates.append(
            {
                "cls": cls_id,
                "handedness": handedness,
                "score": float(scores[i]),
                "keypoints": kp,
                "bbox": np.asarray(bbox, dtype=np.float32),
                "det_bbox": xyxy[i],
            }
        )
    return candidates


MMDET_ARM_CLASSES = ["left_forearm", "right_forearm", "left_hand", "right_hand"]


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float(inter / (area_a + area_b - inter + 1e-9))


def mmdet_forearm_for_side(
    inferencer,
    bgr: np.ndarray,
    hand_bbox: np.ndarray,
    side: str,
    score_thr: float = 0.3,
):
    """Run the official EgoForce MMDetection hand/forearm detector.

    It returns the best same-side forearm candidate associated with the
    selected hand box, matching the official demo's arm-attach logic.
    """
    predictions = inferencer(bgr)["predictions"][0]
    labels = np.asarray(predictions["labels"], dtype=np.int64)
    scores = np.asarray(predictions["scores"], dtype=np.float32)
    bboxes = np.asarray(predictions["bboxes"], dtype=np.float32)
    keypoints = (
        np.asarray(predictions["keypoints"], dtype=np.float32)
        if predictions.get("keypoints") is not None
        else None
    )
    target_label = 0 if side == "left" else 1
    best = None
    best_key = (-1.0, -1.0)
    for i, label in enumerate(labels):
        if int(label) != target_label or float(scores[i]) < score_thr:
            continue
        iou = iou_xyxy(hand_bbox, bboxes[i])
        key = (iou, float(scores[i]))
        if key > best_key:
            best_key = key
            best = {
                "bbox": bboxes[i],
                "keypoints": keypoints[i] if keypoints is not None else None,
                "score": float(scores[i]),
                "iou_with_hand": iou,
            }
    return best


def run_one_candidate(
    model,
    limb_model,
    loader,
    rgb: np.ndarray,
    candidate: dict,
    hand_type: str,
    arm_candidate: dict | None,
    device: torch.device,
):
    bbox = candidate["bbox"]
    kp = candidate["keypoints"]
    bounding_box = {
        "hand": {
            "bbox": bbox,
            "keypoint": kp,
            "score": candidate["score"],
        }
    }
    forearm_present = False
    if arm_candidate is not None:
        bounding_box["arm"] = {
            "bbox": np.asarray(arm_candidate["bbox"], dtype=np.float32),
            "keypoint": (
                np.asarray(arm_candidate["keypoints"], dtype=np.float32)
                if arm_candidate.get("keypoints") is not None
                else np.zeros((3, 2), dtype=np.float32)
            ),
            "score": arm_candidate["score"],
        }
        forearm_present = True
    data, meta = loader.transform(rgb, bounding_box)
    for key, value in list(meta.items()):
        if torch.is_tensor(value):
            meta[key] = value.unsqueeze(0)

    hand_crop = data["hand_crop"].unsqueeze(0).unsqueeze(0).to(device)
    hand_sparse_kpe = data["hand_sparse_kpe"].unsqueeze(0).unsqueeze(0).to(device)
    arm_crop = data["arm_crop"].unsqueeze(0).unsqueeze(0).to(device)
    arm_sparse_kpe = data["arm_sparse_kpe"].unsqueeze(0).unsqueeze(0).to(device)
    hand_type_t = data["hand_type"].view(1).to(device)

    with torch.no_grad():
        outputs = model(hand_crop, hand_sparse_kpe, arm_crop, arm_sparse_kpe)

    if os.environ.get("EGOFORCE_DEBUG_SHAPES") == "1":
        for key, value in outputs.items():
            print(key, tuple(value.shape), flush=True)

    betas = outputs["betas"].float()
    global_orient = outputs["global_orient"].float()
    hand_pose = outputs["hand_pose"].float()
    arm_shape = outputs["arm_shape"].float()
    arm_rot = outputs["arm_R"].float()
    kpts_2d = outputs["hand_kpts_2d"].squeeze(0).float()
    arm_kpts_2d = outputs["arm_kpts_2d"].squeeze(0).float()
    hand_kpt_w = outputs["hand_kpt_w"].squeeze(0).float()
    arm_kpt_w = outputs["arm_kpt_w"].squeeze(0).float()

    zero_transl = torch.zeros(global_orient.shape[0], global_orient.shape[1], 3, device=device)
    limb_output = get_limb(
        cfg,
        limb_model,
        global_orient,
        betas,
        hand_pose,
        zero_transl,
        hand_type_t,
        arm_shape,
        arm_rot,
    )

    limb_output.hand.crop_j2d = kpts_2d
    limb_output.arm.crop_j2d = arm_kpts_2d
    limb_output.hand.confidence = hand_kpt_w
    limb_output.arm.confidence = arm_kpt_w

    cs_limb_output = compute_camera_space_mesh(cfg, meta, limb_output)
    if os.environ.get("EGOFORCE_DEBUG_SHAPES") == "1":
        print("limb hand joints", tuple(limb_output.hand.joints.shape), flush=True)
        print("cs hand joints", tuple(cs_limb_output.hand.joints.shape), flush=True)
    pred_j3d = cs_limb_output.hand.joints
    pred_transl = cs_limb_output.transl
    pred_vertices = cs_limb_output.hand.vertices

    # The camera model here is a Python/numpy OVR624CameraModel. camera_to_uv
    # accepts numpy arrays, so move predicted camera-space joints back to CPU.
    pred_j3d_np = pred_j3d[0].detach().cpu().numpy().astype(np.float64)
    pred_j2d_np = loader.camera_model.camera_to_uv(pred_j3d_np).astype(np.float64)

    required = np.asarray(REQUIRED_JOINTS, dtype=np.int64)
    pred_req = pred_j2d_np[required]
    pred_z_req = pred_j3d_np[required, 2]
    final_valid = bool(
        np.isfinite(pred_req).all()
        and np.isfinite(pred_z_req).all()
        and (pred_z_req > 0).all()
        and (pred_req[:, 0] >= 0).all()
        and (pred_req[:, 0] < 640).all()
        and (pred_req[:, 1] >= 0).all()
        and (pred_req[:, 1] < 480).all()
    )
    transformer_valid = bool(
        torch.isfinite(outputs["hand_kpts_2d"]).all().item()
        and torch.isfinite(outputs["hand_kpt_w"]).all().item()
    )
    absolute_valid = bool(
        np.isfinite(pred_j3d_np).all()
        and np.isfinite(pred_transl.detach().cpu().numpy()).all()
        and (pred_j3d_np[required, 2] > 0).all()
    )
    hand_crop_valid = bool(
        float(bbox[2]) > float(bbox[0]) and float(bbox[3]) > float(bbox[1])
    )
    arm_crop_valid = False
    if forearm_present:
        arm_bbox = np.asarray(arm_candidate["bbox"], dtype=np.float32)
        arm_crop_valid = bool(
            float(arm_bbox[2]) > float(arm_bbox[0])
            and float(arm_bbox[3]) > float(arm_bbox[1])
        )
    return {
        "handedness": hand_type,
        "candidate_score": candidate["score"],
        "candidate_cls": candidate["cls"],
        "forearm_present": forearm_present,
        "forearm_score": (
            float(arm_candidate["score"]) if forearm_present else None
        ),
        "hand_crop_valid": hand_crop_valid,
        "forearm_crop_valid": arm_crop_valid,
        "transformer_valid": transformer_valid,
        "absolute_solver_valid": absolute_valid,
        "final_observation_valid": final_valid,
        "pred_j2d_raw": pred_j2d_np,
        "pred_j3d_cam": pred_j3d_np,
        "pred_transl": pred_transl[0, 0].detach().cpu().numpy().astype(np.float64),
        "pred_vertices": pred_vertices[0, 0].detach().cpu().numpy().astype(np.float64),
        "hand_kpt_w": hand_kpt_w[0].detach().cpu().numpy().astype(np.float64),
        "arm_kpt_w": arm_kpt_w[0].detach().cpu().numpy().astype(np.float64),
        "bbox": bbox,
    }


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg.POSE_3D.CHECKPOINT_PATH = str(
        EGOFORCE_ROOT / "_DATA/model_weights.pth"
    )
    cfg.MANO_PATH = str(EGOFORCE_ROOT / "_DATA/mano")
    cfg.DETECTION.HAND_PATH = str(EGOFORCE_ROOT / "_DATA/detector.torchscript")

    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    clip_to_entry = {item["clip"]: item for item in manifest["items"]}
    rows = long_no_seed_rows()
    if args.limit:
        rows = rows[: args.limit]
    print(f"EgoForce subset rows: {len(rows)}", flush=True)

    print("Loading EgoForce YOLO hand detector...", flush=True)
    yolo = YOLO(cfg.DETECTION.HAND_PATH, task="pose")

    print("Loading EgoForce MMDetection hand/forearm detector...", flush=True)
    cfg.DETECTION.HAND_ARM_PATH = str(EGOFORCE_ROOT / "_DATA/epoch_460.pth")
    mmdet_inferencer = DetInferencer(
        str(EGOFORCE_ROOT / "demo/rtmdet_tiny_8xb32-300e_combined_cutmix.py"),
        weights=cfg.DETECTION.HAND_ARM_PATH,
        device=str(device),
    )

    print("Loading EgoForce HALO model...", flush=True)
    model = HALO(cfg)
    checkpoint_path = cfg.POSE_3D.CHECKPOINT_PATH
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    model = model.to(device).eval()

    limb_model = LimbModel(cfg, device=device, use_pose_pca=False, n_components=5)
    for loader_side in ("left", "right"):
        setattr(limb_model, loader_side, None)

    left_loader = DemoHandArmLoader(
        cfg,
        None,
        undistort_inp=True,
        return_complete_image=False,
        hand_type="left",
    )
    right_loader = DemoHandArmLoader(
        cfg,
        None,
        undistort_inp=True,
        return_complete_image=False,
        hand_type="right",
    )

    candidate_rows: list[dict] = []
    predictions: dict = {}

    for row in rows:
        clip = row["clip"]
        frame = row["frame"]
        entry = clip_to_entry.get(clip)
        if entry is None:
            print("missing manifest entry", clip, file=sys.stderr)
            continue
        tar_path = CLIPS_ROOT / clip
        with tarfile.open(tar_path, "r") as tar:
            cams_json = json.load(
                tar.extractfile(f"{frame:06d}.cameras.json")
            )
            images_rgb = {}
            images_bgr = {}
            for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
                image_bytes = tar.extractfile(
                    f"{frame:06d}.image_{stream_id}.jpg"
                ).read()
                bgr = cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                images_bgr[view] = bgr
                images_rgb[view] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        for view, stream_id in (("left", "1201-1"), ("right", "1201-2")):
            predictions[(clip, frame, view)] = []
            cam_json = cams_json[stream_id]
            camera_model = make_camera(cam_json)
            loader = left_loader if view == "left" else right_loader
            loader.camera_model = camera_model
            loader.limb_model = limb_model

            rgb = images_rgb[view]
            bgr = images_bgr[view]
            candidates = yolo_candidates(yolo, rgb, args.yolo_conf)
            if not candidates:
                continue
            for cand in candidates:
                # Use the YOLO handedness as the model side prior (no GT).
                arm_candidate = mmdet_forearm_for_side(
                    mmdet_inferencer,
                    bgr,
                    np.asarray(cand["bbox"], dtype=np.float32),
                    cand["handedness"],
                )
                try:
                    out = run_one_candidate(
                        model,
                        limb_model,
                        loader,
                        rgb,
                        cand,
                        cand["handedness"],
                        arm_candidate,
                        device,
                    )
                except Exception as exc:  # keep a bad detector crop non-fatal
                    print(
                        f"skip candidate {clip}/{frame}/{view}: {exc}",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
                    continue
                candidate_rows.append(
                    {
                        "clip": clip,
                        "frame": frame,
                        "camera": view,
                        "candidate_cls": out["candidate_cls"],
                        "candidate_handedness": out["handedness"],
                        "candidate_score": out["candidate_score"],
                        "forearm_present": out["forearm_present"],
                        "forearm_score": out["forearm_score"],
                        "hand_crop_valid": out["hand_crop_valid"],
                        "forearm_crop_valid": out["forearm_crop_valid"],
                        "transformer_valid": out["transformer_valid"],
                        "absolute_solver_valid": out["absolute_solver_valid"],
                        "final_observation_valid": out["final_observation_valid"],
                        "mean_required_kpt_w": float(
                            np.mean(out["hand_kpt_w"][REQUIRED_JOINTS])
                        ),
                        "wrist_uv": out["pred_j2d_raw"][0].tolist(),
                        "pred_j2d_raw": out["pred_j2d_raw"].tolist(),
                        "pred_j3d_cam": out["pred_j3d_cam"].tolist(),
                        "pred_transl": out["pred_transl"].tolist(),
                        "bbox": np.asarray(out["bbox"], dtype=np.float32).tolist(),
                    }
                )
                predictions[(clip, frame, view)].append(
                    {
                        "candidate_cls": out["candidate_cls"],
                        "candidate_handedness": out["handedness"],
                        "candidate_score": out["candidate_score"],
                        "forearm_present": out["forearm_present"],
                        "forearm_score": out["forearm_score"],
                        "hand_crop_valid": out["hand_crop_valid"],
                        "forearm_crop_valid": out["forearm_crop_valid"],
                        "transformer_valid": out["transformer_valid"],
                        "absolute_solver_valid": out["absolute_solver_valid"],
                        "final_observation_valid": out["final_observation_valid"],
                        "pred_j2d_raw": out["pred_j2d_raw"].tolist(),
                        "pred_j3d_cam": out["pred_j3d_cam"].tolist(),
                        "pred_transl": out["pred_transl"].tolist(),
                        "hand_kpt_w": out["hand_kpt_w"].tolist(),
                        "arm_kpt_w": out["arm_kpt_w"].tolist(),
                        "bbox": np.asarray(out["bbox"], dtype=np.float32).tolist(),
                    }
                )

    # Keep the automatic top candidate per view: highest mean required keypoint
    # weight, then detector score.
    selected = {}
    for key, cands in predictions.items():
        if not cands:
            continue
        def rank(c):
            return (
                float(np.mean(np.asarray(c["hand_kpt_w"])[REQUIRED_JOINTS])),
                float(c["candidate_score"]),
            )
        selected[str(key)] = max(cands, key=rank)

    candidates_path = args.out / "egoforce_full_matrix.csv"
    with candidates_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(candidate_rows[0].keys()) if candidate_rows else [],
        )
        if candidate_rows:
            writer.writeheader()
            writer.writerows(candidate_rows)

    selected_path = args.out / "egoforce_full_selected_predictions.json"
    selected_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"wrote {candidates_path} and {selected_path}", flush=True)
    print(f"candidate count: {len(candidate_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
