#!/usr/bin/env python3
"""v18 pose-guided one-step re-cropping."""

from __future__ import annotations

import csv
import json
import math
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path("/data_all/zzx/egoengine/video_to_spider")
OUT = ROOT / "runs/hot3d_hand_diagnosis/pose_guided_recrop_v18"
V16 = ROOT / "runs/hot3d_hand_diagnosis/true_auto_stereo_ba_v16"
CLIPS = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/EgoForce"))
sys.path.insert(
    0,
    "/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo",
)

from hand_tracking_toolkit import camera
from hand_tracking_toolkit.dataset import warp_image
from camera_models.fisheye624 import OVR624Distortion


MCP = [1, 5, 9, 13, 17]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jload(s):
    return json.loads(s) if s not in ("", None) else None


def normalize(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def local_camera(raw_cam, center, fov=25.0):
    q = (np.asarray(center, float) - np.asarray(raw_cam.c)) / np.asarray(raw_cam.f)
    q = OVR624Distortion(np.asarray(raw_cam.distort, dtype=np.float32)).inverse_evaluate(q)
    eye_dir = normalize(raw_cam.unproject(q))
    z = normalize(raw_cam.orient() @ eye_dir)
    raw_y = raw_cam.orient()[:, 1]
    y = normalize(raw_y - z * float(np.dot(raw_y, z)))
    x = normalize(np.cross(y, z))
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = raw_cam.pos()
    f = 256 / 2.0 / math.tan(math.radians(fov) / 2.0)
    return camera.PinholePlaneCameraModel(width=256, height=256, f=(f, f), c=(127.5, 127.5), distort_coeffs=[], T_world_from_eye=T)


def load_model():
    from scripts.run_stereo_crop_inference import _load_model
    model, cfg, detector, dataset_cls, recursive_to = _load_model("wilor")
    return model, cfg, dataset_cls, recursive_to


def camera_translation(pred_cam, box_center, box_size, K):
    focal = float((K[0, 0] + K[1, 1]) * 0.5)
    bs = box_size * pred_cam[:, 0] + 1e-9
    return torch.stack([
        2.0 * (box_center[:, 0] - float(K[0, 2])) / bs + pred_cam[:, 1],
        2.0 * (box_center[:, 1] - float(K[1, 2])) / bs + pred_cam[:, 2],
        2.0 * focal / bs,
    ], dim=-1)


def infer(model, cfg, ds, rec, device, img, right, K):
    dataset = ds(cfg, img, np.array([[16, 16, 240, 240]], np.float32), np.array([right], np.float32), rescale_factor=2.0, fp16=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader)); batch = rec(batch, device)
    with torch.no_grad(): out = model(batch)
    pred_cam = out["pred_cam"].clone(); pred_cam[:, 1] *= (2 * batch["right"] - 1)
    cam_t = camera_translation(pred_cam, batch["box_center"].float(), batch["box_size"].float(), K).detach().cpu().numpy()
    root = out["pred_keypoints_3d"].detach().cpu().numpy().copy()
    flip = (2 * np.asarray([right], np.float32) - 1)[:, None, None]
    root[..., 0] *= flip[..., 0]
    joints = root + cam_t[:, None, :]
    uv = (np.einsum("ij,...j->...i", K, joints[0])[:, :2] / joints[0][..., 2:3])
    return joints[0], uv


def transform(T, p):
    return T[:3, :3] @ p + T[:3, 3]


def centers_from_first_pass(row):
    T = np.asarray(jload(row["T_world_from_eye"]), dtype=np.float64)
    joints = np.asarray(jload(row["joints_cam"]), dtype=np.float64)
    world = np.stack([transform(T, j) for j in joints])
    wrist = world[0]
    mcp = world[MCP].mean(axis=0)
    g1 = wrist
    g2 = mcp
    g3 = 0.25 * wrist + 0.75 * mcp
    g4 = 0.5 * wrist + 0.5 * world[9]
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, cfg, ds, rec = load_model(); model = model.to(device).eval()
    r0rows = []
    for name in (
        "real_automatic_strong_predictions.csv",
        "real_automatic_weak_predictions.csv",
    ):
        r0rows += read_csv(V16 / name)
    records = []
    for row in r0rows:
        clip = row["clip"]; frame = int(row["frame"]); view = row["camera"]
        right = 0 if row["selected_side"] == "left" else 1
        with tarfile.open(CLIPS / clip, "r") as tar:
            cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
            raw_cam = camera.from_json(cams["1201-1" if view == "left" else "1201-2"])
            b = tar.extractfile(f"{frame:06d}.image_{'1201-1' if view == 'left' else '1201-2'}.jpg").read()
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        centers = centers_from_first_pass(row)
        for method in ("G2", "G3", "G4"):
            center_world = centers[method]
            center_raw = raw_cam.world_to_window(center_world)
            local = local_camera(raw_cam, center_raw)
            crop = warp_image(raw_cam, local, img).astype(np.uint8)
            K = np.array([[local.f[0], 0, 127.5], [0, local.f[1], 127.5], [0, 0, 1]])
            try:
                joints, uv = infer(model, cfg, ds, rec, device, crop, right, K)
            except Exception:
                continue
            world = local.eye_to_world(joints)
            records.append({
                "clip": clip, "frame": frame, "camera": view,
                "method": method, "center_raw": json.dumps(center_raw.tolist()),
                "joints_cam": json.dumps(joints.tolist()),
                "joints_2d_crop": json.dumps(uv.tolist()),
                "T_world_from_eye": json.dumps(local.T_world_from_eye.tolist()),
                "f": float(local.f[0]),
                "raw_wrist": json.dumps(raw_cam.world_to_window(world[0]).tolist()),
            })
    out_path = OUT / "recrop_predictions.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader(); writer.writerows(records)
    print(f"wrote {out_path} rows={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
