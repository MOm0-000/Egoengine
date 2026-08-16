#!/usr/bin/env python3
"""v16 real weak-view recovery from real automatic strong wrist."""

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
OUT = ROOT / "runs/hot3d_hand_diagnosis/true_auto_stereo_ba_v16"
V13 = ROOT / "runs/hot3d_hand_diagnosis/ego_fullframe_hand_finder_v13"
CLIPS = Path("/data_all/zzx/egoengine/hot3d_clips/train_aria")
GAP_CSV = (
    ROOT
    / "runs/hot3d_hand_diagnosis/stereo_hand_observation_v5/temporal_gap_analysis.csv"
)
MANIFEST = ROOT / "runs/hot3d_hand_diagnosis/subset_manifest.json"
STRONG_CSV = OUT / "real_automatic_strong_predictions.csv"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/EgoForce"))
sys.path.insert(
    0,
    "/data_all/zzx/egoengine/third_party/hot3d/hand_tracking_toolkit_repo",
)

from hand_tracking_toolkit import camera
from hand_tracking_toolkit.dataset import warp_image
from camera_models.fisheye624 import OVR624Distortion


def long_rows():
    rows = []
    with GAP_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["gap"]) > 5:
                rows.append((r["clip"], int(r["frame"])))
    return rows


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jload(s):
    return json.loads(s) if s not in ("", None) else None


def normalize(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def local_camera_from_center(raw_cam, center_uv, fov_deg=25.0, size=256):
    q = (np.asarray(center_uv, float) - np.asarray(raw_cam.c)) / np.asarray(raw_cam.f)
    q = OVR624Distortion(np.asarray(raw_cam.distort, dtype=np.float32)).inverse_evaluate(q)
    eye_dir = normalize(raw_cam.unproject(q))
    z_axis = normalize(raw_cam.orient() @ eye_dir)
    raw_y = raw_cam.orient()[:, 1]
    y_axis = normalize(raw_y - z_axis * float(np.dot(raw_y, z_axis)))
    x_axis = normalize(np.cross(y_axis, z_axis))
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = raw_cam.pos()
    f = size / 2.0 / math.tan(math.radians(fov_deg) / 2.0)
    return camera.PinholePlaneCameraModel(
        width=size,
        height=size,
        f=(f, f),
        c=((size - 1) / 2.0, (size - 1) / 2.0),
        distort_coeffs=[],
        T_world_from_eye=T,
    )


def raw_ray(cam, uv):
    q = (np.asarray(uv, float) - np.asarray(cam.c)) / np.asarray(cam.f)
    q = OVR624Distortion(np.asarray(cam.distort, dtype=np.float32)).inverse_evaluate(q)
    v = cam.unproject(q)
    v = v / np.linalg.norm(v)
    return cam.pos(), cam.orient() @ v


def load_model():
    from scripts.run_stereo_crop_inference import _load_model
    model, cfg, detector, dataset_cls, recursive_to = _load_model("wilor")
    return model, cfg, dataset_cls, recursive_to


def camera_translation(pred_cam, box_center, box_size, K):
    focal = float((K[0, 0] + K[1, 1]) * 0.5)
    bs = box_size * pred_cam[:, 0] + 1e-9
    tz = 2.0 * focal / bs
    tx = 2.0 * (box_center[:, 0] - float(K[0, 2])) / bs + pred_cam[:, 1]
    ty = 2.0 * (box_center[:, 1] - float(K[1, 2])) / bs + pred_cam[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def infer_boxes(model, cfg, dataset_cls, recursive_to, device, img, boxes, right, K):
    dataset = dataset_cls(cfg, img, np.asarray(boxes, dtype=np.float32), np.asarray(right, dtype=np.float32), rescale_factor=2.0, fp16=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    batch = next(iter(loader)); batch = recursive_to(batch, device)
    with torch.no_grad(): out = model(batch)
    pred_cam = out["pred_cam"].clone(); pred_cam[:, 1] *= (2 * batch["right"] - 1)
    cam_t = camera_translation(pred_cam, batch["box_center"].float(), batch["box_size"].float(), K).detach().cpu().numpy()
    root = out["pred_keypoints_3d"].detach().cpu().numpy().copy()
    flip_x = (2.0 * np.asarray(right, dtype=np.float32) - 1.0)[:, None, None]
    root[..., 0] *= flip_x[..., 0]
    joints = root + cam_t[:, None, :]
    uv = np.zeros((joints.shape[0], joints.shape[1], 2))
    for i in range(joints.shape[0]):
        uv[i] = (np.einsum("ij,...j->...i", K, joints[i])[:, :2] / joints[i][..., 2:3])
    return joints[0], uv[0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model, cfg, dataset_cls, recursive_to = load_model(); model = model.to(device).eval()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clip_entry = {i["clip"]: i for i in manifest["items"]}
    v13_views = {
        (r["clip"], int(r["frame"]), r["camera"]): r
        for r in read_csv(V13 / "per_view_hand_finder_metrics.csv")
    }
    strong = {
        (r["clip"], int(r["frame"]), r["camera"]): r
        for r in read_csv(STRONG_CSV)
    }
    rows = long_rows()
    weak_rows = []
    for clip, frame in rows:
        sl = v13_views.get((clip, frame, "left"), {}).get("interformer_m1_center") == "True"
        sr = v13_views.get((clip, frame, "right"), {}).get("interformer_m1_center") == "True"
        if sl == sr:
            continue
        sv, wv = ("left", "right") if sl else ("right", "left")
        srow = strong.get((clip, frame, sv))
        if srow is None:
            continue
        selected_side = clip_entry[clip]["selected_side"]
        right = 0 if selected_side == "left" else 1
        with tarfile.open(CLIPS / clip, "r") as tar:
            cams = json.load(tar.extractfile(f"{frame:06d}.cameras.json"))
            cs = camera.from_json(cams["1201-1" if sv == "left" else "1201-2"])
            cw = camera.from_json(cams["1201-2" if wv == "right" else "1201-1"])
            b = tar.extractfile(f"{frame:06d}.image_{'1201-2' if wv == 'right' else '1201-1'}.jpg").read()
        weak_img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        ray = raw_ray(cs, np.array(jload(srow["raw_wrist"])))
        weak_center = cw.world_to_window(ray[0] + ray[1] * 0.5)
        local_cam = local_camera_from_center(cw, weak_center)
        crop = warp_image(cw, local_cam, weak_img).astype(np.uint8)
        K = np.array([[local_cam.f[0], 0, 127.5], [0, local_cam.f[1], 127.5], [0, 0, 1]])
        try:
            joints_cam, joints_2d = infer_boxes(model, cfg, dataset_cls, recursive_to, device, crop, [[16, 16, 240, 240]], [right], K)
        except Exception as exc:
            print("skip weak", clip, frame, wv, exc, file=sys.stderr)
            continue
        world = local_cam.eye_to_world(joints_cam)
        weak_rows.append({
            "clip": clip, "frame": frame, "camera": wv,
            "raw_center": weak_center.tolist(),
            "raw_wrist": cw.world_to_window(world[0]).tolist(),
            "joints_2d_crop": joints_2d.tolist(),
            "joints_cam": joints_cam.tolist(),
            "T_world_from_eye": local_cam.T_world_from_eye.tolist(),
            "f": float(local_cam.f[0]),
            "selected_side": selected_side,
        })
    out_path = OUT / "real_automatic_weak_predictions.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(weak_rows[0].keys()))
        writer.writeheader(); writer.writerows(weak_rows)
    print(f"wrote {out_path} rows={len(weak_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
