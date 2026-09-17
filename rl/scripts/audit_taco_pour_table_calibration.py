"""Audit table evidence without fitting its height to robot penetrations."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from egoengine_repro.retarget.paper_audit import artifact, support_clearance, verify_artifacts

DATA = ROOT / "data/taco_v1/pour_bowl_plate"
SEQUENCE = "(pour in some, bowl, plate)/20230927_017"
EPISODE = "taco_pour_bowl_plate_20230927_017"
ROWS = (0, 30, 60, 90, 120, 150, 180, 197)


def depth_format_candidate(stream):
    """A format prerequisite only; not a depth-unit/registration certificate."""
    return stream.get("codec_name") == "ffv1" and stream.get("pix_fmt") == "gray16le"


def project(points, extrinsic, intrinsic):
    camera = np.asarray(points) @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
        raise ValueError("source points must be finite and in front of camera")
    image = camera @ intrinsic.T
    return image[:, :2] / image[:, 2:]


def triangulate(intrinsic, first, second, uv_first, uv_second):
    """Use known metric camera extrinsics; return points and geometric filters."""
    a, b = np.asarray(uv_first, dtype=float), np.asarray(uv_second, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or not len(a):
        raise ValueError("nonempty corresponding (N,2) image points required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite image correspondence")
    homogeneous = cv2.triangulatePoints(intrinsic @ first[:3], intrinsic @ second[:3], a.T, b.T).T
    usable = np.abs(homogeneous[:, 3]) > 1e-12
    points = np.zeros((len(a), 3))
    points[usable] = homogeneous[usable, :3] / homogeneous[usable, 3:]
    camera = [points @ e[:3, :3].T + e[:3, 3] for e in (first, second)]
    usable &= (camera[0][:, 2] > 1e-8) & (camera[1][:, 2] > 1e-8)
    error, angle = np.full(len(a), np.inf), np.zeros(len(a))
    if usable.any():
        reprojections = [project(points[usable], e, intrinsic) for e in (first, second)]
        error[usable] = np.maximum(np.linalg.norm(reprojections[0] - a[usable], axis=1),
                                    np.linalg.norm(reprojections[1] - b[usable], axis=1))
        centers = [np.linalg.inv(e)[:3, 3] for e in (first, second)]
        rays = [points[usable] - c for c in centers]
        rays = [r / np.linalg.norm(r, axis=1, keepdims=True) for r in rays]
        angle[usable] = np.rad2deg(np.arccos(np.clip((rays[0] * rays[1]).sum(-1), -1, 1)))
    return points, usable & (error <= 1) & (angle >= 1), error, angle


def table_feature_mask(rgb, excluded_polygons):
    # This is a disclosed Pour-only visual selection, not a general table detector.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([75, 55, 35], np.uint8), np.array([110, 255, 255], np.uint8))
    exclusion = np.zeros(mask.shape, np.uint8)
    for polygon in excluded_polygons:
        hull = cv2.convexHull(np.asarray(polygon, np.float32))
        cv2.fillConvexPoly(exclusion, np.rint(hull).astype(np.int32), 255)
    exclusion = cv2.dilate(exclusion, np.ones((31, 31), np.uint8))
    mask[exclusion > 0] = 0
    return cv2.erode(mask, np.ones((21, 21), np.uint8))


def inspect():
    camera_dir = DATA / "camera/Egocentric_Camera_Parameters" / SEQUENCE
    camera_path = camera_dir / "egocentric_frame_extrinsic.npy"
    intrinsic_path = camera_dir / "egocentric_intrinsic.txt"
    joints_path = DATA / "hand_poses/Hand_Poses" / SEQUENCE / "hand_joints.npy"
    rgb_path = DATA / "rgb" / f"{EPISODE}.mp4"
    depth_path = DATA / "depth_original" / f"{EPISODE}.avi"
    reference_path = ROOT / "runs/taco_pour_bimanual_gt_v1/human_reference.npz"
    paths = [camera_path, intrinsic_path, joints_path, rgb_path, depth_path, reference_path]
    objects, support = [], {}
    for role, object_id in (("tool", "022"), ("target", "135")):
        mesh_path = DATA / "object_models/object_models_released" / f"{object_id}_cm.obj"
        pose_path = DATA / "object_poses/Object_Poses" / SEQUENCE / f"{role}_{object_id}.npy"
        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh.apply_scale(.01)
        poses = np.load(pose_path, allow_pickle=False).astype(float)
        objects.append((mesh.vertices, poses))
        paths.extend([mesh_path, pose_path])
    preserved = [artifact(p) for p in paths]
    camera = np.load(camera_path, allow_pickle=False).astype(float)
    intrinsic = np.loadtxt(intrinsic_path)
    joints = np.load(joints_path, allow_pickle=False).astype(float)
    with np.load(reference_path, allow_pickle=False) as human:
        assumed_height = .72 - float(human["T_sim_world"][2, 3])
    for role, (vertices, poses) in zip(("tool", "target"), objects):
        clearance = support_clearance(vertices, poses, assumed_height)
        _, axes = np.linalg.eigh(np.cov(vertices.T))
        axis = poses[0, :3, :3] @ axes[:, 0]
        support[role] = dict(initial_clearance_m=float(clearance[0]), minimum_clearance_m=float(clearance.min()),
            maximum_clearance_m=float(clearance.max()), initial_20_rows_clearance_range_m=[float(clearance[:20].min()), float(clearance[:20].max())],
            initial_short_mesh_axis_angle_from_world_z_deg=float(np.rad2deg(np.arccos(np.clip(abs(axis[2]), 0, 1)))),
            measured_real_table_plane=False)
    ffprobe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
        "stream=codec_name,width,height,pix_fmt,nb_frames", "-of", "json", str(depth_path)],
        check=True, capture_output=True, text=True, timeout=30)
    stream = json.loads(ffprobe.stdout)["streams"][0]
    features, frame_records = {}, []
    capture = cv2.VideoCapture(str(rgb_path))
    detector = cv2.SIFT_create(nfeatures=10000, contrastThreshold=.003)
    for row in ROWS:
        capture.set(cv2.CAP_PROP_POS_FRAMES, row)
        ok, rgb = capture.read()
        if not ok:
            raise ValueError(f"cannot decode RGB source row {row}")
        polygons = [project(hand, camera[row], intrinsic) for hand in joints[row]]
        for vertices, poses in objects:
            world = vertices @ poses[row, :3, :3].T + poses[row, :3, 3]
            polygons.append(project(world, camera[row], intrinsic))
        mask = table_feature_mask(rgb, polygons)
        keypoints, descriptors = detector.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), mask)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        keep = [i for i, k in enumerate(keypoints) if distance[int(round(k.pt[1])), int(round(k.pt[0]))] >= k.size]
        features[row] = ([keypoints[i] for i in keep], descriptors[keep] if keep else None)
        frame_records.append(dict(source_row_zero_based=row, mask_pixels=int((mask > 0).sum()),
                                  sift_features=len(keypoints), support_inside_mask_features=len(keep)))
    capture.release()
    centers = np.linalg.inv(camera)[:, :3, 3]
    pairs, recovered = [], []
    matcher = cv2.BFMatcher()
    for first in ROWS:
        k0, d0 = features[first]
        for second in ROWS:
            k1, d1 = features[second]
            baseline = np.linalg.norm(centers[first] - centers[second])
            if second <= first or baseline < .02 or d0 is None or d1 is None or min(len(d0), len(d1)) < 2:
                continue
            forward = [m for m, n in matcher.knnMatch(d0, d1, k=2) if m.distance < .7 * n.distance]
            backward = {(m.trainIdx, m.queryIdx) for m, n in matcher.knnMatch(d1, d0, k=2) if m.distance < .7 * n.distance}
            matches = [m for m in forward if (m.queryIdx, m.trainIdx) in backward]
            record = dict(source_rows_zero_based=[first, second], baseline_m=float(baseline), mutual_matches=len(matches), triangulations_passed=0)
            if matches:
                uv0 = np.array([k0[m.queryIdx].pt for m in matches])
                uv1 = np.array([k1[m.trainIdx].pt for m in matches])
                points, passed, error, angle = triangulate(intrinsic, camera[first], camera[second], uv0, uv1)
                record["triangulations_passed"] = int(passed.sum())
                for index in np.flatnonzero(passed):
                    recovered.append(dict(source_rows_zero_based=[first, second], world_point_m=points[index].tolist(),
                        reprojection_error_px=float(error[index]), parallax_deg=float(angle[index])))
            pairs.append(record)
    verify_artifacts(preserved)
    raw_depth_report = ROOT / "runs/taco_pour_raw_depth_table_audit_v2/report.json"
    return dict(status="legacy_rgb_stereo_superseded_by_raw_depth_audit_no_scene_change", inputs=preserved,
        source_table_z_assumed_m=assumed_height,
        original_depth_path=str(depth_path),
        original_depth_path_exists=depth_path.is_file(),
        original_depth_stream=stream, original_depth_is_metric_format_candidate=depth_format_candidate(stream),
        depth_pixels_used_as_metric=False, object_support_consistency=support,
        maximum_camera_baseline_from_first_m=float(np.linalg.norm(centers - centers[0], axis=1).max()),
        rgb_stereo=dict(selection="Pour cyan pixels; projected hand/object convex hulls excluded; eroded patch support",
            selection_is_task_specific=True, frames=frame_records, pairs=pairs, candidates=recovered,
            parameters=dict(hsv_min=[75,55,35], hsv_max=[110,255,255], exclusion_dilation_px=31,
                table_erosion_px=21, sift_contrast=.003, ratio=.7, minimum_baseline_m=.02,
                maximum_reprojection_error_px=1, minimum_parallax_deg=1),
            plane_fit=None, independent_plane_established=False,
            explanation="Candidate matches are not a certified, spatially distributed set of static tabletop observations; no plane is adopted."),
        table_translation_modified=False, table_rotation_modified=False, table_footprint_modified=False,
        robot_penetration_used_to_fit_table=False,
        raw_metric_depth_audit_expected_path=str(raw_depth_report),
        next_evidence_needed="resolve depth/RGB/annotation residuals before changing the simulation table",
        code=artifact(Path(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], depth=report["original_depth_stream"],
        support=report["object_support_consistency"], frames=report["rgb_stereo"]["frames"],
        stereo_candidates=report["rgb_stereo"]["candidates"]), indent=2))


if __name__ == "__main__":
    main()
