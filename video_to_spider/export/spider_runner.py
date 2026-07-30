"""Run the SPIDER preprocessing/IK/MJWP chain while preserving contact variants."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .spider import DATASET_NAME


def _run(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    record = {
        "command": command, "returncode": result.returncode,
        "runtime_s": float(time.monotonic() - started), "log": str(log_path),
    }
    if result.returncode != 0:
        raise RuntimeError(f"SPIDER command failed ({result.returncode}); see {log_path}")
    return record


def _contact_difference(visual_path: Path, spider_path: Path) -> dict[str, Any]:
    with np.load(visual_path, allow_pickle=False) as visual, np.load(spider_path, allow_pickle=False) as detected:
        report = {}
        for side in ("right", "left"):
            first = visual[f"contact_{side}"].astype(bool)
            second = detected[f"contact_{side}"].astype(bool)
            common_frames = min(first.shape[0], second.shape[0])
            common_fingers = min(first.shape[1], second.shape[1])
            first_common = first[:common_frames, :common_fingers]
            second_common = second[:common_frames, :common_fingers]
            first_position = visual[f"contact_pos_{side}"][:common_fingers]
            second_position = detected[f"contact_pos_{side}"][:common_fingers]
            report[side] = {
                "visual_shape": list(first.shape), "spider_shape": list(second.shape),
                "visual_contact_rate": float(first.mean()),
                "spider_contact_rate": float(second.mean()),
                "binary_disagreement_rate_common": float(np.not_equal(first_common, second_common).mean()),
                "visual_only_rate_common": float((first_common & ~second_common).mean()),
                "spider_only_rate_common": float((~first_common & second_common).mean()),
                "contact_position_l2_m_common": np.linalg.norm(
                    first_position - second_position, axis=-1
                ).tolist(),
                "shape_mismatch": bool(first.shape != second.shape),
            }
    return report


def _mjwp_metrics(path: Path, log_path: Path | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {key: np.asarray(artifact[key]) for key in artifact.files}
    metrics: dict[str, Any] = {"keys": sorted(arrays), "record_count": int(next(iter(arrays.values())).shape[0])}
    for key in ("obj_pos_dist", "obj_quat_dist", "pos_dist", "quat_dist"):
        if key in arrays:
            values = arrays[key].astype(np.float64)
            metrics[key] = {
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)), "max": float(np.max(values)),
            }
    position = metrics.get("obj_pos_dist", metrics.get("pos_dist"))
    rotation = metrics.get("obj_quat_dist", metrics.get("quat_dist"))
    if (position is None or rotation is None) and log_path is not None and log_path.exists():
        match = re.search(
            r"Final object tracking error: pos=([0-9.eE+-]+), quat=([0-9.eE+-]+)",
            log_path.read_text(encoding="utf-8"),
        )
        if match:
            position = {"mean": float(match.group(1)), "source": "SPIDER final log"}
            rotation = {"mean": float(match.group(2)), "source": "SPIDER final log"}
            metrics["obj_pos_dist"] = position
            metrics["obj_quat_dist"] = rotation
    metrics["paper_threshold_success"] = bool(
        position is not None and rotation is not None
        and position["mean"] < 0.1 and rotation["mean"] < 0.5
    )
    metrics["paper_thresholds"] = {"position_m": 0.1, "rotation_rad": 0.5}
    return metrics


def run_spider_chain(
    *, dataset_root: str | Path, spider_root: str | Path, task: str, data_id: int,
    embodiment_type: str, robot_type: str = "xhand", gpu: int = 0,
    run_mjwp: bool = True, save_video: bool = True, ik_end_idx: int = -1,
    mjwp_num_samples: int = 256, mjwp_iterations: int = 8,
) -> Path:
    dataset = Path(dataset_root).resolve()
    spider = Path(spider_root).resolve()
    mano_dir = dataset / "processed" / DATASET_NAME / "mano" / embodiment_type / task / str(data_id)
    robot_dir = dataset / "processed" / DATASET_NAME / robot_type / embodiment_type / task / str(data_id)
    keypoints = mano_dir / "trajectory_keypoints.npz"
    if not keypoints.exists():
        raise FileNotFoundError(keypoints)
    logs = robot_dir / "pipeline_logs"
    visual_contact = mano_dir / "trajectory_keypoints_visual_contact.npz"
    detected_contact = mano_dir / "trajectory_keypoints_spider_contact.npz"
    shutil.copy2(keypoints, visual_contact)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "UV_CACHE_DIR": "/tmp/video-to-spider-uv-cache",
        "PYTHONPATH": str(spider),
        # MuJoCo otherwise falls back to GLFW/X11 even when SPIDER is invoked
        # with --no-show-viewer. EGL provides deterministic offscreen rendering
        # for the M4 simulation videos on headless GPU nodes.
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
    })
    base = ["uv", "run", "--frozen", "--no-sync", "python", "-m"]
    common = [
        "--dataset-dir", str(dataset), "--dataset-name", DATASET_NAME,
        "--embodiment-type", embodiment_type, "--task", task, "--data-id", str(data_id),
    ]
    commands = []
    commands.append(_run(
        base + ["spider.preprocess.decompose_fast"] + common + ["--robot-type", robot_type],
        cwd=spider, env=environment, log_path=logs / "01_decompose_fast.log",
    ))
    try:
        commands.append(_run(
            base + ["spider.preprocess.detect_contact"] + common + ["--no-show-viewer", "--no-save-video"],
            cwd=spider, env=environment, log_path=logs / "02_detect_contact.log",
        ))
        shutil.copy2(keypoints, detected_contact)
        contact_difference = _contact_difference(visual_contact, detected_contact)
    finally:
        # The visual contact is the primary V1 observation. Always restore it,
        # including when SPIDER emits a malformed or shape-mismatched result.
        shutil.copy2(visual_contact, keypoints)
    commands.append(_run(
        base + ["spider.preprocess.generate_xml"] + common
        + ["--robot-type", robot_type, "--no-show-viewer"],
        cwd=spider, env=environment, log_path=logs / "03_generate_xml.log",
    ))
    commands.append(_run(
        base + ["spider.preprocess.ik_fast"] + common
        + ["--robot-type", robot_type, "--end-idx", str(ik_end_idx), "--no-show-viewer"]
        + (["--save-video"] if save_video else ["--no-save-video"]),
        cwd=spider, env=environment, log_path=logs / "04_ik_fast.log",
    ))
    trajectory_kinematic = robot_dir / "trajectory_kinematic.npz"
    scene = robot_dir.parent / "scene.xml"
    if not trajectory_kinematic.exists() or not scene.exists():
        raise RuntimeError("SPIDER IK did not produce scene.xml and trajectory_kinematic.npz")
    mjwp_path = robot_dir / "trajectory_mjwp.npz"
    if run_mjwp:
        mjwp_command = [
            "uv", "run", "--frozen", "--no-sync", "python", "examples/run_mjwp.py",
            "+override=gigahand_fast", f"dataset_dir={dataset}", f"dataset_name={DATASET_NAME}",
            f"robot_type={robot_type}", f"embodiment_type={embodiment_type}", f"task={task}",
            f"data_id={data_id}", f"model_path={scene}", f"data_path={trajectory_kinematic}",
            f"output_dir={robot_dir}", "device=cuda:0", "show_viewer=false", "viewer=mujoco",
            f"save_video={'true' if save_video else 'false'}", "save_rerun=false", "save_viser=false",
            f"num_samples={mjwp_num_samples}", f"max_num_iterations={mjwp_iterations}",
            "horizon=0.4", "ctrl_dt=0.2", "knot_dt=0.2", "+sanity_check_seconds=0.0",
        ]
        commands.append(_run(
            mjwp_command, cwd=spider, env=environment, log_path=logs / "05_mjwp.log"
        ))
        if not mjwp_path.exists():
            raise RuntimeError(f"MJWP completed without expected artifact: {mjwp_path}")
    ik_video = robot_dir / "visualization_ik.mp4"
    mjwp_video = robot_dir / "visualization_mjwp.mp4"
    if save_video and not ik_video.exists():
        raise RuntimeError(f"SPIDER IK completed without expected simulation video: {ik_video}")
    if run_mjwp and save_video and not mjwp_video.exists():
        raise RuntimeError(f"MJWP completed without expected simulation video: {mjwp_video}")
    report = {
        "schema_version": "1.0", "dataset_root": str(dataset), "dataset_name": DATASET_NAME,
        "task": task, "data_id": data_id, "embodiment_type": embodiment_type,
        "robot_type": robot_type, "gpu_physical_index": gpu, "commands": commands,
        "artifacts": {
            "visual_contact": str(visual_contact), "spider_detected_contact": str(detected_contact),
            "scene": str(scene), "trajectory_kinematic": str(trajectory_kinematic),
            "ik_video": str(ik_video) if save_video else None,
            "trajectory_mjwp": str(mjwp_path) if run_mjwp else None,
            "mjwp_video": str(mjwp_video) if run_mjwp and save_video else None,
        },
        "contact_crosscheck": contact_difference,
        "mjwp_metrics": _mjwp_metrics(mjwp_path, logs / "05_mjwp.log") if run_mjwp else None,
        "current_spider_thresholds": {"position_m": 0.1, "rotation_rad": 0.5},
    }
    report_path = robot_dir / "spider_run_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-id", type=int, required=True)
    parser.add_argument("--embodiment-type", choices=["right", "left", "bimanual"], required=True)
    parser.add_argument("--robot-type", default="xhand")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-mjwp", action="store_true")
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--ik-end-idx", type=int, default=-1)
    parser.add_argument("--mjwp-num-samples", type=int, default=256)
    parser.add_argument("--mjwp-iterations", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(run_spider_chain(
        dataset_root=args.dataset_root, spider_root=args.spider_root, task=args.task,
        data_id=args.data_id, embodiment_type=args.embodiment_type, robot_type=args.robot_type,
        gpu=args.gpu, run_mjwp=not args.no_mjwp, save_video=not args.no_save_video,
        ik_end_idx=args.ik_end_idx, mjwp_num_samples=args.mjwp_num_samples,
        mjwp_iterations=args.mjwp_iterations,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
