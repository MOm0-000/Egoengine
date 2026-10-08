#!/usr/bin/env python3
"""Render frozen Brush Replay states beside their same-index RGB frames."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RL_ROOT.parent
OUTPUT = RL_ROOT / "runs/taco_brush_replay_failure_visual_compare_v1"
SOURCE = RL_ROOT / "runs/taco_brush_replay_failure_attribution_v1"
SCENE = RL_ROOT / "runs/taco_brush_controlled_physics_replay_v1/experiment_local_scene.xml"
RGB = Path(
    "/data_all/zzx/3.2RL/data/taco_v1/dev4/rgb/"
    "taco_brush_brush_bowl_20230927_027.mp4"
)
SECTIONS = {
    "brush_15_41": [15, 20, 25, 30, 35, 40, 41],
    "self_collision_44_45": [44, 45],
    "bowl_52_60": list(range(52, 61)),
}
PANEL_WIDTH = 640
PANEL_HEIGHT = 480


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def letterbox_rgb(image_bgr: np.ndarray) -> np.ndarray:
    scale = min(PANEL_WIDTH / image_bgr.shape[1], PANEL_HEIGHT / image_bgr.shape[0])
    width = int(round(image_bgr.shape[1] * scale))
    height = int(round(image_bgr.shape[0] * scale))
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    x0, y0 = (PANEL_WIDTH - width) // 2, (PANEL_HEIGHT - height) // 2
    panel[y0:y0 + height, x0:x0 + width] = resized
    return panel


def title(image_bgr: np.ndarray, text: str) -> np.ndarray:
    result = image_bgr.copy()
    cv2.rectangle(result, (0, 0), (PANEL_WIDTH, 42), (0, 0, 0), -1)
    cv2.putText(
        result, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return result


def decode_selected_frames(path: Path, endpoints: list[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open RGB video: {path}")
    wanted = set(endpoints)
    frames: dict[int, np.ndarray] = {}
    index = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame.copy()
            wanted.remove(index)
        index += 1
    capture.release()
    if wanted:
        raise RuntimeError(f"RGB video missing requested frames: {sorted(wanted)}")
    return frames


def camera(*, top: bool) -> mujoco.MjvCamera:
    value = mujoco.MjvCamera()
    value.lookat[:] = [0.60, 0.0, 0.76 if top else 0.78]
    value.distance = 0.70
    value.azimuth = 90
    value.elevation = -90 if top else -18
    return value


def hash_outputs(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "server_artifacts.sha256":
            rows.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "server_artifacts.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    rows_dir = OUTPUT / "rows"
    sheets_dir = OUTPUT / "sheets"
    rows_dir.mkdir()
    sheets_dir.mkdir()

    endpoints = sorted({value for values in SECTIONS.values() for value in values})
    rgb_frames = decode_selected_frames(RGB, endpoints)
    with np.load(SOURCE / "full_endpoint_states.npz", allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if qpos.shape[0] <= max(endpoints):
        raise ValueError("frozen Replay state archive does not cover requested endpoints")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    if qpos.shape[1] != model.nq:
        raise ValueError("Replay qpos/model mismatch")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
    front_camera, top_camera = camera(top=False), camera(top=True)
    rendered_rows: dict[int, np.ndarray] = {}
    row_artifacts = []
    for endpoint in endpoints:
        data.qpos[:] = qpos[endpoint]
        mujoco.mj_forward(model, data)
        simulation = []
        for label, view in (("Replay front", front_camera), ("Replay top", top_camera)):
            renderer.update_scene(data, camera=view)
            render_bgr = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
            simulation.append(title(render_bgr, f"{label} | endpoint {endpoint}"))
        human = title(letterbox_rgb(rgb_frames[endpoint]), f"Human RGB | frame {endpoint}")
        combined = np.concatenate([human, *simulation], axis=1)
        rendered_rows[endpoint] = combined
        path = rows_dir / f"endpoint_{endpoint:03d}_human_front_top.jpg"
        if not cv2.imwrite(str(path), combined, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write {path}")
        row_artifacts.append(artifact(path))
    renderer.close()

    sheet_artifacts = {}
    for section, section_endpoints in SECTIONS.items():
        sheet = np.concatenate([rendered_rows[value] for value in section_endpoints], axis=0)
        path = sheets_dir / f"{section}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write {path}")
        sheet_artifacts[section] = artifact(path)

    overview_endpoints = [15, 30, 40, 44, 45, 52, 57, 60]
    overview = np.concatenate([rendered_rows[value] for value in overview_endpoints], axis=0)
    overview_path = sheets_dir / "overview_keyframes.jpg"
    if not cv2.imwrite(str(overview_path), overview, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed to write {overview_path}")

    manifest = {
        "schema": "taco_brush_replay_failure_visual_compare_v1",
        "status": "READ_ONLY_VISUAL_COMPARISON",
        "frame_endpoint_alignment": "RGB frame N beside frozen physics Replay endpoint N",
        "camera_caveat": (
            "RGB is egocentric; Replay panels use fixed diagnostic front/top cameras. "
            "They are synchronized but not pixel-registered overlays."
        ),
        "sections": SECTIONS,
        "inputs": {
            "rgb": artifact(RGB),
            "scene": artifact(SCENE),
            "frozen_replay_states": artifact(SOURCE / "full_endpoint_states.npz"),
            "observer_parity": artifact(SOURCE / "observer_parity.json"),
            "script": artifact(Path(__file__)),
        },
        "outputs": {
            "overview": artifact(overview_path),
            "sheets": sheet_artifacts,
            "rows": row_artifacts,
        },
        "mutations": {
            "physics_steps": 0,
            "state": 0,
            "commands": 0,
            "geometry": 0,
            "training_steps": 0,
        },
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }
    write_json(OUTPUT / "manifest.json", manifest)
    (OUTPUT / "summary.md").write_text(
        "# Brush Replay human/robot visual comparison v1\n\n"
        "Each row is synchronized by index: human RGB frame, frozen Replay front view, "
        "and frozen Replay top view. The cameras differ, so this is a semantic visual "
        "comparison rather than a pixel overlay. No physics step or state mutation ran.\n",
        encoding="utf-8",
    )
    hash_outputs(OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT),
        "endpoints": endpoints,
        "overview": str(overview_path),
        "sheets": {key: value["path"] for key, value in sheet_artifacts.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
