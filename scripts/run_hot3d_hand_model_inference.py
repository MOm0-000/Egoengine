#!/usr/bin/env python3
"""Run WiLoR and HaMeR monocular adapters over the fixed HOT3D hand subset."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "runs/hot3d_hand_diagnosis"
MANIFEST_PATH = OUTPUT_ROOT / "subset_manifest.json"
GPU_IDS = [2, 3, 4, 5, 6, 7]


def task_command(
    task_index: int,
    run_dir: Path,
    env_name: str,
    model: str,
    camera_view: str,
    gpu: int,
) -> tuple[list[str], Path]:
    module = "video_to_spider.adapters.wilor" if model == "wilor" else "video_to_spider.adapters.hamer"
    if model == "wilor":
        args = [
            "--run-dir", str(run_dir),
            "--camera-view", camera_view,
            "--checkpoint", str(REPO_ROOT / "third_party/WiLoR/pretrained_models/wilor_final.ckpt"),
            "--model-config", str(REPO_ROOT / "third_party/WiLoR/pretrained_models/model_config.yaml"),
            "--detector-checkpoint", str(REPO_ROOT / "third_party/WiLoR/pretrained_models/detector.pt"),
            "--device", "cuda",
            "--overwrite",
        ]
    else:
        args = [
            "--run-dir", str(run_dir),
            "--camera-view", camera_view,
            "--checkpoint", str(REPO_ROOT / "third_party/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"),
            "--detector-checkpoint", str(REPO_ROOT / "third_party/WiLoR/pretrained_models/detector.pt"),
            "--device", "cuda",
            "--overwrite",
        ]
    command = [
        "conda", "run", "--no-capture-output", "-n", env_name,
        "python", "-m", module, *args,
    ]
    log_path = Path(f"/tmp/hot3d_model_{run_dir.name}_{model}_{camera_view}.log")
    return command, log_path


def run_task(item: dict[str, Any]) -> dict[str, Any]:
    index, env_name, model, camera_view, gpu, run_dir = item["index"], item["env"], item["model"], item["view"], item["gpu"], item["run_dir"]
    command, log_path = task_command(index, run_dir, env_name, model, camera_view, gpu)
    print(f"[task {index}] gpu={gpu} {env_name} {model} {camera_view} {run_dir.name}", flush=True)
    environment = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONNOUSERSITE": "1",
    }
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**__import__("os").environ, **environment},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "run": run_dir.name,
        "env": env_name,
        "model": model,
        "view": camera_view,
        "returncode": result.returncode,
        "log": str(log_path),
    }


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    index = 0
    for entry in manifest["items"]:
        run_dir = Path(entry["run_dir"]).resolve()
        for model, env_name in (("wilor", "v2s-wilor"), ("hamer", "v2s-hamer")):
            for view in ("left", "right"):
                tasks.append(
                    {
                        "index": index,
                        "env": env_name,
                        "model": model,
                        "view": view,
                        "run_dir": run_dir,
                        "gpu": GPU_IDS[index % len(GPU_IDS)],
                    }
                )
                index += 1
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (item["run"], item["model"], item["view"]))
    failed = [item for item in results if item["returncode"] != 0]
    summary_path = OUTPUT_ROOT / "model_inference_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "task_count": len(results),
                "failed_count": len(failed),
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"summary -> {summary_path}")
    if failed:
        for item in failed:
            print("FAILED", item)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
