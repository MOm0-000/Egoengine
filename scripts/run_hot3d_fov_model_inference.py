#!/usr/bin/env python3
"""Run WiLoR and HaMeR over P1/P2 FOV rectification runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_RUNS = (
    REPO_ROOT
    / "runs/hot3d_hand_diagnosis/fov_rectification_ablation/runs"
)
GPU_IDS = [2, 3, 4, 5, 6, 7]


def _command(
    run_dir: Path,
    env_name: str,
    model: str,
    camera_view: str,
) -> list[str]:
    module = (
        "video_to_spider.adapters.wilor"
        if model == "wilor"
        else "video_to_spider.adapters.hamer"
    )
    if model == "wilor":
        args = [
            "--run-dir", str(run_dir),
            "--camera-view", camera_view,
            "--checkpoint",
            str(REPO_ROOT / "third_party/WiLoR/pretrained_models/wilor_final.ckpt"),
            "--model-config",
            str(REPO_ROOT / "third_party/WiLoR/pretrained_models/model_config.yaml"),
            "--detector-checkpoint",
            str(REPO_ROOT / "third_party/WiLoR/pretrained_models/detector.pt"),
            "--device", "cuda",
            "--overwrite",
        ]
    else:
        args = [
            "--run-dir", str(run_dir),
            "--camera-view", camera_view,
            "--checkpoint",
            str(REPO_ROOT / "third_party/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"),
            "--detector-checkpoint",
            str(REPO_ROOT / "third_party/WiLoR/pretrained_models/detector.pt"),
            "--device", "cuda",
            "--overwrite",
        ]
    return [
        "conda", "run", "--no-capture-output", "-n", env_name,
        "python", "-m", module, *args,
    ]


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    command = _command(
        task["run_dir"],
        task["env"],
        task["model"],
        task["view"],
    )
    log_path = Path(
        f"/tmp/hot3d_fov_{task['protocol']}_{task['run_dir'].name}_{task['model']}_{task['view']}.log"
    )
    environment = {
        "CUDA_VISIBLE_DEVICES": str(task["gpu"]),
        "PYTHONNOUSERSITE": "1",
    }
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, **environment},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "protocol": task["protocol"],
        "run": task["run_dir"].name,
        "env": task["env"],
        "model": task["model"],
        "view": task["view"],
        "returncode": result.returncode,
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", default="P1,P2")
    args = parser.parse_args()
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    all_results: list[dict[str, Any]] = []
    for protocol in protocols:
        protocol_root = ABLATION_RUNS / protocol
        manifest = json.loads(
            (protocol_root / "manifest.json").read_text(encoding="utf-8")
        )
        tasks: list[dict[str, Any]] = []
        index = 0
        for entry in manifest["items"]:
            run_dir = Path(entry["run_dir"]).resolve()
            for model, env_name in (("wilor", "v2s-wilor"), ("hamer", "v2s-hamer")):
                for view in ("left", "right"):
                    tasks.append(
                        {
                            "index": index,
                            "protocol": protocol,
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
            futures = [executor.submit(_run_task, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: (item["run"], item["model"], item["view"]))
        failed = [item for item in results if item["returncode"] != 0]
        summary_path = protocol_root / "model_inference_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "protocol": protocol,
                    "task_count": len(results),
                    "failed_count": len(failed),
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[{protocol}] summary -> {summary_path}", flush=True)
        if failed:
            for item in failed:
                print("FAILED", item)
        all_results.extend(results)
    total_failed = [item for item in all_results if item["returncode"] != 0]
    if total_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
