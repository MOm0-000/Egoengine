"""GT-isolated EgoDex scanning, validation, and RGB/calibration ingest."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import h5py
import numpy as np

from ..manifest import RunManifest, stage_cache_key
from ..schemas import SCHEMA_VERSION, validate_transforms

DEFAULT_ROOT = Path("/data_all/share/datasets/egodex")
_COLOR_WORDS = {
    "black", "blue", "brown", "gray", "green", "hotpink", "lavender", "pink", "red", "white", "wooden",
}
_STOP_WORDS = {
    "a", "an", "and", "any", "against", "at", "back", "by", "from", "in", "into", "of", "on", "onto",
    "or", "out", "placed", "position", "round", "sitting", "table", "tablecloth", "the", "then", "to", "using",
    "while", "with", "other", "top",
}
_NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
_ACTION_WORDS = {
    "add", "arrange", "assemble", "boil", "build", "catch", "charge", "clean", "clip", "color", "crack",
    "crumple", "deal", "declutter", "disassemble", "disconnect", "dry", "empty", "flatten", "flip", "fry",
    "gather", "pick", "place", "plug", "put", "remove", "return", "serve", "split", "stake", "take", "topple",
    "unbraid", "uncharge", "unclip", "unplug", "unstake", "use",
}
_DIRECTION_WORDS = {
    "down", "downward", "downwards", "horizontal", "horizontally", "left", "right", "up", "upright",
    "upward", "upwards", "vertical", "vertically",
}
_SINGULAR = {"lids": "lid", "cups": "cup", "dominoes": "domino", "legos": "lego", "tiles": "tile",
             "cards": "card", "papers": "paper", "paperclips": "paperclip", "pages": "page", "plates": "plate",
             "forks": "fork", "spoons": "spoon", "knives": "knife", "rings": "ring", "hands": "hand"}
_PHRASE_EXPANSIONS = {
    # EgoDex descriptions often use the generic annotation phrase "plush object".
    # SAM-style open-vocabulary detectors respond more reliably to common visual
    # category names, so add deterministic text-only aliases without consulting GT.
    "plush object": ("plush toy", "white plush toy", "stuffed toy", "stuffed animal"),
}


@dataclass(frozen=True)
class EpisodeRef:
    task: str
    episode_id: str
    mp4_path: Path
    hdf5_path: Path


def scan_episodes(root: str | Path = DEFAULT_ROOT) -> Iterator[EpisodeRef]:
    root_path = Path(root)
    for hdf5_path in sorted(root_path.glob("part*/*/*.hdf5")):
        mp4_path = hdf5_path.with_suffix(".mp4")
        if mp4_path.is_file():
            yield EpisodeRef(hdf5_path.parent.name, hdf5_path.stem, mp4_path, hdf5_path)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_decode_attr(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def select_instruction(attrs: dict[str, Any], task_name: str) -> tuple[str, str]:
    selected = str(attrs.get("which_llm_description", "")).strip()
    preferred = "llm_description2" if selected == "2" else "llm_description"
    for key in (preferred, "llm_description", "llm_description2"):
        value = attrs.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "none":
            return str(value).strip(), key
    return task_name.replace("_", " "), "task_directory"


def keyword_candidates(instruction: str, task_name: str) -> list[str]:
    """Produce deterministic text-only candidates without reading GT object annotations."""
    words = re.findall(r"[A-Za-z]+", instruction.lower())
    candidates: list[str] = []

    def append(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    # Preserve adjacent descriptive noun phrases (for example, "red cup" or
    # "plush object") before falling back to individual category words.  Color
    # words are useful as part of a phrase but remain too ambiguous on their own.
    phrase: list[str] = []
    for word in words:
        normalized = _SINGULAR.get(word, word)
        is_boundary = (
            normalized in _STOP_WORDS
            or normalized in _ACTION_WORDS
            or normalized in _DIRECTION_WORDS
            or normalized in _NUMBER_WORDS
            or len(normalized) < 3
        )
        if is_boundary:
            if len(phrase) >= 2:
                joined = " ".join(phrase)
                append(joined)
                for expansion in _PHRASE_EXPANSIONS.get(joined, ()):
                    append(expansion)
            phrase = []
        else:
            phrase.append(normalized)
    if len(phrase) >= 2:
        joined = " ".join(phrase)
        append(joined)
        for expansion in _PHRASE_EXPANSIONS.get(joined, ()):
            append(expansion)

    for word in words:
        normalized = _SINGULAR.get(word, word)
        if (
            normalized in _STOP_WORDS
            or normalized in _COLOR_WORDS
            or normalized in _ACTION_WORDS
            or normalized in _DIRECTION_WORDS
            or normalized in _NUMBER_WORDS
            or len(normalized) < 3
        ):
            continue
        append(normalized)
    for word in task_name.lower().split("_"):
        normalized = _SINGULAR.get(word, word)
        if normalized not in _ACTION_WORDS and normalized not in _DIRECTION_WORDS:
            append(normalized)
    return candidates[:12]


def inspect_episode(episode: EpisodeRef) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(episode.mp4_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {episode.mp4_path}")
    video = {
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    with h5py.File(episode.hdf5_path, "r") as handle:
        if "camera/intrinsic" not in handle or "transforms/camera" not in handle:
            raise ValueError("required camera/intrinsic or transforms/camera missing")
        K = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        T_world_camera = np.asarray(handle["transforms/camera"], dtype=np.float64)
        attrs = {key: _decode_attr(value) for key, value in handle.attrs.items()}
    if K.shape != (3, 3) or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"invalid camera intrinsic shape/value: {K}")
    validate_transforms("T_world_camera", T_world_camera)
    if video["frame_count"] != T_world_camera.shape[0]:
        raise ValueError(f"frame mismatch video={video['frame_count']} hdf5={T_world_camera.shape[0]}")
    if video["fps"] <= 0:
        raise ValueError("video FPS must be positive")
    instruction, instruction_source = select_instruction(attrs, episode.task)
    return {
        "video": video, "K": K, "T_world_camera": T_world_camera,
        "instruction_text": instruction, "instruction_source": instruction_source,
        "object_keyword_candidates": keyword_candidates(instruction, episode.task),
        "public_attributes": {key: attrs[key] for key in ("task", "environment", "session_name") if key in attrs},
    }


def ingest_episode(
    episode: EpisodeRef, output_dir: str | Path, *, start_frame: int = 0,
    end_frame: int | None = None, overwrite: bool = False,
) -> Path:
    info = inspect_episode(episode)
    frame_count = info["video"]["frame_count"]
    end = frame_count if end_frame is None else end_frame
    if start_frame < 0 or end <= start_frame or end > frame_count:
        raise ValueError(f"invalid frame interval [{start_frame}, {end}) for {frame_count} frames")
    run_dir = Path(output_dir)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"run already exists: {run_dir}; pass overwrite=True")
    for relative in ("input", "frames/rgb", "calibration"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    config = {"start_frame": start_frame, "end_frame": end}
    manifest = RunManifest.create(
        manifest_path, run_id=run_dir.name, source_episode=str(episode.hdf5_path.with_suffix("")),
        config=config, frame_count=end - start_frame, fps=info["video"]["fps"],
    )
    cache_key = stage_cache_key("ingest", config, [episode.mp4_path, episode.hdf5_path])
    manifest.start_stage("ingest", cache_key=cache_key, command=["ingest", str(episode.hdf5_path)], environment="v2s-core")
    source = {
        "schema_version": SCHEMA_VERSION, "task_directory": episode.task, "episode_id": episode.episode_id,
        "mp4_path": str(episode.mp4_path), "hdf5_path": str(episode.hdf5_path),
        "instruction_text": info["instruction_text"], "instruction_source": info["instruction_source"],
        "object_keyword_candidates": info["object_keyword_candidates"], "video": info["video"],
        "selected_frame_interval": [start_frame, end], "public_attributes": info["public_attributes"],
    }
    (run_dir / "input/source.json").write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.save(run_dir / "calibration/intrinsics.npy", info["K"].astype(np.float32))
    np.save(run_dir / "calibration/T_world_camera.npy", info["T_world_camera"][start_frame:end].astype(np.float32))
    capture = cv2.VideoCapture(str(episode.mp4_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = []
    for source_index in range(start_frame, end):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"decode failed at source frame {source_index}")
        relative_path = f"frames/rgb/{source_index:06d}.jpg"
        if not cv2.imwrite(str(run_dir / relative_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            capture.release()
            raise RuntimeError(f"failed to write {relative_path}")
        frame_index.append({
            "frame_index": source_index - start_frame, "source_frame_index": source_index,
            "timestamp_s": (source_index - start_frame) / info["video"]["fps"],
            "source_timestamp_s": source_index / info["video"]["fps"], "rgb_path": relative_path,
        })
    capture.release()
    index_payload = {"schema_version": SCHEMA_VERSION, "fps": info["video"]["fps"], "frames": frame_index}
    (run_dir / "frames/frame_index.json").write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    outputs = ["input/source.json", "frames/frame_index.json", "calibration/intrinsics.npy", "calibration/T_world_camera.npy"]
    manifest.finish_stage("ingest", success=True, outputs=outputs, quality_metrics={
        "decoded_frames": len(frame_index), "hdf5_frames": frame_count,
        "rotation_orthogonality_max": float(np.max(np.abs(
            np.swapaxes(info["T_world_camera"][:, :3, :3], 1, 2) @ info["T_world_camera"][:, :3, :3] - np.eye(3)
        ))),
    })
    return manifest_path


def find_episode(root: str | Path, task: str, episode_id: str) -> EpisodeRef:
    matches = [episode for episode in scan_episodes(root) if episode.task == task and episode.episode_id == str(episode_id)]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one episode for {task}/{episode_id}, found {len(matches)}")
    return matches[0]
