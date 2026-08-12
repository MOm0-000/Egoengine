import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
import zarr

from video_to_spider.adapters.depth_gate import (
    evaluate_depth_consistency,
    evaluate_stereo_depth,
    require_depth_gate,
    write_depth_gate,
)
from video_to_spider.adapters.refit_mesh_scale import run as refit_mesh_scale


def _make_depth_run(tmp_path: Path, ratio: float = 1.4) -> Path:
    run = tmp_path / "run"
    (run / "depth").mkdir(parents=True)
    (run / "depth_crosscheck").mkdir()
    (run / "segmentation").mkdir()
    frame_indices = np.arange(4, dtype=np.int64)
    shape = (4, 20, 24)
    primary_depth = np.ones(shape, dtype=np.float32)
    secondary_depth = primary_depth * ratio
    for path, values in (
        (run / "depth/metric_depth.zarr", primary_depth),
        (run / "depth_crosscheck/unidepth_depth.zarr", secondary_depth),
    ):
        group = zarr.open_group(str(path), mode="w")
        group.create_dataset("depth_m", data=values)
        group.create_dataset("valid", data=np.ones(shape, dtype=bool))
        group.create_dataset("frame_indices", data=frame_indices)
    (run / "depth/metadata.json").write_text(
        json.dumps({"model": "Depth Anything 3 Metric Large"}) + "\n"
    )
    (run / "depth_crosscheck/metadata.json").write_text(
        json.dumps({"model": "UniDepthV2"}) + "\n"
    )
    masks = np.zeros(shape, dtype=bool)
    masks[:, 4:16, 6:18] = True
    np.savez_compressed(
        run / "segmentation/object_masks.npz",
        frame_indices=frame_indices,
        masks=masks,
        valid=np.ones(4, dtype=bool),
    )
    np.savez_compressed(
        run / "segmentation/hand_masks.npz",
        frame_indices=frame_indices,
        masks=np.zeros(shape, dtype=bool),
    )
    (run / "calibration").mkdir()
    np.save(
        run / "calibration/intrinsics.npy",
        np.array([[20.0, 0.0, 12.0], [0.0, 20.0, 10.0], [0.0, 0.0, 1.0]]),
    )
    return run


def test_depth_gate_accepts_stable_cross_model_offset_without_rescaling(tmp_path: Path):
    run = _make_depth_run(tmp_path, ratio=1.4)
    payload = evaluate_depth_consistency(run)
    assert payload["accepted"]
    assert payload["ground_truth_used"] is False
    assert payload["secondary_to_primary_object_depth_ratio"]["median"] == pytest.approx(1.4)
    assert "no rescaling or blending" in payload["policy"]


def test_da3_requires_fresh_accepted_gate(tmp_path: Path):
    run = _make_depth_run(tmp_path)
    with pytest.raises(RuntimeError, match="depth_gate_missing"):
        require_depth_gate(run)
    gate = write_depth_gate(run)
    assert require_depth_gate(run)["accepted"]
    (run / "depth_crosscheck/metadata.json").write_text(
        json.dumps({"model": "UniDepthV2", "changed": True}) + "\n"
    )
    with pytest.raises(RuntimeError, match="depth_gate_stale"):
        require_depth_gate(run)
    assert gate.exists()


def test_da3_gate_hashes_both_depth_artifacts(tmp_path: Path):
    run = _make_depth_run(tmp_path)
    write_depth_gate(run)
    assert require_depth_gate(run)["accepted"]

    secondary = zarr.open_group(
        str(run / "depth_crosscheck/unidepth_depth.zarr"), mode="a"
    )
    values = np.asarray(secondary["depth_m"])
    values[0, 0, 0] += 0.25
    secondary["depth_m"][:] = values

    with pytest.raises(RuntimeError, match="secondary depth artifact changed after gating"):
        require_depth_gate(run)


def test_depth_gate_rejects_gross_scale_conflict(tmp_path: Path):
    run = _make_depth_run(tmp_path, ratio=3.0)
    gate = write_depth_gate(run)
    payload = json.loads(gate.read_text())
    assert not payload["accepted"]
    assert not payload["checks"]["median_scale_ratio_in_range"]
    with pytest.raises(RuntimeError, match="depth_gate_rejected"):
        require_depth_gate(run)


def _make_stereo_depth_run(tmp_path: Path, valid_ratio: float = 1.0) -> Path:
    run = tmp_path / "stereo_run"
    for relative in ("depth", "calibration", "frames", "input"):
        (run / relative).mkdir(parents=True, exist_ok=True)
    shape = (3, 10, 12)
    K = np.array([[100.0, 0, 6.0], [0, 100.0, 5.0], [0, 0, 1.0]])
    np.save(run / "calibration/intrinsics.npy", K)
    np.save(run / "calibration/intrinsics_right.npy", K)
    np.save(run / "calibration/stereo_common_valid.npy", np.ones(shape[1:], dtype=bool))
    stereo = {
        "accepted": True,
        "K_rect_left": K.tolist(),
        "K_rect_right": K.tolist(),
        "baseline_m": 0.064,
    }
    (run / "calibration/stereo.json").write_text(json.dumps(stereo) + "\n")
    rows = [
        {"frame_index": index, "source_frame_index": index, "rgb_path": f"frames/{index}.png"}
        for index in range(shape[0])
    ]
    (run / "frames/frame_index.json").write_text(json.dumps({"frames": rows}) + "\n")
    (run / "input/source.json").write_text(
        json.dumps({"video": {"height": shape[1], "width": shape[2]}}) + "\n"
    )
    depth = np.ones(shape, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    valid.reshape(-1)[: int(valid.size * valid_ratio)] = True
    depth[~valid] = np.nan
    group = zarr.open_group(str(run / "depth/metric_depth.zarr"), mode="w")
    group.create_dataset("depth_m", data=depth)
    group.create_dataset("valid", data=valid)
    group.create_dataset("frame_indices", data=np.arange(shape[0]))
    (run / "depth/metadata.json").write_text(
        json.dumps(
            {
                "model": "FoundationStereo",
                "ground_truth_used": False,
                "scale_or_shift_alignment_applied": False,
                "K_rect": K.tolist(),
                "baseline_m": 0.064,
                "common_valid_mask": "calibration/stereo_common_valid.npy",
                "common_valid_pixel_ratio": 1.0,
            }
        )
        + "\n"
    )
    return run


def _add_stereo_object_masks(run: Path) -> None:
    (run / "segmentation").mkdir(exist_ok=True)
    masks = np.zeros((3, 10, 12), dtype=bool)
    masks[:, 2:8, :4] = True
    np.savez_compressed(
        run / "segmentation/object_masks.npz",
        frame_indices=np.arange(3), masks=masks,
        valid=np.ones(3, dtype=bool),
    )


def test_stereo_depth_gate_accepts_raw_calibrated_metric(tmp_path: Path):
    run = _make_stereo_depth_run(tmp_path)
    _add_stereo_object_masks(run)
    payload = evaluate_stereo_depth(run)
    assert payload["accepted"]
    assert payload["gate_kind"] == "calibrated_stereo_native_metric"
    gate = write_depth_gate(run)
    assert gate.is_file()
    assert require_depth_gate(run)["accepted"]


def test_stereo_depth_gate_rejects_sparse_or_changed_calibration(tmp_path: Path):
    run = _make_stereo_depth_run(tmp_path, valid_ratio=0.5)
    _add_stereo_object_masks(run)
    gate = write_depth_gate(run)
    payload = json.loads(gate.read_text())
    assert not payload["accepted"]
    assert not payload["checks"]["valid_pixel_ratio"]
    run = _make_stereo_depth_run(tmp_path / "fresh")
    _add_stereo_object_masks(run)
    write_depth_gate(run)
    stereo = json.loads((run / "calibration/stereo.json").read_text())
    stereo["baseline_m"] = 0.08
    (run / "calibration/stereo.json").write_text(json.dumps(stereo) + "\n")
    with pytest.raises(RuntimeError, match="depth_gate_stale"):
        require_depth_gate(run)


def test_stereo_depth_gate_rejects_object_region_dropout(tmp_path: Path):
    run = _make_stereo_depth_run(tmp_path)
    _add_stereo_object_masks(run)
    group = zarr.open_group(str(run / "depth/metric_depth.zarr"), mode="a")
    valid = np.asarray(group["valid"])
    depth = np.asarray(group["depth_m"])
    valid[:, 2:8, :4] = False
    depth[:, 2:8, :4] = np.nan
    group["valid"][:] = valid
    group["depth_m"][:] = depth

    payload = evaluate_stereo_depth(run)

    assert payload["primary"]["valid_pixel_ratio"] > 0.75
    assert not payload["accepted"]
    assert not payload["checks"]["object_valid_coverage"]
    assert payload["object_depth_coverage"]["usable_object_frame_ratio"] == 0.0


def test_stereo_depth_gate_rejects_distinct_left_right_rectified_k(tmp_path: Path):
    run = _make_stereo_depth_run(tmp_path)
    _add_stereo_object_masks(run)
    K_right = np.load(run / "calibration/intrinsics_right.npy")
    K_right[0, 2] += 1.0
    np.save(run / "calibration/intrinsics_right.npy", K_right)
    stereo = json.loads((run / "calibration/stereo.json").read_text())
    stereo["K_rect_right"] = K_right.tolist()
    (run / "calibration/stereo.json").write_text(json.dumps(stereo) + "\n")

    payload = evaluate_stereo_depth(run)

    assert not payload["accepted"]
    assert not payload["checks"]["shared_left_right_rectified_intrinsics"]


def test_stereo_depth_gate_hashes_object_masks(tmp_path: Path):
    run = _make_stereo_depth_run(tmp_path)
    _add_stereo_object_masks(run)
    write_depth_gate(run)
    assert require_depth_gate(run)["accepted"]

    masks_path = run / "segmentation/object_masks.npz"
    with np.load(masks_path, allow_pickle=False) as artifact:
        payload = {key: np.asarray(artifact[key]) for key in artifact.files}
    payload["masks"][0, 0, 0] = True
    np.savez_compressed(masks_path, **payload)

    with pytest.raises(RuntimeError, match="object masks changed after gating"):
        require_depth_gate(run)


def test_stereo_depth_gate_hashes_depth_tree(tmp_path: Path):
    run = _make_stereo_depth_run(tmp_path)
    _add_stereo_object_masks(run)
    write_depth_gate(run)
    assert require_depth_gate(run)["accepted"]
    group = zarr.open_group(str(run / "depth/metric_depth.zarr"), mode="a")
    depth = np.asarray(group["depth_m"])
    depth[0, 0, 0] = 1.25
    group["depth_m"][:] = depth

    with pytest.raises(RuntimeError, match="primary depth artifact changed after gating"):
        require_depth_gate(run)


def test_stereo_depth_gate_requires_automatic_object_masks(tmp_path: Path):
    run = _make_stereo_depth_run(tmp_path)
    with pytest.raises(FileNotFoundError, match="requires automatic object masks"):
        evaluate_stereo_depth(run)


def test_stereo_depth_gate_coverage_denominator_is_common_valid_domain(tmp_path: Path):
    run = _make_stereo_depth_run(tmp_path)
    _add_stereo_object_masks(run)
    common = np.ones((10, 12), dtype=bool)
    common[:, 8:] = False
    np.save(run / "calibration/stereo_common_valid.npy", common)
    metadata_path = run / "depth/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["common_valid_pixel_ratio"] = float(common.mean())
    metadata_path.write_text(json.dumps(metadata) + "\n")
    group = zarr.open_group(str(run / "depth/metric_depth.zarr"), mode="a")
    valid = np.asarray(group["valid"])
    depth = np.asarray(group["depth_m"])
    valid[:, :, 8:] = False
    depth[:, :, 8:] = np.nan
    group["valid"][:] = valid
    group["depth_m"][:] = depth

    payload = evaluate_stereo_depth(run)

    assert payload["accepted"]
    assert payload["primary"]["valid_pixel_ratio"] == pytest.approx(1.0)


def test_stereo_depth_gate_rejects_object_outside_common_domain(tmp_path: Path):
    run = _make_stereo_depth_run(tmp_path)
    _add_stereo_object_masks(run)
    common = np.ones((10, 12), dtype=bool)
    common[:, :4] = False
    np.save(run / "calibration/stereo_common_valid.npy", common)
    metadata_path = run / "depth/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["common_valid_pixel_ratio"] = float(common.mean())
    metadata_path.write_text(json.dumps(metadata) + "\n")
    group = zarr.open_group(str(run / "depth/metric_depth.zarr"), mode="a")
    valid = np.asarray(group["valid"])
    depth = np.asarray(group["depth_m"])
    valid[:, :, :4] = False
    depth[:, :, :4] = np.nan
    group["valid"][:] = valid
    group["depth_m"][:] = depth

    payload = evaluate_stereo_depth(run)

    assert not payload["accepted"]
    assert not payload["checks"]["object_within_stereo_common_domain"]
    assert payload["object_depth_coverage"]["object_pixel_ratio_within_common_domain"] == 0.0


def test_metric_refit_preserves_keyframes_for_foundationpose(tmp_path: Path):
    run = _make_depth_run(tmp_path)
    write_depth_gate(run)
    proposal_dir = run / "mesh_proposals/p0"
    proposal_dir.mkdir(parents=True)
    trimesh.creation.box(extents=[1.0, 1.0, 0.2]).export(proposal_dir / "visual.obj")
    keyframes = [{"frame_index": 0, "mask_index": 0, "score": 1.0}]
    source = run / "mesh_proposals/mesh_ranking.json"
    source.write_text(json.dumps({
        "keyframes": keyframes,
        "proposals": [{
            "proposal_id": "p0",
            "frame_index": 0,
            "mask_index": 0,
            "visual_mesh": "p0/visual.obj",
            "model_layout": {
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 1.0],
                "scale": [0.5, 0.5, 0.5],
            },
        }],
    }) + "\n")
    destination = refit_mesh_scale(run)
    payload = json.loads(destination.read_text())
    assert payload["keyframes"] == keyframes
    assert payload["qualified_count"] == 1
