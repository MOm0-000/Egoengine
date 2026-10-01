"""Cavity repair may remove approximation overfill, never actual CAD material."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_taco_bowl_collision import main, subtract_empty_box
from audit_taco_collision_repair import union_contains


def fixture_parts(tmp_path):
    shell = trimesh.creation.box(extents=[.04, .04, .04])
    hole = trimesh.creation.box(extents=[.02, .02, .06])
    native = trimesh.boolean.difference([shell, hole], engine="manifold")
    source, part = tmp_path / "native.obj", tmp_path / "0.obj"
    native.export(source)
    shell.export(part)
    record = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  unit_scale=1., artifacts=[dict(path=str(part), sha256=hashlib.sha256(part.read_bytes()).hexdigest())])
    (tmp_path / "provenance.json").write_text(json.dumps(record))
    return source


def test_certified_empty_cavity_is_removed_but_native_surface_is_covered(tmp_path):
    source = fixture_parts(tmp_path)
    before = source.read_bytes()
    out = tmp_path / "result"
    subtract_empty_box(tmp_path, out, [-.00999, -.00999, -.03, .00999, .00999, .03])
    record = json.loads((out / "provenance.json").read_text())
    parts = [trimesh.load_mesh(row["path"], process=True) for row in record["artifacts"]]
    assert record["native_intersection_faces"] == 0
    assert all(part.is_volume and part.is_convex for part in parts)
    assert not union_contains(parts, [[0., 0., 0.]])[0]
    assert union_contains(parts, trimesh.load_mesh(source).vertices, tolerance=1e-7).all()
    assert source.read_bytes() == before


def test_carving_actual_material_fails_before_output(tmp_path):
    fixture_parts(tmp_path)
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="removes native material"):
        subtract_empty_box(tmp_path, out, [-.03, -.03, -.03, .03, .03, .03])
    assert not out.exists()


def test_invalid_coacd_part_leaves_no_partial_directory(tmp_path, monkeypatch):
    import coacd
    source = fixture_parts(tmp_path)
    valid = trimesh.creation.box()
    invalid = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    monkeypatch.setattr(coacd, "run_coacd", lambda *a, **kw:
                        [(valid.vertices, valid.faces), (invalid, np.array([[0, 1, 2]]))])
    out = tmp_path / "failed"
    monkeypatch.setattr(sys, "argv", ["prepare", "--source", str(source), "--output", str(out), "--unit-scale", "1"])
    with pytest.raises(ValueError, match="invalid convex part 1"):
        main()
    assert not out.exists()
