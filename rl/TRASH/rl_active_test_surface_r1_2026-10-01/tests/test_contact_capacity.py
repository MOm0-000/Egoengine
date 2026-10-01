"""Capacity is not just final contact count, nor an average constraint budget."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_taco_contact_capacity import allocation_usage, run


def counter(value):
    return SimpleNamespace(numpy=lambda: np.asarray(value))


@pytest.mark.parametrize("contacts,pairs,constraints,overflow", [
    (289, 832, [456, 412, 288, 0], True),  # Real old-capacity broadphase failure.
    (513, 500, [100, 100, 100, 100], True),
    (400, 500, [513, 0, 0, 0], True),  # One world's constraints cannot borrow.
    (512, 512, [512, 512, 512, 512], False),
    (460, 848, [460, 460, 460, 460], True),
])
def test_unclipped_overflow_counters(contacts, pairs, constraints, overflow):
    data = SimpleNamespace(nacon=counter([contacts]), ncollision=counter([pairs]),
                           nefc=counter(constraints), naconmax=512, njmax=512)
    report = allocation_usage(data)
    assert report["overflow"] is overflow
    assert report["contacts_total"] == contacts and report["broadphase_pairs_total"] == pairs


def test_no_overwrite_or_nonpositive_allocations(tmp_path):
    with pytest.raises(FileExistsError):
        run(SimpleNamespace(output=tmp_path))
    with pytest.raises(ValueError, match="positive"):
        run(SimpleNamespace(output=tmp_path / "unused.json", worlds=0, nconmax=128, njmax=512, steps=0))
