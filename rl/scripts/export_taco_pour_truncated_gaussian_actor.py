#!/usr/bin/env python3
"""Export the accepted experiment actor without the large optimizer checkpoint."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/taco_pour_state_feasible_truncated_gaussian_experiment_v1"


def model_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    report = json.loads((RUN / "report.json").read_text())
    training = report["diagnostic"]["training_runs"][0]
    checkpoints = training["checkpoint_artifacts"]
    if len(checkpoints) != 1:
        raise ValueError("the authorized experiment must have exactly one checkpoint")
    source = Path(checkpoints[0]["path"])
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != checkpoints[0]["sha256"]:
        raise ValueError("source optimizer checkpoint changed")
    checkpoint = torch.load(
        io.BytesIO(source_bytes), map_location="cpu", weights_only=False
    )
    actor_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in checkpoint["model"].items()
    }
    actor_sha = model_sha256(actor_state)
    if actor_sha != training["actor_state_sha256"]:
        raise ValueError("exported actor does not match the CPU-validated actor hash")
    payload = {
        "schema": "taco_pour_truncated_gaussian_actor_v1",
        "actor_state_sha256": actor_sha,
        "source_checkpoint_sha256": checkpoints[0]["sha256"],
        "model": actor_state,
    }
    raw = io.BytesIO()
    torch.save(payload, raw)
    encoded = gzip.compress(raw.getvalue(), compresslevel=9, mtime=0)
    artifact = RUN / "cpu_validated_actor.pt.gz"
    artifact.write_bytes(encoded)
    manifest = {
        "schema": "taco_pour_truncated_gaussian_actor_artifact_v1",
        "artifact": {
            "path": str(artifact.resolve().relative_to(ROOT)),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "uncompressed_pt_sha256": hashlib.sha256(raw.getvalue()).hexdigest(),
            "bytes": len(encoded),
        },
        "actor_state_sha256": actor_sha,
        "source_checkpoint": {
            "path": str(source.resolve()),
            "sha256": checkpoints[0]["sha256"],
            "path_semantics": "local_full_checkpoint_not_in_compact_versioned_evidence",
            "portable_artifact_contains_optimizer_or_critic": False,
        },
    }
    (RUN / "actor_artifact.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["artifact"], indent=2))


if __name__ == "__main__":
    main()
