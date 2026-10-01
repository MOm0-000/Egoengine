#!/usr/bin/env python3
"""Reclassify the archived v2 epoch-4 numerical identity evidence only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SOURCE = (
    ROOT
    / "TRASH/algorithmic_training_v2_candidate_A_failclosed_likelihood_identity_final_20260927"
    / "update_audit/epoch_0004_identity_failure.npz"
)
OUTPUT = ROOT / "runs/taco_pour_likelihood_identity_numeric_reclassification_v3"
EXPECTED_SOURCE_SHA256 = (
    "1f3f4ea366c1cb8920a10fec95ed7746f6f653333cd7becefe8aa015c7549c93"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not SOURCE.is_file() or sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise ValueError("archived immutable v2 numerical evidence changed")

    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        LikelihoodIdentityGateSpec,
        numerical_likelihood_identity_report,
    )

    gate = LikelihoodIdentityGateSpec.float32_ulp_aware_v1()
    with np.load(SOURCE, allow_pickle=False) as archive:
        old_mu = torch.from_numpy(np.asarray(archive["old_mu"]))
        current_mu = torch.from_numpy(np.asarray(archive["failed_mu"]))
        old_sigma = torch.from_numpy(np.asarray(archive["old_sigma"]))
        current_sigma = torch.from_numpy(np.asarray(archive["failed_sigma"]))
        old_neglogp = torch.from_numpy(np.asarray(archive["old_neglogp"]))
        current_neglogp = torch.from_numpy(np.asarray(archive["failed_neglogp"]))
        ratio = torch.from_numpy(np.asarray(archive["failed_ratio"]))
    numerical = numerical_likelihood_identity_report(
        old_mu=old_mu,
        current_mu=current_mu,
        old_sigma=old_sigma,
        current_sigma=current_sigma,
        old_neglogp=old_neglogp,
        current_neglogp=current_neglogp,
        ratio=ratio,
        gate=gate,
    )
    if not numerical["policy_output_numerical_equivalence_passed"]:
        raise RuntimeError("archived v2 C/D evidence exceeds the frozen v3 gate")
    report = {
        "schema": "taco_pour_likelihood_identity_numeric_reclassification_v3",
        "status": "v2_epoch4_numerical_C_D_pass_under_v3_gate",
        "classification": "archived_evidence_reclassification_only",
        "source": {
            "path": str(SOURCE.resolve()),
            "sha256": sha256(SOURCE),
            "repository_retention": "TRASH_archived_not_formal_runtime_input",
        },
        "numerical_C_D": numerical,
        "full_v3_A_B_C_D_gate_replayed": False,
        "actor_parameter_hash_equality": "not_recorded_in_archived_npz",
        "actor_RMS_hash_and_version_equality": "not_recorded_in_archived_npz",
        "v2_training_status_changed": False,
        "v2_training_remains": (
            "halted_fail_closed_pre_optimizer_likelihood_identity_epoch4"
        ),
        "optimizer_updates_in_failed_epoch": 0,
        "warm_start_allowed": False,
        "purpose": (
            "Confirm that the archived float32 policy-output and likelihood "
            "differences fit the predeclared v3 numerical envelope. This does "
            "not convert v2 into valid training and does not reconstruct missing "
            "actor/RMS identity hashes."
        ),
    }
    OUTPUT.mkdir(parents=True)
    path = OUTPUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
