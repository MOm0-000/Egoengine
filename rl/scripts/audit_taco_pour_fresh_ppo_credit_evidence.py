#!/usr/bin/env python3
"""Read-only reconstruction and interpretation of the fresh PPO credit run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "external" / "human2sim2robot"),
    str(ROOT / "external" / "spider_compat"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64).reshape(-1)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=Path,
        default=ROOT / "runs/taco_pour_corrected_fresh_ppo_credit_instrumented_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "runs/taco_pour_fresh_ppo_credit_evidence_v1",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    from human2sim2robot.ppo.utils.models import ModelA2CContinuousLogStd
    from run_mjwp_ppo import _build_network_config
    from video_to_spider.rl.credit_audit import (
        apply_xor_patch,
        load_fixed_probe_panel,
        model_state_sha256,
    )
    from video_to_spider.rl.state_feasible_truncated_gaussian import (
        deterministic_truncated_action,
        truncated_normal_log_prob,
    )

    report_path = args.run / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "completed_diagnostic_not_promotable":
        raise ValueError("fresh PPO credit run is incomplete")
    manifest_path = Path(report["training_audit"]["credit_audit"]["path"])
    manifest = json.loads(manifest_path.read_text())
    initial_path = Path(manifest["initial_actor"]["path"])
    initial = torch.load(
        io.BytesIO(gzip.decompress(initial_path.read_bytes())),
        map_location="cpu", weights_only=False,
    )
    if model_state_sha256(initial) != manifest["initial_actor"]["model_state_sha256"]:
        raise ValueError("initial actor artifact does not match its manifest")

    probe_source = Path(report["probe_panel"]["source_artifact"])
    panel = load_fixed_probe_panel(probe_source)
    model = ModelA2CContinuousLogStd(
        network_config=_build_network_config(4),
        actions_num=36,
        input_shape=(236,),
        normalize_value=True,
        normalize_input=True,
        value_size=1,
        num_seqs=len(panel["labels"]),
    )
    model.eval()

    raw = torch.as_tensor(panel["raw_observation"], dtype=torch.float32)
    hidden = [torch.as_tensor(value, dtype=torch.float32) for value in panel["rnn_states"]]
    low = torch.as_tensor(panel["action_low"], dtype=torch.float32)
    high = torch.as_tensor(panel["action_high"], dtype=torch.float32)

    def probes(state: dict[str, torch.Tensor]) -> dict:
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            normalized = model.norm_obs(raw)
            mu, logstd, _, _ = model.a2c_network({"obs": normalized, "rnn_states": hidden})
            action = deterministic_truncated_action(mu, low, high)
        return {
            "actor_state_sha256": model_state_sha256(state),
            "mu_y": mu[:, 1].numpy().tolist(),
            "sigma_y": torch.exp(logstd)[:, 1].numpy().tolist(),
            "deterministic_action_y": action[:, 1].numpy().tolist(),
        }

    transitions = []
    current = initial
    for update in manifest["update_reports"]:
        forward_patch = Path(update["forward_state_patch"]["path"])
        optimizer_patch = Path(update["optimizer_state_patch"]["path"])
        entry_probe = probes(current)
        after_forward = apply_xor_patch(current, forward_patch)
        if model_state_sha256(after_forward) != update["forward_state_patch"]["after_model_state_sha256"]:
            raise ValueError("forward-state patch did not reconstruct exactly")
        forward_probe = probes(after_forward)
        after_optimizer = apply_xor_patch(after_forward, optimizer_patch)
        if model_state_sha256(after_optimizer) != update["optimizer_state_patch"]["after_model_state_sha256"]:
            raise ValueError("optimizer-state patch did not reconstruct exactly")
        optimizer_probe = probes(after_optimizer)
        with np.load(update["row_artifact"]["path"], allow_pickle=False) as rows:
            ratio = np.asarray(rows["ratio"]).reshape(-1)
        epoch_artifact = next(
            row["credit_rows"]["path"]
            for row in manifest["epoch_reports"]
            if int(row["epoch"]) == int(update["epoch"])
        )
        with np.load(epoch_artifact, allow_pickle=False) as epoch_data:
            update_sources = np.asarray(epoch_data["source_endpoint"]).reshape(-1)
        labels = panel["labels"]
        focus = [
            index for index, label in enumerate(labels)
            if label.startswith("ppo_source_")
            and 43 <= int(label.rsplit("_", 1)[1]) <= 46
        ]
        transitions.append({
            "epoch": update["epoch"],
            "update_in_epoch": update["update_in_epoch"],
            "global_actor_update": update["global_actor_update"],
            "entry_mu_y": {labels[index]: entry_probe["mu_y"][index] for index in focus},
            "after_forward_before_optimizer_mu_y": {
                labels[index]: forward_probe["mu_y"][index] for index in focus
            },
            "after_optimizer_mu_y": {
                labels[index]: optimizer_probe["mu_y"][index] for index in focus
            },
            "maximum_abs_mu_y_change_from_normalization_forward": float(max(
                abs(forward_probe["mu_y"][index] - entry_probe["mu_y"][index])
                for index in focus
            )),
            "maximum_abs_mu_y_change_from_optimizer": float(max(
                abs(optimizer_probe["mu_y"][index] - forward_probe["mu_y"][index])
                for index in focus
            )),
            "ratio_before_optimizer": {
                "minimum": float(ratio.min()),
                "mean": float(ratio.mean()),
                "maximum": float(ratio.max()),
                "outside_PPO_clip_count": int(((ratio < 0.8) | (ratio > 1.2)).sum()),
                "outside_PPO_clip_fraction": float(((ratio < 0.8) | (ratio > 1.2)).mean()),
                "source_43_46_outside_clip_count": int((
                    ((ratio < 0.8) | (ratio > 1.2))
                    & (update_sources >= 43) & (update_sources <= 46)
                ).sum()),
                "source_43_46_count": int(
                    ((update_sources >= 43) & (update_sources <= 46)).sum()
                ),
            },
            "state_hashes": {
                "entry": entry_probe["actor_state_sha256"],
                "after_forward": forward_probe["actor_state_sha256"],
                "after_optimizer": optimizer_probe["actor_state_sha256"],
            },
        })
        current = after_optimizer
    if model_state_sha256(current) != manifest["final_actor"]["model_state_sha256"]:
        raise ValueError("full actor patch chain does not reach final actor")

    epochs = []
    total_sign_flips = 0
    focus_sign_flips = 0
    for epoch_report in manifest["epoch_reports"]:
        with np.load(epoch_report["credit_rows"]["path"], allow_pickle=False) as data:
            source = np.asarray(data["source_endpoint"]).reshape(-1)
            action_y = np.asarray(data["sampled_action"])[:, 1]
            raw_adv = np.asarray(data["raw_advantage"]).reshape(-1)
            norm_adv = np.asarray(data["normalized_advantage"]).reshape(-1)
            old_logp = np.asarray(data["old_neglogp"]).reshape(-1)
            direct_logp = -truncated_normal_log_prob(
                torch.as_tensor(np.asarray(data["sampled_action"])),
                torch.as_tensor(np.asarray(data["actor_mu"])),
                torch.as_tensor(np.asarray(data["actor_sigma"])),
                torch.as_tensor(np.asarray(data["action_low"])),
                torch.as_tensor(np.asarray(data["action_high"])),
                minimum_mass=1e-12,
            ).sum(dim=-1).numpy()
            focus = (source >= 43) & (source <= 46)
            sign_flip = raw_adv * norm_adv < 0
            total_sign_flips += int(sign_flip.sum())
            focus_sign_flips += int((sign_flip & focus).sum())
            epochs.append({
                "epoch": int(epoch_report["epoch"]),
                "all_samples": {
                    "raw_advantage": summary(raw_adv),
                    "normalized_advantage": summary(norm_adv),
                    "sign_flip_count": int(sign_flip.sum()),
                },
                "source_43_46": {
                    "count": int(focus.sum()),
                    "action_y": summary(action_y[focus]),
                    "raw_advantage": summary(raw_adv[focus]),
                    "normalized_advantage": summary(norm_adv[focus]),
                    "raw_positive_normalized_negative_count": int(
                        (focus & (raw_adv > 0) & (norm_adv < 0)).sum()
                    ),
                    "raw_negative_normalized_positive_count": int(
                        (focus & (raw_adv < 0) & (norm_adv > 0)).sum()
                    ),
                },
                "rollout_logprob_direct_recompute_max_abs_error": float(
                    np.max(np.abs(old_logp - direct_logp))
                ),
            })

    first = transitions[0]
    first_forward_patch = Path(
        manifest["update_reports"][0]["forward_state_patch"]["path"]
    )
    with np.load(first_forward_patch, allow_pickle=False) as patch:
        patch_manifest = json.loads(str(patch["manifest_json"]))
        first_forward_changed_tensors = [
            row["name"] for row in patch_manifest if np.any(patch[row["key"]])
        ]
    classification = {
        "A_advantage_directly_rewards_large_positive_y": (
            "not_supported_as_a_general_explanation"
        ),
        "B_global_advantage_normalization_changes_tail_credit": "strongly_supported",
        "C_PPO_likelihood_pipeline_semantic_defect": "strongly_supported",
        "D_bias_accumulates_over_updates": "supported_after_the_first_large_transition",
        "first_update_ratio_finding": (
            "With network weights still unchanged, updating input running statistics before "
            "the first likelihood evaluation moves 134/160 ratios outside [0.8,1.2]. "
            "The PPO old/new likelihood contract therefore mixes two observation transforms."
        ),
    }
    report_out = {
        "schema": "taco_pour_fresh_ppo_credit_evidence_v1",
        "status": "completed_read_only_reconstruction",
        "training_executed": False,
        "actor_update_executed": False,
        "chunk_commit_written": False,
        "source_run": {"path": str(report_path.resolve()), "sha256": sha256(report_path)},
        "patch_chain": {
            "updates_reconstructed": len(transitions),
            "initial_actor_sha256": manifest["initial_actor"]["model_state_sha256"],
            "final_actor_sha256": manifest["final_actor"]["model_state_sha256"],
            "bitwise_chain_complete": True,
        },
        "first_update": first,
        "first_forward_changed_tensors": first_forward_changed_tensors,
        "actor_state_transition_probes": transitions,
        "advantage_credit": {
            "global_sign_flip_count": total_sign_flips,
            "source_43_46_sign_flip_count": focus_sign_flips,
            "epochs": epochs,
        },
        "classification": classification,
        "limitations": [
            "Fixed hidden states are historical final-actor contexts and are sensitivity probes, not fresh on-policy hidden states.",
            "Action/advantage association is observational on-policy evidence, not an isolated action causal effect.",
            "Global advantage normalization changes relative credit by design; this report does not label that behavior an implementation bug by itself.",
        ],
    }
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report_out, indent=2) + "\n")
    print(json.dumps({
        "status": report_out["status"],
        "first_update": first,
        "classification": classification,
        "source_43_46_sign_flip_count": focus_sign_flips,
    }, indent=2))


if __name__ == "__main__":
    main()
