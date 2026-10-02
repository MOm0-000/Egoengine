"""Bounded virtual-object-assistance experiment for the active RL core."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import mujoco
import numpy as np
import torch
import yaml

from video_to_spider.rl.object_assistance import ToolAssistSpec, assistance_scale

from .audit import append_jsonl, append_rollout_npz, distribution, module_sha256
from .env import IndependentWorlds
from .policy import (
    ASSISTED_CRITIC_INPUT_DIM,
    ASSISTED_CRITIC_INPUT_SPEC,
    PolicyBundle,
    burn_in_prefix,
    policy_step,
)
from .ppo import PPOConfig, PPOTrainer
from .rollout import FixedBoundaryCollector
from .state_io import (
    build_training_checkpoint,
    capture_rng_states,
    load_boundary_context,
    load_torch_gzip,
    manifest_entry,
    restore_training_checkpoint,
    sha256,
    validate_training_checkpoint,
    verify_artifact,
    write_json,
    write_torch_gzip_atomic,
)


SCHEMA = "egoengine_taco_pour_virtual_object_assist_v1"
RUN_DIRECTORY = "runs/taco_pour_virtual_object_assist_v1"
BASE_COMMIT = "0975392070bc7dac20d3b4ae7945b09eca3aa746"


def _runner():
    # Lazy import avoids a module cycle: runner dispatches to this module.
    from . import runner

    return runner


def _load(
    path: Path, asset_root_override: Path | None
) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported virtual-assist configuration")
    if config.get("status") != "authorized_bounded_single_seed_training":
        raise ValueError("virtual-assist pilot is not authorized")
    if config.get("base_commit") != BASE_COMMIT:
        raise ValueError("virtual-assist base commit changed")
    if any(
        (
            config.get("paper_faithful") is not False,
            config.get("training_dynamics_changed") is not True,
            config.get("evaluation_dynamics_changed") is not False,
            config.get("chunk_commit_enabled") is not False,
            config.get("automatic_followon") is not False,
        )
    ):
        raise ValueError("virtual-assist classification flags changed")
    execution = config["execution"]
    expected_execution = {
        "task": "TACO_20230927_017",
        "device": "cpu",
        "seed": 0,
        "tracking_variant": "tool_only",
        "source_endpoint": 40,
        "final_endpoint": 80,
        "worlds": 4,
        "horizon": 40,
        "sequence_length": 4,
        "actor_observation_dim": 236,
        "privileged_extra_dim": 108,
        "critic_input_dim": ASSISTED_CRITIC_INPUT_DIM,
        "critic_input_spec": ASSISTED_CRITIC_INPUT_SPEC,
        "actions": 36,
        "physics_steps_per_control": 10,
        "reset_distribution": "original_committed_s40_only",
    }
    if execution != expected_execution:
        raise ValueError("virtual-assist execution contract changed")
    if config["validation"]["official_epochs"] != [0, 50, 100, 150, 200, 250]:
        raise ValueError("official evaluation schedule changed")
    if config["validation"]["assisted_diagnostic_epochs"] != [50, 150]:
        raise ValueError("assisted diagnostic schedule changed")
    budget = config["budget"]
    frozen_budget = {
        "verification_and_all_initialization_physics_steps_max": 16000,
        "functional_attempts_max": 2,
        "tiny_body_test_physics_steps_max_per_attempt": 200,
        "formal_training_physics_steps_max": 400000,
        "formal_training_control_intervals_max": 40000,
        "fixed_evaluation_physics_steps_max": 3200,
        "strict_success_confirmation_physics_steps_reserved": 800,
        "all_in_physics_steps_max": 420000,
        "formal_actor_optimizer_steps_max": 250,
        "formal_critic_optimizer_steps_max": 1000,
        "all_in_actor_optimizer_steps_max": 256,
        "all_in_critic_optimizer_steps_max": 1024,
        "auto_restart_formal_run": False,
        "reinvest_early_stop_savings": False,
        "all_attempts_share_persistent_ledger": True,
        "count_native_setup_and_warmup_steps": True,
    }
    if budget != frozen_budget:
        raise ValueError("virtual-assist budget changed")
    root = (asset_root_override or Path(config["asset_root"])).resolve(strict=True)
    assets: dict[str, Path] = {}
    for name, row in config["assets"].items():
        candidate = (root / row["path"]).resolve(strict=True)
        if row.get("sha256"):
            verify_artifact(candidate, row["sha256"])
        assets[name] = candidate
    expected_blobs = {
        "rl/external/spider_compat/spider/simulators/mjwp.py": config["sources"][
            "original_backend_blob_sha"
        ],
        "rl/src/video_to_spider/rl/mjwp_env.py": config["sources"][
            "original_environment_blob_sha"
        ],
    }
    for file_name, expected in expected_blobs.items():
        observed = subprocess.check_output(
            ["git", "rev-parse", f"{BASE_COMMIT}:{file_name}"], text=True
        ).strip()
        if observed != expected:
            raise ValueError(f"base blob identity changed for {file_name}")
    return config, root, assets


def _spec(config: dict[str, Any]) -> ToolAssistSpec:
    row = config["assistance"]
    return ToolAssistSpec(
        body_name=row["expected_tool_body_name"],
        position_frequency=float(row["natural_frequency_position_per_s"]),
        rotation_frequency=float(row["natural_frequency_rotation_per_s"]),
        damping_ratio=float(row["damping_ratio"]),
        force_cap_gravity_multiple=3.0,
        torque_cap_acceleration=100.0,
        substeps=10,
        control_dt=1.0 / 30.0,
    )


def _make_world(
    assets: dict[str, Path],
    boundary: dict[str, Any],
    seed: int,
    *,
    asymmetric: bool,
    spec: ToolAssistSpec | None,
):
    return _runner()._make_core_world(
        assets,
        boundary,
        seed,
        asymmetric=asymmetric,
        object_assistance=spec,
    )


def _make_runtime(
    config: dict[str, Any],
    assets: dict[str, Path],
    *,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _runner()._seed_everything(0)
    context = load_boundary_context(assets["context"])
    donor = load_torch_gzip(assets["donor"])
    if donor.get("schema") != "taco_pour_algorithmic_benchmark_checkpoint_v1":
        raise ValueError("unsupported donor checkpoint")
    policy = PolicyBundle.create(
        worlds=4,
        with_critic=True,
        critic_input_dim=ASSISTED_CRITIC_INPUT_DIM,
        critic_input_spec=ASSISTED_CRITIC_INPUT_SPEC,
    )
    policy.load_actor_only(
        donor["actor"], version=int(donor["observation_normalization_version"])
    )
    assist = _spec(config)
    worlds = [
        _make_world(
            assets,
            context["physics_state"],
            index,
            asymmetric=True,
            spec=assist,
        )
        for index in range(4)
    ]
    environment = IndependentWorlds(worlds)
    boundary = environment.states()[0]
    prefix = tuple(context["observation_prefix"])
    collector = FixedBoundaryCollector(
        environment=environment,
        policy=policy,
        boundary_state=boundary,
        observation_prefix=prefix,
        horizon=40,
        start_endpoint=40,
        end_endpoint=80,
    )
    trainer = PPOTrainer(policy=policy, collector=collector, config=PPOConfig())
    if checkpoint is not None:
        validate_training_checkpoint(
            checkpoint,
            expected_critic_input_spec=ASSISTED_CRITIC_INPUT_SPEC,
            expected_critic_input_dimension=ASSISTED_CRITIC_INPUT_DIM,
        )
        if not _runner()._nested_equal(checkpoint["boundary_state"], boundary):
            raise ValueError("resume boundary differs from committed s40")
        if not _runner()._nested_equal(checkpoint["observation_prefix"], prefix):
            raise ValueError("resume prefix differs from sources20--39")
        restore_training_checkpoint(policy, checkpoint)
    return {
        "policy": policy,
        "environment": environment,
        "boundary_state": boundary,
        "observation_prefix": prefix,
        "collector": collector,
        "trainer": trainer,
    }


def _evaluate(
    assets: dict[str, Path],
    *,
    actor_state: dict[str, torch.Tensor],
    normalization_version: int,
    boundary_state: dict[str, Any],
    prefix: tuple[Any, ...],
    spec: ToolAssistSpec | None,
    alpha: float,
    zero_residual: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any] | None]:
    arrays, endpoint60 = _runner()._evaluation_arrays(
        assets,
        actor_state=actor_state,
        normalization_version=normalization_version,
        boundary_state=boundary_state,
        observation_prefix=prefix,
        seed=0,
        object_assistance=spec,
        assistance_alpha=alpha,
        zero_residual=zero_residual,
    )
    summary = _runner()._evaluation_summary(arrays)
    summary.update(
        assistance_alpha=float(alpha),
        dynamics_label="assisted" if alpha > 0.0 else "original_unassisted",
        strict_success_eligible=bool(alpha == 0.0),
    )
    return summary, arrays, endpoint60


def _tiny_body_test() -> dict[str, Any]:
    xml = """
    <mujoco><option timestep="0.0033333333333333335" gravity="0 0 -9.81"/>
      <worldbody><body name="tool" pos="0 0 1"><freejoint/>
      <geom type="sphere" size="0.05" mass="1"/></body></worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    before = data.qvel.copy()
    data.xfrc_applied[1, 0] = 1.0
    data.xfrc_applied[1, 3] = 0.1
    mujoco.mj_step(model, data)
    return {
        "physics_steps": 1,
        "world_force_changes_linear_velocity": bool(data.qvel[0] > before[0]),
        "world_torque_changes_angular_velocity": bool(data.qvel[3] > before[3]),
        "passed": bool(data.qvel[0] > before[0] and data.qvel[3] > before[3]),
    }


def _arrays_exact(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return set(left) == set(right) and all(
        left[name].dtype == right[name].dtype
        and left[name].shape == right[name].shape
        and left[name].tobytes() == right[name].tobytes()
        for name in left
    )


def _checkpoint(
    output: Path,
    runtime: dict[str, Any],
    config_path: Path,
    assets: dict[str, Path],
    epoch: int,
    *,
    milestone: bool,
) -> dict[str, Any]:
    payload = build_training_checkpoint(
        policy=runtime["policy"],
        boundary_state=runtime["boundary_state"],
        observation_prefix=runtime["observation_prefix"],
        next_epoch=epoch + 1,
        cost={
            "training_control_intervals": epoch * 160,
            "training_physics_steps": epoch * 1600,
            "actor_optimizer_steps": epoch,
            "critic_optimizer_steps": epoch * 4,
        },
        metadata={
            "config_sha256": sha256(config_path),
            "donor_sha256": sha256(assets["donor"]),
            "boundary_context_sha256": sha256(assets["context"]),
            "critic_input_spec": ASSISTED_CRITIC_INPUT_SPEC,
            "critic_input_dimension": ASSISTED_CRITIC_INPUT_DIM,
            "training_dynamics": "virtual_object_assist_fixed_schedule",
            "official_evaluation_dynamics": "original_unassisted",
            "chunk_commit_authorized": False,
        },
    )
    write_torch_gzip_atomic(output / "checkpoints/latest.pt.gz", payload)
    if milestone:
        write_torch_gzip_atomic(
            output / f"checkpoints/epoch_{epoch:04d}.pt.gz", payload
        )
    return payload


def inspect_virtual(
    config_path: Path, asset_root: Path | None
) -> dict[str, Any]:
    config, root, assets = _load(config_path, asset_root)
    return {
        "status": "INSPECT_OK",
        "config": manifest_entry(config_path),
        "asset_root": str(root),
        "assets": {name: manifest_entry(path) for name, path in assets.items()},
        "controller": _spec(config).__dict__,
        "paper_faithful": False,
        "training_dynamics_changed": True,
        "official_evaluation_assistance_alpha": 0.0,
        "chunk_commit_enabled": False,
    }


def verify_virtual(
    config_path: Path, asset_root: Path | None, output: Path
) -> dict[str, Any]:
    """Run the only authorized bounded preflight and cold-resume gate."""
    started = time.time()
    config, root, assets = _load(config_path, asset_root)
    output.mkdir(parents=True, exist_ok=True)
    prior = output / "verification.json"
    previous = json.loads(prior.read_text()) if prior.is_file() else None
    attempt = 1 if previous is None else len(previous.get("attempts", ())) + 1
    if attempt > config["budget"]["functional_attempts_max"]:
        raise RuntimeError("virtual-assist verification attempt budget exhausted")
    runtime = _make_runtime(config, assets)
    policy = runtime["policy"]
    actor = deepcopy(policy.actor.state_dict())
    version = int(policy.normalization_version)
    boundary = runtime["boundary_state"]
    prefix = runtime["observation_prefix"]
    spec = _spec(config)
    probes = []
    probe_arrays: list[dict[str, np.ndarray]] = []
    definitions = (
        ("original_path_donor_alpha0", None, 0.0, False),
        ("configured_path_donor_alpha0", spec, 0.0, False),
        ("configured_path_donor_alpha1", spec, 1.0, False),
        ("configured_path_zero_residual_alpha1", spec, 1.0, True),
        ("reset_original_s40_then_donor_alpha0", spec, 0.0, False),
    )
    probe_controls = 0
    for name, probe_spec, alpha, zero in definitions:
        summary, arrays, _ = _evaluate(
            assets,
            actor_state=actor,
            normalization_version=version,
            boundary_state=boundary,
            prefix=prefix,
            spec=probe_spec,
            alpha=alpha,
            zero_residual=zero,
        )
        summary["name"] = name
        probes.append(summary)
        probe_arrays.append(arrays)
        probe_controls += summary["executed_control_intervals"]
        _runner()._write_npz_atomic(output / f"preflight/{name}.npz", arrays)
    alpha0_exact = _arrays_exact(probe_arrays[0], probe_arrays[1]) and _arrays_exact(
        probe_arrays[0], probe_arrays[4]
    )
    historical = _runner()._compare_epoch0_to_v1(
        output / "preflight/original_path_donor_alpha0.npz",
        assets["donor_epoch0_trajectory"],
    )
    assistance_opens_tail = probes[2]["valid_prefix_intervals"] >= int(
        config["validation"]["preflight_donor_alpha1_minimum_valid_prefix"]
    )
    tiny = _tiny_body_test()
    if not alpha0_exact or not historical["all_common_arrays_bitwise_equal"]:
        passed_preflight = False
    else:
        passed_preflight = bool(tiny["passed"] and assistance_opens_tail)
    functional = {"ran": False, "passed": False}
    optimizer_cost = {"actor": 0, "critic": 0}
    functional_controls = 0
    if passed_preflight:
        continuous = _make_runtime(config, assets)
        continuous["environment"].set_assistance_alpha(1.0)
        first = continuous["trainer"].run_epoch()
        checkpoint = build_training_checkpoint(
            policy=continuous["policy"],
            boundary_state=continuous["boundary_state"],
            observation_prefix=continuous["observation_prefix"],
            next_epoch=2,
            cost={
                "training_control_intervals": 160,
                "training_physics_steps": 1600,
                "actor_optimizer_steps": 1,
                "critic_optimizer_steps": 4,
            },
            metadata={"config_sha256": sha256(config_path), "verification_only": True},
        )
        checkpoint_path = output / "epoch_0001_verification_checkpoint.pt.gz"
        write_torch_gzip_atomic(checkpoint_path, checkpoint)
        second = continuous["trainer"].run_epoch()
        state_continuous = _runner()._training_state(continuous)
        resumed = _make_runtime(
            config, assets, checkpoint=load_torch_gzip(checkpoint_path)
        )
        resumed["environment"].set_assistance_alpha(1.0)
        second_resumed = resumed["trainer"].run_epoch()
        state_resumed = _runner()._training_state(resumed)
        exact = {
            "batch": _runner()._batch_exact(second["batch"], second_resumed["batch"]),
            "report": _runner()._nested_equal(
                {k: v for k, v in second.items() if k != "batch"},
                {k: v for k, v in second_resumed.items() if k != "batch"},
            ),
            "state": _runner()._nested_equal(state_continuous, state_resumed),
        }
        alpha_exact = all(
            bool((item["batch"].assistance_alpha == 1.0).all())
            for item in (first, second, second_resumed)
        )
        functional = {
            "ran": True,
            "passed": bool(all(exact.values()) and alpha_exact),
            "cold_resume": exact,
            "alpha_constant_and_visible_to_critic": alpha_exact,
            "likelihood_identity": [
                first["actor"]["live_rollout_ratio_max_abs_error"],
                second["actor"]["live_rollout_ratio_max_abs_error"],
                second_resumed["actor"]["live_rollout_ratio_max_abs_error"],
            ],
        }
        functional_controls = 480
        optimizer_cost = {"actor": 3, "critic": 12}
    physics = (probe_controls + functional_controls) * 10 + tiny["physics_steps"]
    prior_physics = int(previous.get("cost", {}).get("physics_steps", 0)) if previous else 0
    cost = {
        "physics_steps": prior_physics + physics,
        "this_attempt_physics_steps": physics,
        "actor_optimizer_steps": optimizer_cost["actor"],
        "critic_optimizer_steps": optimizer_cost["critic"],
        "attempts": attempt,
    }
    if cost["physics_steps"] > config["budget"][
        "verification_and_all_initialization_physics_steps_max"
    ]:
        raise RuntimeError("verification/init budget exceeded")
    passed = bool(passed_preflight and functional["passed"])
    status = (
        "VIRTUAL_ASSIST_TRAINING_CHAIN_VERIFIED"
        if passed
        else (
            "ASSISTANCE_NOT_OPENING_TAIL"
            if not assistance_opens_tail
            else "VIRTUAL_ASSIST_VERIFICATION_FAILED"
        )
    )
    manifests = runtime["environment"].assistance_manifests()
    result = {
        "status": status,
        "training_authorized": passed,
        "config": manifest_entry(config_path),
        "preflight_probes": probes,
        "alpha0_original_path_bitwise": alpha0_exact,
        "historical_donor_parity": historical,
        "tiny_free_body": tiny,
        "functional_validation": functional,
        "controller_manifest": manifests,
        "cost": cost,
        "chunk_commit_enabled": False,
        "attempts": [
            *(previous.get("attempts", []) if previous else []),
            {"ordinal": attempt, "status": status, "physics_steps": physics},
        ],
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "verification.json", result)
    write_json(output / "controller_manifest.json", manifests[0])
    write_json(output / "input_manifest.json", {
        name: manifest_entry(path) for name, path in assets.items()
    })
    write_json(output / "cost.json", cost)
    return result


def _assistance_epoch_metrics(epoch: int, report: dict[str, Any]) -> dict[str, Any]:
    row = _runner()._epoch_metrics(epoch, report)
    batch = report["batch"]
    wrench = batch.assistance_wrench.to(torch.float64)
    force_norm = torch.linalg.vector_norm(wrench[..., :3], dim=-1)
    torque_norm = torch.linalg.vector_norm(wrench[..., 3:], dim=-1)
    velocity = batch.critic_observations[:, 236:242].to(torch.float64)
    estimated_power = (
        wrench[..., :3] * velocity[:, None, :3]
        + wrench[..., 3:] * velocity[:, None, 3:6]
    ).sum(dim=-1)
    right_tool = batch.contact_flags[:, 0, 0].any(dim=-1)
    row["assistance"] = {
        "alpha": float(batch.assistance_alpha[0]),
        "dynamics_label": (
            "assisted" if float(batch.assistance_alpha[0]) > 0.0 else "original_unassisted"
        ),
        "maximum_force_norm": float(force_norm.max()),
        "maximum_torque_norm": float(torque_norm.max()),
        "force_cap_count": int(batch.assistance_force_cap_count.sum()),
        "torque_cap_count": int(batch.assistance_torque_cap_count.sum()),
        "integrated_force_norm_estimate": float(force_norm.sum() / 300.0),
        "integrated_torque_norm_estimate": float(torque_norm.sum() / 300.0),
        "estimated_work_from_transition_start_privileged_velocity": float(
            estimated_power.sum() / 300.0
        ),
    }
    row["real_right_tool_contact"] = {
        "first_20_fraction": float(right_tool[:80].float().mean()),
        "last_20_fraction": float(right_tool[80:].float().mean()),
    }
    return row


def _record_eval(
    output: Path,
    *,
    label: str,
    epoch: int,
    runtime: dict[str, Any],
    assets: dict[str, Path],
    spec: ToolAssistSpec | None,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any] | None]:
    summary, arrays, endpoint60 = _evaluate(
        assets,
        actor_state=deepcopy(runtime["policy"].actor.state_dict()),
        normalization_version=runtime["policy"].normalization_version,
        boundary_state=runtime["boundary_state"],
        prefix=runtime["observation_prefix"],
        spec=spec,
        alpha=alpha,
    )
    summary.update(epoch=epoch, label=label)
    _runner()._write_npz_atomic(output / f"evaluations/{label}_epoch_{epoch:04d}.npz", arrays)
    write_json(output / f"evaluations/{label}_epoch_{epoch:04d}.json", summary)
    return summary, arrays, endpoint60


def _coverage_count(batch: Any) -> int:
    count = 0
    for world in range(4):
        mask = (
            (batch.world_index == world)
            & (batch.episode_serial == 0)
            & (batch.outcome_endpoint == 70)
            & ~batch.terminated
        )
        count += int(bool(mask.any()))
    return count


def _write_server_artifact_hashes(output: Path) -> None:
    """Hash the immutable server evidence without recursively hashing the list."""
    target = output / "server_artifacts.sha256"
    rows = [
        f"{sha256(path)}  {path.relative_to(output)}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != target
    ]
    target.write_text("\n".join(rows) + "\n")


def finalize_virtual_artifacts(output: Path, decision: dict[str, Any]) -> None:
    """Write the bounded run's required lightweight terminal artifacts."""
    cost = decision["cost"]
    write_json(output / "total_cost_accounting.json", cost)
    official = decision["evaluations"]
    official_text = ", ".join(
        f"e{row['epoch']}={row['valid_prefix_intervals']}/40"
        for row in official
    )
    assisted = []
    for path in sorted((output / "evaluations").glob("assisted_diagnostic_*.json")):
        row = json.loads(path.read_text())
        assisted.append(
            f"e{row['epoch']}@alpha={row['assistance_alpha']:.6g}="
            f"{row['valid_prefix_intervals']}/40"
        )
    assisted_text = ", ".join(assisted) if assisted else "none"
    (output / "summary.md").write_text(
        "# Virtual object assist v1\n\n"
        f"Status: `{decision['status']}`\n\n"
        "This local, non-paper-faithful experiment changed training dynamics "
        "with a bounded tool wrench. All official evaluations used the original "
        "unassisted dynamics from the committed endpoint-40 boundary.\n\n"
        f"- completed epoch: `{decision['completed_epoch']}`\n"
        f"- assisted tail coverage at epochs 41--50: "
        f"`{decision['assisted_tail_eligible_episodes_passing_endpoint70']}/40`\n"
        f"- official unassisted fixed evaluations: `{official_text}`\n"
        f"- assisted diagnostics (not success eligible): `{assisted_text}`\n"
        f"- all-in physics steps: `{cost['all_in_physics_steps']}`\n"
        f"- actor / critic optimizer steps: "
        f"`{cost['actor_optimizer_steps']} / {cost['critic_optimizer_steps']}`\n"
        "- strict chunk commit: `false`\n"
        "- automatic follow-on: `false`\n"
    )
    _write_server_artifact_hashes(output)


def train_virtual(
    config_path: Path,
    asset_root: Path | None,
    output: Path,
    *,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    config, root, assets = _load(config_path, asset_root)
    verification_path = root / RUN_DIRECTORY / "verification.json"
    if not verification_path.is_file():
        raise RuntimeError("virtual-assist verification report is missing")
    verification = json.loads(verification_path.read_text())
    if verification.get("status") != "VIRTUAL_ASSIST_TRAINING_CHAIN_VERIFIED":
        raise RuntimeError("virtual-assist verification did not authorize training")
    if verification.get("config", {}).get("sha256") != sha256(config_path):
        raise RuntimeError("verification used a different config")
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "checkpoints/latest.pt.gz"
    if resume_path is None and latest.exists():
        raise FileExistsError("formal output already contains a checkpoint")
    checkpoint = load_torch_gzip(resume_path) if resume_path else None
    if checkpoint is not None:
        validate_training_checkpoint(
            checkpoint,
            expected_critic_input_spec=ASSISTED_CRITIC_INPUT_SPEC,
            expected_critic_input_dimension=ASSISTED_CRITIC_INPUT_DIM,
        )
        if checkpoint["metadata"].get("config_sha256") != sha256(config_path):
            raise ValueError("resume checkpoint used another contract")
    runtime = _make_runtime(config, assets, checkpoint=checkpoint)
    spec = _spec(config)
    start_epoch = int(checkpoint["next_epoch"]) if checkpoint else 1
    verification_physics = int(verification["cost"]["physics_steps"])
    worst_case = verification_physics + 400000 + 3200 + 800
    if worst_case > config["budget"]["all_in_physics_steps_max"]:
        raise RuntimeError("all-in budget lacks worst-case capacity before formal training")
    write_json(output / "frozen_config.json", config)
    write_json(output / "input_manifest.json", {
        name: manifest_entry(path) for name, path in assets.items()
    })
    write_json(output / "controller_manifest.json", runtime["environment"].assistance_manifests()[0])
    evaluations: list[dict[str, Any]] = []
    evaluation_controls = 0
    coverage = 0
    completed = start_epoch - 1
    success_endpoint60 = None
    status = "RUNNING"
    if checkpoint is None:
        _checkpoint(output, runtime, config_path, assets, 0, milestone=True)
        initial, arrays, _ = _record_eval(
            output,
            label="official_unassisted",
            epoch=0,
            runtime=runtime,
            assets=assets,
            spec=None,
            alpha=0.0,
        )
        evaluations.append(initial)
        evaluation_controls += initial["executed_control_intervals"]
        parity = _runner()._compare_epoch0_to_v1(
            output / "evaluations/official_unassisted_epoch_0000.npz",
            assets["donor_epoch0_trajectory"],
        )
        write_json(output / "epoch0_donor_parity.json", parity)
        if (
            not parity["all_common_arrays_bitwise_equal"]
            or initial["valid_prefix_intervals"] != 20
            or initial["first_failure_endpoint"] != 61
        ):
            raise RuntimeError("formal epoch-0 donor parity failed")
    for epoch in range(start_epoch, 251):
        alpha = assistance_scale(epoch)
        runtime["environment"].set_assistance_alpha(alpha)
        report = runtime["trainer"].run_epoch()
        metrics = _assistance_epoch_metrics(epoch, report)
        append_jsonl(output / "training_metrics.jsonl", metrics)
        append_rollout_npz(output / "training_batches.npz", epoch=epoch, batch=report["batch"])
        if 41 <= epoch <= 50:
            coverage += _coverage_count(report["batch"])
        milestone = epoch in config["validation"]["official_epochs"]
        payload = _checkpoint(
            output, runtime, config_path, assets, epoch, milestone=milestone
        )
        completed = epoch
        if milestone:
            official, arrays, endpoint60 = _record_eval(
                output,
                label="official_unassisted",
                epoch=epoch,
                runtime=runtime,
                assets=assets,
                spec=None,
                alpha=0.0,
            )
            evaluations.append(official)
            evaluation_controls += official["executed_control_intervals"]
            if official["strict_40_of_40"]:
                confirmation, confirm_arrays, confirm_endpoint60 = _record_eval(
                    output,
                    label="strict_confirmation_unassisted",
                    epoch=epoch,
                    runtime=runtime,
                    assets=assets,
                    spec=None,
                    alpha=0.0,
                )
                evaluation_controls += confirmation["executed_control_intervals"]
                if not confirmation["strict_40_of_40"] or not _arrays_exact(arrays, confirm_arrays):
                    raise RuntimeError("cold strict confirmation diverged")
                # The full run captures the actual endpoint-60 state and its
                # matching recurrent state.  Replaying the second half from it
                # is verified by the helper below.
                if confirm_endpoint60 is None or not _validate_split(
                    assets, payload, runtime["observation_prefix"], confirm_endpoint60, confirm_arrays
                ):
                    raise RuntimeError("split-at-actual-s60 validation diverged")
                evaluation_controls += 20
                success_endpoint60 = confirm_endpoint60
                status = "STRICT_UNASSISTED_WINDOW_SUCCESS"
                break
        if epoch in config["validation"]["assisted_diagnostic_epochs"]:
            assisted, _, _ = _record_eval(
                output,
                label="assisted_diagnostic",
                epoch=epoch,
                runtime=runtime,
                assets=assets,
                spec=spec,
                alpha=alpha,
            )
            evaluation_controls += assisted["executed_control_intervals"]
        if epoch == 50 and coverage < 20:
            status = "ASSISTED_TAIL_COVERAGE_NOT_ESTABLISHED"
            break
    if status == "RUNNING":
        status = "COMPLETED_NO_STRICT_WINDOW_SUCCESS"
    if evaluation_controls * 10 > config["budget"]["fixed_evaluation_physics_steps_max"] + (
        config["budget"]["strict_success_confirmation_physics_steps_reserved"]
        if success_endpoint60 is not None
        else 0
    ):
        raise RuntimeError("evaluation budget exceeded")
    if success_endpoint60 is not None:
        write_torch_gzip_atomic(output / "endpoint60_candidate_run_only.pt.gz", success_endpoint60)
    cost = {
        "verification_and_initialization_physics_steps": verification_physics,
        "formal_training_control_intervals": completed * 160,
        "formal_training_physics_steps": completed * 1600,
        "fixed_evaluation_control_intervals": evaluation_controls,
        "fixed_evaluation_physics_steps": evaluation_controls * 10,
        "actor_optimizer_steps": completed,
        "critic_optimizer_steps": completed * 4,
        "all_in_physics_steps": verification_physics + completed * 1600 + evaluation_controls * 10,
    }
    if cost["all_in_physics_steps"] > config["budget"]["all_in_physics_steps_max"]:
        raise RuntimeError("all-in physics budget exceeded")
    decision = {
        "status": status,
        "completed_epoch": completed,
        "assisted_tail_eligible_episodes_passing_endpoint70": coverage,
        "evaluations": evaluations,
        "cost": cost,
        "training_dynamics_changed": True,
        "official_evaluation_dynamics_changed": False,
        "assisted_experience_without_verified_handoff": status
        != "STRICT_UNASSISTED_WINDOW_SUCCESS",
        "chunk_commit_enabled": False,
        "next_action": "stop; no automatic follow-on is authorized",
    }
    write_json(output / "decision.json", decision)
    write_json(output / "cost.json", cost)
    finalize_virtual_artifacts(output, decision)
    return decision


def _validate_split(
    assets: dict[str, Path],
    checkpoint: dict[str, Any],
    prefix: tuple[Any, ...],
    endpoint60: dict[str, Any],
    full_arrays: dict[str, np.ndarray],
) -> bool:
    policy = PolicyBundle.create(worlds=1, with_critic=False)
    policy.load_actor_only(
        checkpoint["actor"],
        version=int(checkpoint["observation_normalization_version"]),
    )
    states = tuple(state.clone() for state in endpoint60["rnn_states"])
    world = _make_world(
        assets, checkpoint["boundary_state"], 0, asymmetric=False, spec=None
    )
    world.set_env_state(endpoint60["physics_state"])
    observed = []
    for _ in range(20):
        observation = torch.as_tensor(world.current_observation(), dtype=torch.float32)
        low, high = world.current_normalized_action_bounds()
        action, result = policy_step(
            policy.actor,
            observation,
            states,
            low,
            high,
            stochastic=False,
            distribution=policy.distribution,
        )
        states = result.rnn_states
        _, _, _, info = world.step(action, auto_reset=False)
        observed.append(float(info["object_tracking_error"][0]))
    return np.asarray(observed).tobytes() == np.asarray(full_arrays["tracking_score"][20:]).tobytes()


def evaluate_virtual(
    config_path: Path,
    asset_root: Path | None,
    checkpoint_path: Path,
    output: Path,
) -> dict[str, Any]:
    config, _, assets = _load(config_path, asset_root)
    payload = load_torch_gzip(checkpoint_path)
    validate_training_checkpoint(
        payload,
        expected_critic_input_spec=ASSISTED_CRITIC_INPUT_SPEC,
        expected_critic_input_dimension=ASSISTED_CRITIC_INPUT_DIM,
    )
    summary, arrays, _ = _evaluate(
        assets,
        actor_state=payload["actor"],
        normalization_version=int(payload["observation_normalization_version"]),
        boundary_state=payload["boundary_state"],
        prefix=tuple(payload["observation_prefix"]),
        spec=None,
        alpha=0.0,
    )
    result = {
        "status": "STRICT_WINDOW_SUCCESS" if summary["strict_40_of_40"] else "STRICT_WINDOW_FAILED",
        "evaluation": summary,
        "assistance_forced_to_zero": True,
        "chunk_commit_enabled": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evaluation.json", result)
    _runner()._write_npz_atomic(output / "evaluation_trajectory.npz", arrays)
    return result

