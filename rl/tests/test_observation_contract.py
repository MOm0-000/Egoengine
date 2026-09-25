from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_to_spider.rl.observation_contract import load_runtime_observation


PROTOCOL = ROOT / "configs/replay_rl_protocol.yaml"
PROFILE = ROOT / "configs/taco_pour_observation_local_236d_v1.yaml"


def test_exact_observation_requires_an_explicit_local_profile():
    with pytest.raises(ValueError, match="paper does not publish"):
        load_runtime_observation(PROTOCOL, None, require_run_ready=True)


def test_transition_aligned_local_observation_is_fully_resolved():
    observation = load_runtime_observation(PROTOCOL, PROFILE, require_run_ready=False)
    assert observation.observation_id == "taco_pour_transition_aligned_236d_v1"
    assert not observation.paper_faithful
    assert observation.actor_dim == 236 and observation.critic_dim == 108
    assert observation.goal_reference_offset == observation.command_offset == 1
    assert observation.command_preview_offset == 2
    assert observation.include_command_preview and observation.include_contact_flags
    assert observation.profile_sha256


def test_formal_observation_is_ready_after_reward_alignment_fix():
    observation = load_runtime_observation(PROTOCOL, PROFILE, require_run_ready=True)
    assert observation.observation_id == "taco_pour_transition_aligned_236d_v1"


def test_profile_content_is_hash_bound_to_protocol(tmp_path):
    changed = tmp_path / "changed.yaml"
    profile = yaml.safe_load(PROFILE.read_text())
    profile["actor"]["components"][5]["timestamp"] = "t"
    changed.write_text(yaml.safe_dump(profile, sort_keys=False))
    with pytest.raises(ValueError, match="hash does not match"):
        load_runtime_observation(PROTOCOL, changed, require_run_ready=False)
