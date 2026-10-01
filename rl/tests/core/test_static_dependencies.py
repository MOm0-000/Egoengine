import ast
from pathlib import Path


FORBIDDEN = {
    "video_to_spider.rl.algorithmic_training",
    "video_to_spider.rl.algorithmic_training_v5",
    "video_to_spider.rl.algorithmic_training_v6",
    "video_to_spider.rl.policy_warm_start",
    "video_to_spider.rl.state_feasible_truncated_gaussian",
    "human2sim2robot.ppo.ppo_agent",
}

RETIRED_ACTIVE_PATHS = (
    "src/video_to_spider/rl/algorithmic_training.py",
    "src/video_to_spider/rl/algorithmic_training_v5.py",
    "src/video_to_spider/rl/algorithmic_training_v6.py",
    "src/video_to_spider/rl/policy_warm_start.py",
    "src/video_to_spider/rl/replay_rl.py",
    "src/video_to_spider/rl/state_feasible_truncated_gaussian.py",
    "scripts/run_mjwp_ppo.py",
    "scripts/run_taco_replay_rl.py",
    "scripts/run_taco_pour_candidate_G_policy_warm_start_v1.py",
)


def test_core_has_no_historical_trainer_dependency():
    root = Path(__file__).resolve().parents[2] / "src/video_to_spider/rl/core"
    found = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if any(name == blocked or name.startswith(blocked + ".") for blocked in FORBIDDEN):
                    found.append((path.name, name))
    assert found == []


def test_retired_implementations_are_not_beside_the_active_core():
    root = Path(__file__).resolve().parents[2]
    assert all(not (root / relative).exists() for relative in RETIRED_ACTIVE_PATHS)
    environment_source = (root / "src/video_to_spider/rl/mjwp_env.py").read_text()
    assert "class IndependentMJWPTrainingEnv" not in environment_source
    assert (root / "scripts/run_rl.py").is_file()
