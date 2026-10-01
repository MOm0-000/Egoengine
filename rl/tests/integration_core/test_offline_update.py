from pathlib import Path

from video_to_spider.rl.core.runner import _offline_actor_parity


def test_three_saved_batches_reproduce_full_actor_exactly():
    root = Path("/data_all/zzx/3.2RL/runs/taco_pour_candidate_G_value_loss_isolation_v1")
    rows, passed = _offline_actor_parity(root)
    assert passed
    assert len(rows) == 3
    assert all(row["actor_optimizer_steps"] == 2 for row in rows)
