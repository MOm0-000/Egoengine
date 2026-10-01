from types import SimpleNamespace

import numpy as np
import torch

from video_to_spider.rl.core.audit import append_jsonl, append_rollout_npz


def test_epoch_logging_is_passive_and_npz_is_resume_safe(tmp_path):
    torch.manual_seed(17)
    before = torch.get_rng_state().clone()
    batch = SimpleNamespace(
        actions=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        reset_states=(torch.ones(1, 2, 4),),
        normalization_version=4,
    )
    append_jsonl(tmp_path / "metrics.jsonl", {"epoch": 1, "loss": 0.5})
    append_rollout_npz(tmp_path / "batches.npz", epoch=1, batch=batch)
    assert torch.equal(before, torch.get_rng_state())
    with np.load(tmp_path / "batches.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["epoch_0001/actions"], batch.actions.numpy())
        assert np.array_equal(
            archive["epoch_0001/reset_states_0"], batch.reset_states[0].numpy()
        )
