# RL core R1 legacy archive

This directory is the recoverable relocation of the superseded RL
implementation after the R1 core passed bounded verification.

Relocation source:

- repository: `MOm0-000/Egoengine`
- branch: `3.2RL`
- implementation baseline before R1: `a1622a4055160ee600c02bfaf88c47f2cbcbf2ee`
- historical numerical oracle: `96e8c032595cb0db7126923804e7733e3e2ef3c9`
- relocation date: `2026-10-01`

The archive contains 146 retired files (145 moved plus one full-source safety
copy used while extracting the active environment):

- 12 superseded `src/video_to_spider/rl` orchestration modules plus the
  pre-extraction `mjwp_env.py` containing `IndependentMJWPTrainingEnv`;
- 78 runners/audits/summarizers in their dependency closure;
- 28 tests tied to those retired entrypoints;
- 24 historical execution configurations;
- 3 historical handoff documents.

Two evidence-only YAML contracts remain in `configs/` because active read-only
tests bind their hashes; neither is an executable entrypoint.

`MANIFEST.sha256` records each archived file's hash and its original repository-relative
path. The dependency closure was moved with `git mv`; the full environment
source was copied before its retired multi-world class was removed from the
active adapter. Recovery is possible from this directory or Git history. These
files must not be imported by active source, scripts or tests.

The sole active entrypoint is now:

```text
scripts/run_rl.py -> video_to_spider.rl.core.runner
```

Historical run artifacts were deliberately not moved. The new verifier reads
several immutable checkpoints, batches and trajectories from `runs/` by
SHA-256, so those files remain evidence rather than an alternate code path.

The low-level `MJWPVectorEnv` remains active because the compact core uses it as
the audited MuJoCo-Warp physics adapter. The old Candidate/PpoAgent training
orchestration and `IndependentMJWPTrainingEnv` were removed from active source.
