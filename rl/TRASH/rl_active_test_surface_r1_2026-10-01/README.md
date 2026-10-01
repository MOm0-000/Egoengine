# Retired default test surface

This directory contains 55 tests that were removed from default collection
during the RL active-test-surface cleanup on 2026-10-01.

They cover retired Candidate/Gate runners, source-specific counterfactual
audits, or pre-core data/physics stages. Their assertions remain useful as
historical evidence, but many require immutable artifacts or absolute paths
that are not present in a clean checkout. They must not make an unrelated
change to `video_to_spider.rl.core` appear red.

No test was deleted. `MANIFEST.sha256` records each original path and byte
hash. Explicit historical collection is documented in `tests/README.md`.
