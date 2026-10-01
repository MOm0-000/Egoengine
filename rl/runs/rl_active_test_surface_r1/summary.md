# RL active test surface R1

Status: `ACTIVE_TEST_SURFACE_GREEN`

- Default clean-checkout surface: `30 passed`, repeated successfully in a
  detached worktree containing only committed files.
- Explicit saved-batch integration: `1 passed` with the immutable asset root.
- Historical relocation: 55 files, 377 tests collectable only by explicit
  selection.
- Runtime behavior changed: `false`.
- Training enabled: `false`.
- Chunk commit enabled: `false`.

The historical suite was not executed: it intentionally retains old artifact
and path contracts. Its collection surface and original bytes remain auditable
through the TRASH manifest.
