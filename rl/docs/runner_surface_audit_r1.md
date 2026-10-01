# Core runner surface audit R1

`core/runner.py` was reviewed after the active test and MJWP-environment
cleanup. No code split was made.

The 442-line file has four contiguous responsibilities that match the bounded
entrypoint:

1. resolve and hash-check the frozen execution config/assets;
2. inspect the environment and boundary schemas;
3. execute the offline actor, closed-loop trajectory and collector parity
   checks;
4. serialize the bounded verification report and parse the two-command CLI.

Hashing and JSON primitives already live in `state_io.py`; model-state hashes
live in `audit.py`; physics construction lives in `env.py`. Moving the remaining
small helpers would make `verify()` cross more modules without removing a
second implementation or shortening its execution order. The current runner
also exposes no train or chunk-commit command.

Decision: retain the file as-is. Revisit only if a second active command creates
duplicated asset resolution or report formatting.
