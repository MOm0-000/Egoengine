# Candidate-C v5 checkpoint-label preflight quarantine

This directory is not algorithmic evidence. The first seed-0 launch stopped
before any actor update because the immutable v4 checkpoint capture helper
accepts only its historical `A`/`B` labels.

The v4 helper was not broadened. The v5 runner now captures Candidate C using
its Candidate-B replay-preserving lineage and relabels the complete payload as
Candidate C before v5 validation and serialization. The fixed fresh runs in
`runs/taco_pour_algorithmic_reproduction_training_v5/` are the only active v5
evidence.

No optimizer update, retry of an algorithmic fail-closed run, tolerance
relaxation, or chunk commit occurred here.
