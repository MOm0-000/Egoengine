# Quarantined partial Candidate-B seed-0 continuation

This partial run was interrupted after epoch 125 because its epoch-125
checkpoint was written while the extension YAML briefly contained an added
summarizer-hash field. The simulator, optimizer, and policy settings were not
changed, but the checkpoint's bound `extension_contract` SHA-256 therefore did
not match the frozen contract used to launch the run.

The partial run is excluded from all algorithm evidence and comparison. Seed 0
was restarted from the exact verified epoch-62 / 100k checkpoint after the YAML
was restored byte-for-byte. No checkpoint or validation from this directory may
be promoted or selected.
