# Quarantine reason

Rejected pre-update Candidate-G seed-0 attempt.  The immutable shared checkpoint
serializer accepted only its historical candidate enum, so this attempt failed
before any optimizer update.  The corrected runner uses the common serializer
and then binds Candidate-G lineage before hashing.  Nothing in this directory is
eligible for training continuation, comparison, promotion, or chunk commit.
