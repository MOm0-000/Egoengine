# Invalidated derived classification

The raw CPU MuJoCo trajectory is retained for provenance, but this run is not
part of the active evidence chain.  Its derived summary used the 50 µm static
hand/material tolerance for compliant object/table contacts and used the
ambiguous label `proxy_native_mismatch` for a preceding-interval force versus
endpoint-native-surface comparison.

The physical trajectory itself was not used for promotion.  The active v1 run
was regenerated only after separating:

- endpoint proxy/native geometry sign comparison;
- preceding-interval proxy-force/native-clear exposure; and
- the pre-existing 0.5 mm dynamic object/table penetration limit from the
  isolated bowl-settling contract.

