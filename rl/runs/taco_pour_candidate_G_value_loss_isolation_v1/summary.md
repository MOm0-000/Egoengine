# Candidate G internal value-loss isolation v1

| seed | branch | pre-clip grad L2 | mean exact KL | valid prefix N |
|---:|---|---:|---:|---:|
| 0 | FULL | 83.7770691 | 0.521691169 | 14 |
| 0 | POLICY_ONLY | 41.4496498 | 0.914648219 | 12 |
| 0 | VALUE_ONLY | 72.77845 | 0.0218777282 | 15 |
| 1 | FULL | 127.915001 | 0.509575328 | 14 |
| 1 | POLICY_ONLY | 59.1747131 | 1.02024755 | 14 |
| 1 | VALUE_ONLY | 113.421211 | 0.0226995358 | 15 |
| 2 | FULL | 109.978477 | 0.605544303 | 20 |
| 2 | POLICY_ONLY | 41.7024422 | 1.36955176 | 13 |
| 2 | VALUE_ONLY | 101.776886 | 0.022243082 | 15 |

**结论：已测到 value→policy 耦合，但没有足够证据选择移除内部 value 项。**

- validity: `COMPLETED_BOUNDED_VALUE_LOSS_ISOLATION`
- recommendation: `VALUE_POLICY_COUPLING_PRESENT_BUT_BENEFIT_UNRESOLVED`
- negative evidence: `ISOLATION_NOT_FAVORED_AT_TESTED_UPDATES`
- All three reconstructed epoch-1 batches are bitwise-equal to history; all three FULL shadow updates pass the frozen historical regression.
- Explicit `ppo_flat_index` alignment also reproduces all four independent historical credit fields bitwise for all three seeds.
- The weighted value/policy all-parameter gradient-norm ratios are `1.756 / 1.917 / 2.441`; their cosines are approximately zero (`+0.000634 / -0.000281 / -0.000278`).
- POLICY_ONLY increases mean exact KL in every seed (`0.915 / 1.020 / 1.370`) relative to FULL (`0.522 / 0.510 / 0.606`).
- Closed-loop valid prefixes are FULL `14 / 14 / 20` versus POLICY_ONLY `12 / 14 / 13`; POLICY_ONLY is worse in two seeds and never restores the BASE `20 / 20 / 20` contract across all seeds.
- VALUE_ONLY changes bounded policy means by `0.0535 / 0.0542 / 0.0514`, directly confirming shared-representation coupling at these update points.
- Bounded cost is exactly `960` control intervals / `9,600` physics steps / `9` actor steps / `12` external-critic steps. Report-only recovery repeated no physical interval or optimizer step.
- Candidate G classification unchanged; no long training and no chunk commit authorized.
- Scope is limited to three reconstructed first batches and one paired shadow update per branch.
